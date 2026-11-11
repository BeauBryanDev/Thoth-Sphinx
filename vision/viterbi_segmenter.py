# Viterbi segmenter: split a flat Gardiner sequence into the most probable
# words by dynamic programming over the trie. Moved verbatim from
# sphinx_corrector.py; the scoring is fragile, change it only with tests.

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from vision.edit_costs import compute_edit_penalty
from vision.language_model import transition_log_prob
from vision.linguistic_priors import (
    DETERMINATIVE_BONUS,
    LAYOUT_BONUS,
    PARTICLE_BONUS,
    load_determinatives,
    load_initial_particles,
)
from vision.lexicon_trie import SphinxTrie

# Maximum Gardiner codes in a single dictionary entry.
MAX_WORD_LEN = 12

# Maximum Levenshtein distance for the trie fuzzy lookup.
FUZZY_MAX_DIST = 2

# Shorter segments only match exactly: 2 edits can turn them into anything.
MIN_FUZZY_LEN = 4

# Segments of 4 or 5 signs get at most 1 edit; from 6 signs, FUZZY_MAX_DIST.
FULL_FUZZY_LEN = 6

NEG_INF = float('-inf')

# Step over one sign no entry explains. Always worse than a real word.
UNREAD_PENALTY = -0.5


def fuzzy_distance(span: int, max_dist: int) -> int:
    """Edits a fuzzy match may use on a segment of span signs."""
    if span < MIN_FUZZY_LEN:
        return 0

    if span < FULL_FUZZY_LEN:
        return min(1, max_dist)

    return max_dist


def viterbi_segment(
    codes: list[str],
    trie: SphinxTrie,
    log_prob: dict,
    unigrams: Counter,
    sub_cost_matrix: Optional[dict] = None,
    max_word_len: int = MAX_WORD_LEN,
    fuzzy_max_dist: int = FUZZY_MAX_DIST,
    boundary_hints: Optional[list] = None,
    determinatives: Optional[set] = None,
    particles: Optional[list] = None,
) -> tuple[list[tuple[list[str], list, int]], bool]:
    """Best word path as (segment_codes, records, edit_dist), plus a flag.
    determinatives and particles default to the lazy-loaded JSON priors."""
    N = len(codes)
    if N == 0:
        return [], False

    if determinatives is None:
        determinatives = load_determinatives()

    if particles is None:
        particles = load_initial_particles()

    boundary_set = set(boundary_hints) if boundary_hints else set()

    # dp[i] is the best total score for the prefix codes[0:i].
    # back[i] is (effective_segment, records, edit_dist, span): the codes used
    # for translit and transition (after fuzzy), and the input length consumed.
    # The cursor walks span, never len(effective_segment): a fuzzy hit may
    # have a different length, and walking it caused an off-by-one.
    dp = [NEG_INF] * (N + 1)
    back = [None] * (N + 1)
    dp[0] = 0.0

    for i in range(N):
        if dp[i] == NEG_INF:
            continue

        # Unread step: pass one sign through with no record, so a single sign
        # the trie cannot explain does not block every word after it. It pays
        # the same bigram transition a word would, so a real word always wins.
        unread_trans = 0.0
        if back[i] is not None and back[i][0]:
            unread_trans = transition_log_prob(
                back[i][0][-1], codes[i], log_prob, unigrams
            )

        unread_score = dp[i] + unread_trans + UNREAD_PENALTY
        if unread_score > dp[i + 1]:
            dp[i + 1] = unread_score
            back[i + 1] = ([codes[i]], [], 0, 1)

        for j in range(i + 1, min(i + max_word_len + 1, N + 1)):
            original_segment = codes[i:j]
            span = j - i
            edit_dist = 0
            effective_segment = original_segment

            records = trie.search(original_segment)

            # Fuzzy lookup: the allowed edits grow with the segment length.
            allowed_dist = fuzzy_distance(span, fuzzy_max_dist)
            if records is None and allowed_dist > 0:
                candidates = trie.fuzzy_search(
                    original_segment,
                    max_distance=allowed_dist,
                    max_results=1,
                )
                if candidates:
                    best = candidates[0]
                    edit_dist = best['distance']
                    records = best['records']
                    effective_segment = best['gardiner_seq'].split()

            if records is None:
                continue

            total_freq = sum(r.freq for r in records)
            freq_score = math.log(total_freq + 1)
            edit_pen = compute_edit_penalty(
                original_segment, effective_segment, edit_dist, sub_cost_matrix
            )

            # Bigram transition from the previous word's last code.
            trans_score = 0.0
            if back[i] is not None and effective_segment:
                prev_eff_seg = back[i][0]
                if prev_eff_seg:
                    trans_score = transition_log_prob(
                        prev_eff_seg[-1], effective_segment[0],
                        log_prob, unigrams
                    )

            # Determinative bonus: the last sign of the segment is one.
            det_bonus = 0.0
            if determinatives and effective_segment:
                if effective_segment[-1] in determinatives:
                    det_bonus = DETERMINATIVE_BONUS

            # Particle bonus: the span is a phrase-initial particle spelling.
            part_bonus = 0.0
            if particles:
                seg_tuple = tuple(original_segment)
                for spelling, _meta in particles:
                    if len(spelling) == span and seg_tuple == spelling:
                        part_bonus = PARTICLE_BONUS
                        break

            # Layout bonus: the segment ends at an image-derived break.
            layout_bonus = LAYOUT_BONUS if j in boundary_set else 0.0

            score = (dp[i] + freq_score + trans_score + edit_pen
                     + det_bonus + part_bonus + layout_bonus)

            if score > dp[j]:
                dp[j] = score
                back[j] = (effective_segment, records, edit_dist, span)

    # Graceful fallback: one sign per segment, no trie records.
    if dp[N] == NEG_INF:
        fallback = [([c], [], 0) for c in codes]
        return fallback, True

    path, pos = [], N
    while pos > 0 and back[pos] is not None:
        _eff_seg, recs, ed, span = back[pos]
        # Emit the ORIGINAL codes consumed (the YOLO span), so the cursor
        # accounting in correct() stays simple. The corrected reading is
        # carried by the records (translit, gardiner_orig).
        original_seg = codes[pos - span:pos]
        path.append((original_seg, recs, ed))
        pos -= span

    # Any remaining prefix not covered (partial match).
    if pos > 0:
        for c in reversed(codes[:pos]):
            path.append(([c], [], 0))

    path.reverse()

    # Unread steps always reach the end, so fallback means no word was read.
    return path, not any(recs for _, recs, _ in path)
