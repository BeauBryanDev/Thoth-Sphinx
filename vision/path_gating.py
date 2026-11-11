# Post-Viterbi steps of the corrector: the confidence gate, which reverts
# fuzzy corrections YOLO was surer about, and the SegmentedWord assembly.

from __future__ import annotations

import math
from collections import Counter
from operator import attrgetter

from vision.correction_types import SegmentedWord
from vision.edit_costs import EDIT_PENALTY, should_correct
from vision.lexicon_trie import SourceRecord, SphinxTrie
from vision.viterbi_segmenter import viterbi_segment

# One path segment: (codes, trie records, edit distance).
Segment = tuple[list[str], list[SourceRecord], int]

# Confidence given to a slot YOLO left without candidates.
NEUTRAL_CONF = 0.5

# Score penalty for an edit distance EDIT_PENALTY does not list.
UNLISTED_EDIT_PENALTY = -5.0


def top1_confidences(yolo_topk: list[list[tuple[str, float]]]) -> list[float]:
    """YOLO's top-1 confidence per slot."""
    confidences: list[float] = []

    for candidates in yolo_topk:
        if candidates:
            confidences.append(candidates[0][1])
        else:
            confidences.append(NEUTRAL_CONF)

    return confidences


def mean_conf(slot_confs: list[float], 
              cursor: int, 
              seg_len: int
              ) -> float:
    """Mean YOLO confidence over a span of slots."""
    if seg_len <= 0:
        return NEUTRAL_CONF

    return sum(slot_confs[cursor:cursor + seg_len]) / seg_len


def gate_path(
    path: list[Segment],
    best_codes: list[str],
    slot_confs: list[float],
    trie: SphinxTrie,
    log_prob: dict,
    unigrams: Counter,
    conf_threshold: float,
    max_word_len: int,
) -> list[Segment]:
    """Revert fuzzy corrections where YOLO was more reliable than the trie."""
    gated_path: list[Segment] = []
    cursor = 0

    for segment in path:
        seg_codes, _, edit_dist = segment
        seg_len = len(seg_codes)
        confidence = mean_conf(slot_confs, cursor, seg_len)

        if edit_dist > 0 and not should_correct(
            confidence, edit_dist, conf_threshold
        ):
            # Keep YOLO's codes and re-read the span with exact matches only,
            # so real words inside the rejected chunk come back; the rest
            # stays unread.
            exact_path, _ = viterbi_segment(
                best_codes[cursor:cursor + seg_len],
                trie,
                log_prob,
                unigrams,
                max_word_len=max_word_len,
                fuzzy_max_dist=0,
            )
            gated_path.extend(exact_path)
        else:
            gated_path.append(segment)

        cursor += seg_len

    return gated_path


def _to_word(segment: Segment, confidence: float) -> SegmentedWord:
    """The most frequent record names the word; none means a fallback."""
    seg_codes, records, edit_dist = segment

    if not records:
        return SegmentedWord(
            codes=seg_codes,
            translit=' '.join(seg_codes),
            translation='',
            freq=0,
            edit_dist=edit_dist,
            confidence=confidence,
            source='fallback',
        )

    best = max(records, key=attrgetter('freq'))

    return SegmentedWord(
        codes=seg_codes,
        translit=best.translit,
        translation=best.translation,
        freq=best.freq,
        edit_dist=edit_dist,
        confidence=confidence,
        source=best.source,
    )


def segmented_words(
    gated_path: list[Segment],
    slot_confs: list[float],
) -> tuple[list[SegmentedWord], float]:
    """SegmentedWord per path segment, and the total score."""
    words: list[SegmentedWord] = []
    total_score = 0.0
    cursor = 0

    for segment in gated_path:
        seg_len = len(segment[0])
        confidence = mean_conf(slot_confs, cursor, seg_len)
        word = _to_word(segment, round(confidence, 4))

        total_score += math.log(word.freq + 1)
        total_score += EDIT_PENALTY.get(word.edit_dist, UNLISTED_EDIT_PENALTY)

        words.append(word)
        cursor += seg_len

    return words, total_score


def join_translit(words: list[SegmentedWord]) -> str:
    return ' '.join(w.translit for w in words if w.translit)


def join_translation(words: list[SegmentedWord]) -> str:
    """Translations of the words the trie read, fallbacks left out."""
    parts: list[str] = []

    for word in words:
        if word.translation and word.source != 'fallback':
            parts.append(word.translation)

    return ' | '.join(parts)
