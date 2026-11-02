#!/usr/bin/env python3
"""
NLP pre-transformer correction layer for SphinxEyes.

Role in the pipeline
    YOLOv11 (top-3 per slot)
    BigramModel       (Markov chain built from BBAW corpus)
        ViterbiSegmenter  (dynamic programming over SphinxTrie)
        ConfidenceGate    (Naive Bayes: respect high-conf YOLO predictions)
        UnknownResolver   (enumerate trie children at Unknown slots)
        correct()         (single entry point for FastAPI)
        CorrectionResult  (JSON-ready dict for GPT prompt)

Algorithms used
    1. Bigram language model  (Markov chain, order 1)
       Built from BBAW gardiner_seq column.
       P(code_B | code_A) = count(A->B) / count(A)
       Stored as log-probabilities to avoid float underflow.

    2. Viterbi segmenter
       Dynamic programming over the flat YOLO sequence.
       At each position i, tries all segments codes[i:j] (j-i <= MAX_WORD_LEN).
       Scores each candidate segment with:
           freq_score   = log(total_record_freq + 1)
           trans_score  = log P(segment[0] | prev_last_code)   [bigram]
           edit_penalty = 0.0 (exact) | -1.0 (fuzzy dist=1) | -2.5 (dist=2)
       Backtracks from dp[N] to recover the optimal word sequence.
       Fallback: if dp[N] == -inf, segments sign-by-sign (graceful degradation).

    3. Beam search over top-3 YOLO candidates
       At each slot, YOLO supplies up to 3 (code, confidence) pairs.
       The beam keeps the top-B partial sequences, expanding each with all
       3 candidates. At the end, the highest-scoring complete path wins.
       This replaces the single-best assumption and lets the corrector
       recover from YOLO's first-choice errors.

    4. Naive Bayes confidence gate
       Prevents the corrector from overriding a high-confidence YOLO prediction.
       P(correction_correct | yolo_conf, edit_dist) ~ (1-yolo_conf)/(edit_dist+1)
       If this probability is below a threshold, the original YOLO code is kept.

    5. Unknown-slot resolution
       When YOLO outputs 'Unknown' for a slot, the trie enumerates all child
       edges at that depth in surviving beam nodes and proposes the most
       frequent real Gardiner code as a replacement. gardiner_orig on the
       SourceRecord provides the ground-truth code the Unknown was substituted for.

Usage
    from vision.sphinx_corrector import build_bigram_model, correct, CorrectionResult
    from vision.sphinx_trie import SphinxTrie

    trie      = SphinxTrie.from_pickle('sphinx_trie_v4.pkl')
    log_prob, unigrams = build_bigram_model('bbaw_clean.parquet')

    # yolo_topk: list of slots, each slot is list of (code, confidence)
    yolo_topk = [
        [('G17', 0.97), ('G18', 0.02), ('Unknown', 0.01)],
        [('N35', 0.91), ('N36', 0.06), ('Z7',  0.03)],
        [('D21', 0.88), ('D22', 0.08), ('D19', 0.04)],
    ]
    result = correct(yolo_topk, trie, log_prob, unigrams)
    print(result['flat_translit'])
"""
from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# Public re-export so callers only need to import this module

try:
    from vision.sphinx_trie import SphinxTrie, SourceRecord, UNKNOWN_TOKEN
except ImportError:
    # Allow the module to be imported standalone for unit tests
    SphinxTrie     = None       # type: ignore
    SourceRecord   = None       # type: ignore
    UNKNOWN_TOKEN  = 'Unknown'

# Constants

MAX_WORD_LEN    = 12    # max Gardiner codes in a single dictionary entry
BEAM_WIDTH      = 8     # number of partial paths kept during beam search
FUZZY_MAX_DIST  = 2     # max Levenshtein distance for trie fuzzy lookup
NEG_INF         = float('-inf')
YOLO_TRUST_THRESHOLD = 0.75
# Edit penalty table indexed by Levenshtein distance.
# Used when sub_cost_matrix is unavailable, or when the trie's fuzzy hit
# changed segment length (insertions/deletions — no per-pair lookup possible).
EDIT_PENALTY = {0: 0.0, 1: -1.0, 2: -2.5}


def _compute_edit_penalty(
    orig_seg : list[str],
    eff_seg  : list[str],
    edit_dist: int,
    sub_cost : Optional[dict],
) -> float:
    """
    Score contribution (negative, i.e. a penalty) for an edited segment.

    When `sub_cost` is None, or the original and effective lengths differ
    (insert/delete), fall back to the coarse `EDIT_PENALTY` table.

    When lengths match and `sub_cost` is provided, sum per-pair
    `-(-log P(truth|pred))` = `log P(truth|pred)` over the substituted
    positions. Frequently-confused pairs (high P) → small penalty;
    rarely-confused pairs (low P) → large penalty.
    """
    if edit_dist == 0:
        return 0.0
    
    if sub_cost is None or len(orig_seg) != len(eff_seg):
        return EDIT_PENALTY.get(edit_dist, -5.0)
    total = 0.0
    
    for o, e in zip(orig_seg, eff_seg):
        if o == e:
            continue
        
        c = sub_cost.get((o, e))
        
        if c is None:
            total += EDIT_PENALTY.get(1, -1.0)   # per-pair fallback
        else:
            total += -c     
            # cost ≥ 0; penalty ≤ 0
    return total

# Soft linguistic priors added to the Viterbi segment score.
# Tuned conservatively so corpus frequency still dominates; raise these
# if the segmenter under-uses determinatives / particles in practice.
DETERMINATIVE_BONUS = 0.5    # last code of segment is a determinative
PARTICLE_BONUS      = 1.0    # segment exactly matches a phrase-initial particle
LAYOUT_BONUS        = 1.5    # segment ends at an image-derived row/col break
UNREAD_PENALTY      = -0.5   # step over one sign no entry explains (always worse than a real word)

# Lazy-loaded prior artifacts. Populated from determinatives.json and
# initial_particles.json on first use. Set to None to disable.
_DETERMINATIVES_CACHE: Optional[set] = None
_PARTICLES_CACHE: Optional[list] = None
_PRIORS_DIR = Path(__file__).parent

# Data Structures

@dataclass
class SegmentedWord:
    codes       : list[str]
    translit    : str
    translation : str
    freq        : int
    edit_dist   : int               # 0 = exact match, 1-2 = fuzzy corrected
    confidence  : float             # mean YOLO confidence over the slot range
    source      : str               # "dickson" | "bbaw" | "fallback"


@dataclass
class ResolvedUnknown:
    slot        : int               # index in the flat yolo sequence
    proposed    : str               # the Gardiner code we propose
    reason      : str               # human-readable justification
    freq        : int               # corpus frequency of the proposal


@dataclass
class CorrectionResult:
    segmented_words    : list[SegmentedWord]
    unknowns_resolved  : list[ResolvedUnknown]
    flat_corrected_seq : list[str]          # for GPT prompt
    flat_translit      : str                # for GPT prompt
    flat_translation   : str                # for GPT prompt (Dickson English)
    score              : float              # total Viterbi score
    had_fallback       : bool               # True if graceful degradation used

    def to_dict(self) -> dict:
        return {
            'segmented_words'   : [asdict(w) for w in self.segmented_words],
            'unknowns_resolved' : [asdict(u) for u in self.unknowns_resolved],
            'flat_corrected_seq': self.flat_corrected_seq,
            'flat_translit'     : self.flat_translit,
            'flat_translation'  : self.flat_translation,
            'score'             : self.score,
            'had_fallback'      : self.had_fallback,
        }

# Linguistic prior loaders (determinatives, initial particles)


def load_determinatives(path: Optional[Path] = None) -> set:
    """
    Load the set of Gardiner codes flagged as determinatives. Cached.
    Falls back to an empty set if determinatives.json is absent.
    """
    global _DETERMINATIVES_CACHE
    
    if _DETERMINATIVES_CACHE is not None:
        
        return _DETERMINATIVES_CACHE
    
    p = Path(path) if path else _PRIORS_DIR / 'determinatives.json'
    
    if not p.exists():
        
        _DETERMINATIVES_CACHE = set()
        
        return _DETERMINATIVES_CACHE
    
    with open(p, encoding='utf-8') as f:
        
        data = json.load(f)
        
    _DETERMINATIVES_CACHE = set(data.get('codes', {}).keys())
    
    return _DETERMINATIVES_CACHE


def load_initial_particles(path: Optional[Path] = None) -> list:
    """
    Load phrase-initial particles as a list of (gardiner_spelling, meta)
    tuples, sorted by spelling length descending so longer matches are
    tested first. Cached. Returns [] if initial_particles.json is absent.
    """
    global _PARTICLES_CACHE
    
    if _PARTICLES_CACHE is not None:
        
        return _PARTICLES_CACHE
    
    p = Path(path) if path else _PRIORS_DIR / 'initial_particles.json'
    
    if not p.exists():
        
        _PARTICLES_CACHE = []
        
        return _PARTICLES_CACHE
    
    with open(p, encoding='utf-8') as f:
        
        data = json.load(f)
        
    out: list = []
    
    for name, meta in data.get('particles', {}).items():
        
        spellings = [meta.get('gardiner', [])]
        
        spellings.extend(meta.get('alt_spellings', []))
        
        for spelling in spellings:
            
            if spelling:
                
                out.append((tuple(spelling),
                            
                            {'name': name, 'meaning': meta.get('meaning', '')}))
                
    out.sort(key=lambda kv: -len(kv[0]))
    
    _PARTICLES_CACHE = out
    
    return _PARTICLES_CACHE


# Bigram language model  (Markov chain, order 1)


def build_bigram_model(
    bbaw_source,   # str | Path to parquet, OR pandas DataFrame
    gardiner_col: str = 'gardiner_seq',
    smoothing: float = 1e-6,
) -> tuple[dict, Counter]:
    """
    Build a bigram language model from the BBAW corpus.

    Parameters:
    bbaw_source   : path to bbaw_clean.parquet  OR  a pandas DataFrame
    gardiner_col  : column containing space-separated Gardiner codes
    smoothing     : additive (Laplace) smoothing for unseen transitions.
                    Small positive float avoids log(0).

    Returns
    log_prob : dict[str, dict[str, float]]
        log_prob[A][B] = log P(B | A)
    unigrams : Counter
        Raw unigram counts, used for vocabulary size in the Viterbi.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    if isinstance(bbaw_source, (str, Path)):
        df = pd.read_parquet(bbaw_source)
    else:
        df = bbaw_source

    bigrams  = defaultdict(Counter)
    unigrams = Counter()

    for seq_str in df[gardiner_col].dropna():
        
        codes = seq_str.strip().split()
        
        if not codes:
            continue
        for c in codes:
            
            unigrams[c] += 1
            
        for a, b in zip(codes, codes[1:]):
            
            bigrams[a][b] += 1

    # Convert to log-probabilities with smoothing
    vocab_size = len(unigrams)
    log_prob: dict[str, dict[str, float]] = {}

    for a, followers in bigrams.items():
        
        total = sum(followers.values()) + smoothing * vocab_size
        
        log_prob[a] = {
            
            b: math.log((cnt + smoothing) / total)
            
            for b, cnt in followers.items()
        }

    return log_prob, unigrams


def transition_log_prob(
    prev_code : str,
    next_code : str,
    log_prob  : dict,
    unigrams  : Counter,
    smoothing : float = 1e-6,
) -> float:
    """
    Return log P(next_code | prev_code) with fallback to unigram smoothing.
    """
    if prev_code in log_prob and next_code in log_prob[prev_code]:
        
        return log_prob[prev_code][next_code]
    
    # Unigram fallback with smoothing
    vocab_size  = len(unigrams)
    total       = sum(unigrams.values()) + smoothing * vocab_size
    count       = unigrams.get(next_code, 0) + smoothing
    
    return math.log(count / total)



# Viterbi segmenter


def viterbi_segment(
    codes       : list[str],
    trie        : 'SphinxTrie',
    log_prob    : dict,
    unigrams    : Counter,
    sub_cost_matrix : Optional[dict] = None,
    max_word_len    : int = MAX_WORD_LEN,
    fuzzy_max_dist  : int = FUZZY_MAX_DIST,
    boundary_hints  : Optional[list] = None,
    determinatives  : Optional[set] = None,
    particles       : Optional[list] = None,
) -> tuple[list[tuple[list[str], list, int]], bool]:
    """
    Segment a flat sequence of Gardiner codes into the most probable
    sequence of words using dynamic programming over the trie.

    Parameters
    ----------
    codes           : flat list of Gardiner IDs from YOLO
    trie            : SphinxTrie with Dickson + BBAW entries
    log_prob        : bigram log-probability table
    unigrams        : unigram counts (for smoothing)
    sub_cost_matrix : optional dict[(a,b)] -> float substitution cost.
                      If None, uses uniform cost 1.0 for all substitutions.
    max_word_len    : maximum number of signs in a single dictionary word
    fuzzy_max_dist  : maximum Levenshtein distance for fuzzy trie lookup
    boundary_hints  : optional list[int] of slot indices where the image
                      breaks (end of row/column). Ending a segment at one
                      of these positions earns LAYOUT_BONUS.
    determinatives  : optional set of Gardiner codes treated as
                      determinatives. Defaults to lazy-load from
                      determinatives.json.
    particles       : optional list of (gardiner_spelling_tuple, meta)
                      pairs for phrase-initial particles. Defaults to
                      lazy-load from initial_particles.json.

    Returns
    -------
    path        : list of (segment_codes, records, edit_dist)
                  segment_codes is the post-fuzzy-correction segment when
                  a fuzzy hit fired; cursor accounting uses the original
                  span length internally so backtrack stays correct.
    had_fallback: True if graceful sign-by-sign fallback was used
    """
    N = len(codes)
    if N == 0:
        return [], False

    if determinatives is None:
        
        determinatives = load_determinatives()
        
    if particles is None:
        
        particles = load_initial_particles()
        
    boundary_set = set(boundary_hints) if boundary_hints else set()

    # dp[i]   = best total score for the prefix codes[0:i]
    # back[i] = (effective_segment, records, edit_dist, span)
    #   effective_segment : codes used for translit/transition (post-fuzzy)
    #   span              : original-input length consumed (= j-i at insert)
    #   The cursor walks `span`, never `len(effective_segment)`. This is the
    #   fix for the off-by-one when fuzzy returns a different-length match.
    dp   = [NEG_INF] * (N + 1)
    back = [None]    * (N + 1)
    dp[0] = 0.0

    for i in range(N):
        if dp[i] == NEG_INF:
            continue

        # Unread step: pass one sign through with no record, so a single sign the
        # trie can't explain no longer blocks every word after it on the line.
        # It pays the same bigram transition a word would, so a real word always wins.
        unread_trans = (transition_log_prob(back[i][0][-1], codes[i], log_prob, unigrams)
                        if back[i] is not None and back[i][0] else 0.0)
        unread_score = dp[i] + unread_trans + UNREAD_PENALTY
        if unread_score > dp[i + 1]:
            dp[i + 1]   = unread_score
            back[i + 1] = ([codes[i]], [], 0, 1)

        for j in range(i + 1, min(i + max_word_len + 1, N + 1)):
            
            original_segment   = codes[i:j]
            span               = j - i
            edit_dist          = 0
            effective_segment  = original_segment

            #  Exact trie lookup  --
            records = trie.search(original_segment)

            # Fuzzy trie lookup (Levenshtein) 
            if records is None and fuzzy_max_dist > 0:
                
                candidates = trie.fuzzy_search(
                    original_segment,
                    max_distance=fuzzy_max_dist,
                    max_results=1
                )
                if candidates:
                    best              = candidates[0]
                    edit_dist         = best['distance']
                    records           = best['records']
                    effective_segment = best['gardiner_seq'].split()

            if records is None:
                continue

            # --- Score components ---
            total_freq  = sum(r.freq for r in records)
            freq_score  = math.log(total_freq + 1)
            edit_pen    = _compute_edit_penalty(
                original_segment, effective_segment, edit_dist, sub_cost_matrix
            )

            # Bigram transition from previous word's last code
            if back[i] is not None and effective_segment:
                
                prev_eff_seg = back[i][0]
                
                if prev_eff_seg:
                    
                    trans_score = transition_log_prob(
                        prev_eff_seg[-1], effective_segment[0],
                        log_prob, unigrams
                    )
                else:
                    trans_score = 0.0
            else:
                trans_score = 0.0

            # --- Linguistic priors ---
            # Determinative bonus: last sign of segment is a determinative
            det_bonus = (DETERMINATIVE_BONUS
                         if (determinatives
                             and effective_segment
                             and effective_segment[-1] in determinatives)
                         else 0.0)

            # Particle bonus: this exact span matches a phrase-initial
            # particle's Gardiner spelling
            part_bonus = 0.0
            if particles:
                
                seg_tuple = tuple(original_segment)
                
                for spelling, _meta in particles:
                    
                    if len(spelling) == span and seg_tuple == spelling:
                        
                        part_bonus = PARTICLE_BONUS
                        break

            # Layout bonus: segment ends at an image-derived break
            layout_bonus = LAYOUT_BONUS if j in boundary_set else 0.0

            score = (dp[i] + freq_score + trans_score + edit_pen
                     + det_bonus + part_bonus + layout_bonus)

            if score > dp[j]:
                dp[j]   = score
                back[j] = (effective_segment, records, edit_dist, span)

    # --- Reconstruct path ---
    if dp[N] == NEG_INF:
        # Graceful fallback: one sign per segment, no trie records
        fallback = [([c], [], 0) for c in codes]
        return fallback, True

    path, pos = [], N
    while pos > 0 and back[pos] is not None:
        _eff_seg, recs, ed, span = back[pos]
        # Emit ORIGINAL codes consumed (the YOLO span), so downstream
        # cursor accounting in correct() stays simple. The trie's corrected
        # reading is conveyed via records (record.translit, record.gardiner_orig).
        original_seg = codes[pos - span:pos]
        path.append((original_seg, recs, ed))
        pos -= span

    # Handle any remaining prefix not covered (partial match)
    if pos > 0:
        for c in reversed(codes[:pos]):
            path.append(([c], [], 0))

    path.reverse()
    # Unread steps always reach the end now, so "fallback" means no word was read at all.
    return path, not any(recs for _, recs, _ in path)



#  Beam search over top-3 YOLO candidates


@dataclass
class BeamState:
    codes       : list[str]     = field(default_factory=list)
    confs       : list[float]   = field(default_factory=list)
    score       : float         = 0.0

    def extend(self, code: str, conf: float, trans_score: float) -> 'BeamState':
        return BeamState(
            codes  = self.codes  + [code],
            confs  = self.confs  + [conf],
            score  = self.score + math.log(conf + 1e-9) + trans_score,
        )


def beam_decode_topk(
    slots     : list[list[tuple[str, float]]],
    log_prob  : dict,
    unigrams  : Counter,
    beam_width: int = BEAM_WIDTH,
    trust_threshold:  float = YOLO_TRUST_THRESHOLD,
) ->  tuple[list[str], list[float]]:
    """
    Given a list of slots where each slot is [(code, conf), ...],
    return the single best flat code sequence using beam search
    with bigram language model scoring.

    Parameters
    ----------
    slots      : list of slots, each slot is a list of (code, confidence)
                 sorted by confidence descending (as YOLO produces them)
    log_prob   : bigram log-prob table
    unigrams   : unigram counts
    beam_width : number of partial hypotheses to keep at each step

    Returns
    -------
    best_codes : list[str]  the single best code sequence
    """
    if not slots:
        return [], []

    # Initialize beam with empty state
    beam: list[BeamState] = [BeamState()]

    for slot_candidates in slots:
        new_beam: list[BeamState] = []

        for state in beam:
            for code, conf in slot_candidates:
                
                trans = (transition_log_prob(state.codes[-1], code, log_prob, unigrams) if state.codes else 0.0)
                
                new_beam.append( state.extend(code, conf, trans) )
        
        new_beam.sort(key=lambda s: s.score, reverse=True)

        beam = new_beam[:beam_width]

    best = beam[0] if beam else BeamState()

    # Compare best codes to YOLO's top-1 per slot and revert if YOLO was more confident
    overridden = []
    
    for i, chosen in enumerate(best.codes):
        
        top1_code, top1_conf = slots[i][0]
        
        deviated = (chosen != top1_code)
        
        if deviated and top1_conf >= trust_threshold:
            
            best.codes[i] = top1_code     # revertir: YOLO manda
            overridden.append(False)
        else:
            overridden.append(deviated)

    return best.codes, overridden

# 4. Naive Bayes confidence gate


def should_correct(
    yolo_conf   : float,
    edit_dist   : int,
    threshold   : float = 0.30,
) -> bool:
    """
    Decide whether to apply the trie correction or keep the YOLO prediction.

    Uses a simple Bayesian estimate:
        P(correction_correct | yolo_conf, edit_dist)
            ~ (1 - yolo_conf) / (edit_dist + 1)

    If this probability is below `threshold`, the original YOLO code is kept
    (YOLO is probably right and the trie correction is risky).

    Parameters
    ----------
    yolo_conf  : YOLO's confidence for this prediction (0.0 to 1.0)
    edit_dist  : Levenshtein distance of the trie correction (0 = exact)
    threshold  : minimum probability required to apply correction

    Returns
    -------
    True  = apply the trie correction
    False = keep the original YOLO prediction
    """
    if edit_dist == 0:
        return True     # exact match: always accept
    p_correction = (1.0 - yolo_conf) / (edit_dist + 1)
    return p_correction >= threshold


# 5. Unknown-slot resolution

def resolve_unknowns(
    path       : list[tuple[list[str], list, int]],
    trie       : 'SphinxTrie',
    yolo_topk  : list[ list[ tuple[ str,  float]]],
    slot_offset: int = 0,
    trust_threshold: float = YOLO_TRUST_THRESHOLD,
) -> list[ResolvedUnknown]:
    """
    For each segment that contains the UNKNOWN_TOKEN edge, enumerate the
    trie's terminal records to find the most frequent real Gardiner code
    that was originally substituted.

    Uses the gardiner_orig field stored on SourceRecord during ingestion
    with oov_strategy='unknown'.

    Parameters

    path        : output of viterbi_segment
    trie        : SphinxTrie (needed for context; records are already in path)
    slot_offset : global offset of this path in the full YOLO sequence

    Returns

    list of ResolvedUnknown, one per Unknown slot that could be resolved
    """
    resolved = []
    current_slot = slot_offset

    for seg_codes, records, _ in path:

        for idx, code in enumerate(seg_codes):

            if code != UNKNOWN_TOKEN or not records:
                continue

            global_idx = current_slot + idx
            slot_candidates = (yolo_topk[global_idx]
                               if global_idx < len(yolo_topk) else [])
            yolo_alt_codes = {c for c, _ in slot_candidates
                              if c != UNKNOWN_TOKEN}

            # Tally the ground-truth codes this Unknown stood in for.
            orig_counter: Counter = Counter()
            for rec in records:
                if rec.gardiner_orig:
                    orig_codes = rec.gardiner_orig.split()
                    if idx < len(orig_codes):
                        orig_counter[orig_codes[idx]] += rec.freq

            if not orig_counter:
                continue

            overlap = [c for c in orig_counter if c in yolo_alt_codes]

            if overlap:
                # Best case: corpus proposal is also one of YOLO's top-3.
                best_code = max(overlap, key=lambda c: orig_counter[c])
                best_freq = orig_counter[best_code]
                reason = (f"corpus proposal '{best_code}' confirmed by a "
                          f"YOLO top-3 candidate at this slot")
            else:
                # No overlap: prefer a decent YOLO alternative, else corpus.
                decent_yolo_alt = next(
                    (c for c, conf in slot_candidates
                     if c != UNKNOWN_TOKEN and conf >= 0.2),
                    None,
                )
                if decent_yolo_alt is not None:
                    best_code = decent_yolo_alt
                    best_freq = 0
                    reason = (f"no corpus/YOLO overlap; falling back to "
                              f"YOLO's own alternative '{decent_yolo_alt}'")
                else:
                    best_code, best_freq = orig_counter.most_common(1)[0]
                    reason = (f"no usable YOLO alternative; corpus frequency "
                              f"proposal '{best_code}' (freq={best_freq})")

            resolved.append(ResolvedUnknown(
                slot=global_idx, proposed=best_code,
                reason=reason, freq=best_freq,
            ))

        current_slot += len(seg_codes)

    return resolved
                

# 6. correct() — single entry point for FastAPI

def correct(
    yolo_topk       : list[list[tuple[str, float]]],
    trie            : 'SphinxTrie',
    log_prob        : dict,
    unigrams        : Counter,
    sub_cost_matrix : Optional[dict] = None,
    conf_threshold  : float = 0.30,
    beam_width      : int   = BEAM_WIDTH,
    max_word_len    : int   = MAX_WORD_LEN,
    fuzzy_max_dist  : int   = FUZZY_MAX_DIST,
    boundary_hints  : Optional[list] = None,
) -> CorrectionResult:
    """
    Main entry point. Takes YOLO's top-K predictions per slot and returns
    a fully corrected, segmented, transliterated result ready for GPT.

    Parameters
    ----------
    yolo_topk       : list of slots. Each slot is a list of (code, confidence)
                      tuples sorted by confidence descending.
                      Minimum: 1 tuple per slot. Maximum recommended: 3.

    trie            : SphinxTrie loaded from sphinx_trie_v4.pkl
    log_prob        : bigram log-prob table from build_bigram_model()
    unigrams        : unigram counter from build_bigram_model()
    sub_cost_matrix : optional dict[(a,b)] -> float.
                      If provided, overrides uniform substitution cost.
                      Load from confusion_matrix.csv after V2 training.
    conf_threshold  : Naive Bayes threshold (default 0.30).
                      Lower = more aggressive correction.
                      Higher = more conservative, trust YOLO more.
    beam_width      : beam search width (default 8)
    max_word_len    : max signs per dictionary entry (default 12)
    fuzzy_max_dist  : max Levenshtein distance for trie fuzzy lookup (default 2)

    Returns
    -------
    CorrectionResult dataclass. Call .to_dict() for JSON serialization.
    """
    if not yolo_topk:
        return CorrectionResult(
            segmented_words    = [],
            unknowns_resolved  = [],
            flat_corrected_seq = [],
            flat_translit      = '',
            flat_translation   = '',
            score              = NEG_INF,
            had_fallback       = False,
        )

    #   Beam search over top-3 candidates to get best flat sequence
 
    best_codes, beam_overrides = beam_decode_topk(
        yolo_topk, log_prob, unigrams, beam_width=beam_width
    )

    # Mean confidence per slot (for the confidence gate)
    slot_confs = [
        candidates[0][1] if candidates else 0.5
        for candidates in yolo_topk
    ]
    #  Viterbi segmentation over best_codes
 
    path, had_fallback = viterbi_segment(
        best_codes, trie, log_prob, unigrams,
        sub_cost_matrix = sub_cost_matrix,
        max_word_len    = max_word_len,
        fuzzy_max_dist  = fuzzy_max_dist,
        boundary_hints  = boundary_hints,
    )

 
    # Apply confidence gate — revert corrections where YOLO was
    # more reliable than the trie correction
 
    gated_path = []
    cursor     = 0
    for seg_codes, records, edit_dist in path:
        seg_len    = len(seg_codes)
        mean_conf  = (
            sum(slot_confs[cursor:cursor + seg_len]) / seg_len
            if seg_len > 0 else 0.5
        )

        if edit_dist > 0 and not should_correct(mean_conf, edit_dist, conf_threshold):
            # Revert: keep original YOLO codes for this segment
            orig_codes = best_codes[cursor:cursor + seg_len]
            gated_path.append((orig_codes, [], 0))
        else:
            gated_path.append((seg_codes, records, edit_dist))

        cursor += seg_len
 
    #  Resolve Unknown slots
 
    unknowns_resolved = resolve_unknowns(
    gated_path, trie, yolo_topk, slot_offset=0
    )

    # Apply Unknown resolutions to the corrected sequence
    flat_corrected = []
    for seg_codes, records, _ in gated_path:
        flat_corrected.extend(seg_codes)

    for resolution in unknowns_resolved:
        if resolution.slot < len(flat_corrected):
            flat_corrected[resolution.slot] = resolution.proposed

 
    #  Build SegmentedWord list and flat strings
 
    segmented_words : list[SegmentedWord] = []
    total_score      = 0.0
    cursor           = 0

    for seg_codes, records, edit_dist in gated_path:
        seg_len   = len(seg_codes)
        mean_conf = (
            sum(slot_confs[cursor:cursor + seg_len]) / seg_len
            if seg_len > 0 else 0.5
        )

        # Pick best record by frequency
        if records:
            best_rec = max(records, key=lambda r: r.freq)
            translit    = best_rec.translit
            translation = best_rec.translation
            source      = best_rec.source
            freq        = best_rec.freq
        else:
            translit    = ' '.join(seg_codes)
            translation = ''
            source      = 'fallback'
            freq        = 0

        total_score += math.log(freq + 1) + EDIT_PENALTY.get(edit_dist, -5.0)

        segmented_words.append(SegmentedWord(
            codes       = seg_codes,
            translit    = translit,
            translation = translation,
            freq        = freq,
            edit_dist   = edit_dist,
            confidence  = round(mean_conf, 4),
            source      = source,
        ))

        cursor += seg_len

    flat_translit   = ' '.join(w.translit    for w in segmented_words if w.translit)
    flat_translation = ' | '.join(
        w.translation for w in segmented_words
        if w.translation and w.source != 'fallback'
    )

    return CorrectionResult(
        segmented_words    = segmented_words,
        unknowns_resolved  = unknowns_resolved,
        flat_corrected_seq = flat_corrected,
        flat_translit      = flat_translit,
        flat_translation   = flat_translation,
        score              = total_score,
        had_fallback       = had_fallback,
    )


# ---------------------------------------------------------------------------
# Convenience: load artifacts from disk
# ---------------------------------------------------------------------------

def load_corrector(
    trie_pkl     : str | Path,
    bbaw_parquet : str | Path,
    confusion_csv: Optional[str | Path] = None,
    gardiner_col : str = 'gardiner_seq',
) -> tuple['SphinxTrie', dict, Counter, Optional[dict]]:
    """
    Load every artifact needed by correct() in one call.

    Parameters
    ----------
    trie_pkl      : path to sphinx_trie_v4.pkl
    bbaw_parquet  : path to bbaw_clean.parquet (for the bigram LM)
    confusion_csv : optional path to a YOLO confusion-matrix CSV
                    (e.g. confusion_matrix_v2_on_v3val_normalized.csv).
                    When provided, a per-pair substitution-cost dict is
                    built and returned; otherwise the 4th return is None
                    and the corrector falls back to coarse EDIT_PENALTY.
    gardiner_col  : column name inside bbaw_parquet

    Returns
    -------
    trie     : SphinxTrie
    log_prob : bigram log-probability table
    unigrams : unigram Counter
    sub_cost : dict[(pred, truth) -> -log P(truth|pred)] or None
    """
    if SphinxTrie is None:
        raise ImportError("sphinx_trie.py must be on the Python path")

    trie = SphinxTrie.from_pickle(trie_pkl)
    log_prob, unigrams = build_bigram_model(bbaw_parquet, gardiner_col)
    sub_cost = load_sub_cost_matrix(confusion_csv) if confusion_csv else None
    return trie, log_prob, unigrams, sub_cost


# ---------------------------------------------------------------------------
# Cost matrix loader (stub — activate after V2 training)
# ---------------------------------------------------------------------------

def load_sub_cost_matrix(
    confusion_csv: str | Path,
    classes      : Optional[list[str]] = None,
    epsilon      : float = 1e-4,
) -> dict:
    """
    Build a per-pair substitution cost dict from a YOLO confusion matrix CSV.

    CSV orientation (Ultralytics `ConfusionMatrix.matrix` convention, after
    the V3 notebook's `axis=1` row-normalization):
        rows    = predicted class
        columns = truth class
        cell[pred, truth] = P(truth | pred)

    The extra `background` row/column that Ultralytics adds (index nc) is
    dropped — substituting against background is never a valid Gardiner edit.

    Cost convention
    ---------------
        cost(pred, truth) = -log( P(truth | pred) + epsilon )    # ≥ 0
            ~0  for frequently-confused pairs   (cheap to substitute)
            ~9.2 for never-observed pairs       (expensive to substitute,
                                                 epsilon=1e-4 caps -log here)

    The caller's score adds `-cost` so the penalty contribution is ≤ 0,
    matching the sign convention of `EDIT_PENALTY`. See `_compute_edit_penalty`.

    Parameters
    ----------
    confusion_csv : path to e.g. `confusion_matrix_v2_on_v3val_normalized.csv`
    classes       : optional whitelist of Gardiner codes to keep. If None,
                    keeps everything in the CSV except `background`.
    epsilon       : Laplace floor (1e-4 → -log caps near 9.2 instead of inf)

    Returns
    -------
    dict[(pred_code, truth_code)] -> float cost (non-negative).
    Pairs absent from the matrix are silently absent from the dict;
    `_compute_edit_penalty` then falls back to `EDIT_PENALTY[1]`.

    Usage
    -----
        sub_cost = load_sub_cost_matrix(
            'confusion_matrix_v2_on_v3val_normalized.csv'
        )
        result = correct(yolo_topk, trie, log_prob, unigrams,
                         sub_cost_matrix=sub_cost)
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    df = pd.read_csv(confusion_csv, index_col=0)

    # Drop Ultralytics' nc+1 "background" axis on both ends if present.
    for label in ('background', 'Background'):
        if label in df.index:
            df = df.drop(index=label)
        if label in df.columns:
            df = df.drop(columns=label)

    # Optional whitelist
    if classes is not None:
        keep_idx = [c for c in classes if c in df.index]
        keep_col = [c for c in classes if c in df.columns]
        df = df.loc[keep_idx, keep_col]

    # The CSV is already row-normalized (per V3 notebook cell 121:
    # `cm_norm = cm / cm.sum(axis=1)`). Read rows directly as P(truth|pred).
    cost = {}
    rows, cols = list(df.index), list(df.columns)
    mat = df.values.astype(float)
    for i, pred in enumerate(rows):
        for j, truth in enumerate(cols):
            p = float(mat[i, j])
            cost[(pred, truth)] = -math.log(p + epsilon)
            
    return cost

