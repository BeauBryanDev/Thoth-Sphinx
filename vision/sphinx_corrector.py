# NLP correction layer entry point: beam search over the YOLO top-3, Viterbi
# segmentation over the trie, a confidence gate and Unknown resolution.
# The pieces live in helper modules.
#
#   correction_types     SegmentedWord, ResolvedUnknown, CorrectionResult
#   edit_costs           edit penalties, confusion matrix, confidence gate
#   language_model       bigram model from the BBAW corpus
#   linguistic_priors    determinatives, particles, curated trie additions
#   viterbi_segmenter    dynamic programming over the trie
#   beam_decoder         beam search over the top-3 candidates
#   path_gating          confidence gate and word assembly
#   unknown_resolution   proposals for Unknown slots

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from vision.beam_decoder import (
    BEAM_WIDTH,
    YOLO_TRUST_THRESHOLD,
    BeamState,
    beam_decode_topk,
)
from vision.correction_types import (
    CorrectionResult,
    ResolvedUnknown,
    SegmentedWord,
)
from vision.edit_costs import (
    EDIT_PENALTY,
    compute_edit_penalty,
    load_sub_cost_matrix,
    should_correct,
)
from vision.language_model import build_bigram_model, transition_log_prob
from vision.linguistic_priors import (
    DETERMINATIVE_BONUS,
    LAYOUT_BONUS,
    PARTICLE_BONUS,
    apply_trie_additions,
    load_determinatives,
    load_initial_particles,
)
from vision.path_gating import (
    Segment,
    gate_path,
    join_translation,
    join_translit,
    segmented_words,
    top1_confidences,
)
from vision.lexicon_trie import UNKNOWN_TOKEN, SourceRecord, SphinxTrie
from vision.unknown_resolution import resolve_unknowns
from vision.viterbi_segmenter import (
    FULL_FUZZY_LEN,
    FUZZY_MAX_DIST,
    MAX_WORD_LEN,
    MIN_FUZZY_LEN,
    NEG_INF,
    UNREAD_PENALTY,
    fuzzy_distance,
    viterbi_segment,
)

# Old private name, still imported by tests.
_compute_edit_penalty = compute_edit_penalty

__all__ = [
    'BEAM_WIDTH',
    'DETERMINATIVE_BONUS',
    'EDIT_PENALTY',
    'FULL_FUZZY_LEN',
    'FUZZY_MAX_DIST',
    'LAYOUT_BONUS',
    'MAX_WORD_LEN',
    'MIN_FUZZY_LEN',
    'NEG_INF',
    'PARTICLE_BONUS',
    'UNKNOWN_TOKEN',
    'UNREAD_PENALTY',
    'YOLO_TRUST_THRESHOLD',
    'BeamState',
    'CorrectionResult',
    'ResolvedUnknown',
    'SegmentedWord',
    'SourceRecord',
    'SphinxTrie',
    'apply_trie_additions',
    'beam_decode_topk',
    'build_bigram_model',
    'correct',
    'fuzzy_distance',
    'load_corrector',
    'load_determinatives',
    'load_initial_particles',
    'load_sub_cost_matrix',
    'resolve_unknowns',
    'should_correct',
    'transition_log_prob',
    'viterbi_segment',
]


def _empty_result() -> CorrectionResult:
    # NOTE: score is -inf here, which strict JSON cannot carry. It has
    # never shown up in testing, so it is left as is.
    return CorrectionResult(
        segmented_words=[],
        unknowns_resolved=[],
        flat_corrected_seq=[],
        flat_translit='',
        flat_translation='',
        score=NEG_INF,
        had_fallback=False,
    )


def _flat_codes(
    gated_path: list[Segment],
    unknowns_resolved: list[ResolvedUnknown],
) -> list[str]:
    """The path's codes in order, resolved Unknowns swapped in."""
    flat: list[str] = []
    for seg_codes, _, _ in gated_path:
        flat.extend(seg_codes)

    for resolution in unknowns_resolved:
        if resolution.slot < len(flat):
            flat[resolution.slot] = resolution.proposed

    return flat


def correct(
    yolo_topk: list[list[tuple[str, float]]],
    trie: SphinxTrie,
    log_prob: dict,
    unigrams: Counter,
    sub_cost_matrix: Optional[dict] = None,
    conf_threshold: float = 0.30,
    beam_width: int = BEAM_WIDTH,
    max_word_len: int = MAX_WORD_LEN,
    fuzzy_max_dist: int = FUZZY_MAX_DIST,
    boundary_hints: Optional[list] = None,
) -> CorrectionResult:
    """Correct, segment and transliterate YOLO's top-K codes per slot.
    Each slot is [(code, conf), ...], best first; max 3 recommended."""
    if not yolo_topk:
        return _empty_result()

    # The beam's overrides are not used: the gate reads top-1 confidence.
    best_codes, _ = beam_decode_topk(
        yolo_topk, log_prob, unigrams, beam_width=beam_width
    )
    slot_confs = top1_confidences(yolo_topk)

    path, had_fallback = viterbi_segment(
        best_codes,
        trie,
        log_prob,
        unigrams,
        sub_cost_matrix=sub_cost_matrix,
        max_word_len=max_word_len,
        fuzzy_max_dist=fuzzy_max_dist,
        boundary_hints=boundary_hints,
    )

    gated_path = gate_path(
        path,
        best_codes,
        slot_confs,
        trie,
        log_prob,
        unigrams,
        conf_threshold,
        max_word_len,
    )

    unknowns_resolved = resolve_unknowns(
        gated_path, trie, yolo_topk, slot_offset=0
    )
    words, total_score = segmented_words(gated_path, slot_confs)

    return CorrectionResult(
        segmented_words=words,
        unknowns_resolved=unknowns_resolved,
        flat_corrected_seq=_flat_codes(gated_path, unknowns_resolved),
        flat_translit=join_translit(words),
        flat_translation=join_translation(words),
        score=total_score,
        had_fallback=had_fallback,
    )


def load_corrector(
    trie_pkl: str | Path,
    bbaw_parquet: str | Path,
    confusion_csv: Optional[str | Path] = None,
    gardiner_col: str = 'gardiner_seq',
) -> tuple[SphinxTrie, dict, Counter, Optional[dict]]:
    """Load the trie, bigram model and cost matrix that correct() needs.
    Without confusion_csv the 4th value is None (coarse EDIT_PENALTY)."""
    trie = SphinxTrie.from_pickle(trie_pkl)
    apply_trie_additions(trie)
    log_prob, unigrams = build_bigram_model(bbaw_parquet, gardiner_col)

    sub_cost = None
    if confusion_csv:
        sub_cost = load_sub_cost_matrix(confusion_csv)

    return trie, log_prob, unigrams, sub_cost
