"""Ported from the sphinx_corrector.py __main__ smoke tests.

Pure-python contracts run without artifacts; correction tests use the
session-scoped `corrector` fixture (trie + parquet + confusion csv).
"""
from __future__ import annotations

import pytest

from vision.sphinx_corrector import (
    EDIT_PENALTY,
    _compute_edit_penalty,
    correct,
    should_correct,
)


# ---------------------------------------------------------------------------
# Pure contracts (no artifacts)
# ---------------------------------------------------------------------------

def test_edit_penalty_contracts():
    assert _compute_edit_penalty([], [], 0, None) == 0.0
    assert _compute_edit_penalty(['x'], ['y'], 1, None) == EDIT_PENALTY[1]


def test_edit_penalty_length_mismatch_falls_back():
    fake_matrix = {('a', 'b'): 0.5}
    assert _compute_edit_penalty(['a'], ['b', 'c'], 2, fake_matrix) \
        == EDIT_PENALTY[2]


def test_should_correct_gate():
    assert should_correct(0.99, 0) is True          # exact: always accept
    assert should_correct(0.95, 2) is False         # confident YOLO, far edit
    assert should_correct(0.20, 1) is True          # weak YOLO, close edit


# ---------------------------------------------------------------------------
# Sub-cost matrix structure (artifact-backed)
# ---------------------------------------------------------------------------

def test_sub_cost_matrix_structure(corrector):
    _, _, _, sub_cost = corrector
    if sub_cost is None:
        pytest.skip('confusion matrix CSV not loaded')
    diag = [sub_cost[(c, c)] for c in ('g17', 'i9', 'd21', 'm17')
            if (c, c) in sub_cost]
    assert diag and all(c < 2.0 for c in diag)


def test_sub_cost_per_pair_penalty_finite(corrector):
    _, _, _, sub_cost = corrector
    if sub_cost is None:
        pytest.skip('confusion matrix CSV not loaded')
    for o, e in (('g17', 'g18'), ('i9', 'i10'), ('m17', 'g17')):
        if (o, e) in sub_cost:
            pen = _compute_edit_penalty([o], [e], 1, sub_cost)
            assert -10.0 < pen <= 0


# ---------------------------------------------------------------------------
# Correction end-to-end over the real trie + bigram LM
# ---------------------------------------------------------------------------

def test_exact_match_kept(corrector):
    trie, log_prob, unigrams, _ = corrector
    slots = [
        [('G17', 0.97), ('G18', 0.02), ('Unknown', 0.01)],
        [('I9', 0.94), ('I10', 0.05), ('Unknown', 0.01)],
        [('D21', 0.88), ('D22', 0.08), ('D19', 0.04)],
    ]
    r = correct(slots, trie, log_prob, unigrams)
    assert r.flat_corrected_seq == ['G17', 'I9', 'D21']
    assert r.flat_translit                     # non-empty reading


def test_one_wrong_code_corrected(corrector):
    """Low-confidence G18 passes the gate and is corrected to G17."""
    trie, log_prob, unigrams, _ = corrector
    slots = [
        [('G18', 0.30), ('G17', 0.60), ('Unknown', 0.10)],
        [('I9', 0.93), ('I10', 0.05), ('Unknown', 0.02)],
        [('D21', 0.91), ('D22', 0.06), ('D19', 0.03)],
    ]
    r = correct(slots, trie, log_prob, unigrams)
    assert r.flat_corrected_seq == ['G17', 'I9', 'D21']


def test_confident_wrong_code_gated(corrector):
    """At conf 0.45 the gate ((1-0.45)/2 < 0.30) keeps YOLO's G18."""
    trie, log_prob, unigrams, _ = corrector
    slots = [
        [('G18', 0.45), ('G17', 0.40), ('Unknown', 0.15)],
        [('I9', 0.93), ('I10', 0.05), ('Unknown', 0.02)],
        [('D21', 0.91), ('D22', 0.06), ('D19', 0.03)],
    ]
    r = correct(slots, trie, log_prob, unigrams)
    assert r.flat_corrected_seq[0] == 'G18'
    assert r.flat_corrected_seq[1:] == ['I9', 'D21']


def test_unknown_slot_resolution(corrector):
    trie, log_prob, unigrams, _ = corrector
    slots = [
        [('Unknown', 0.60), ('G17', 0.30), ('G18', 0.10)],
        [('N35', 0.95), ('N36', 0.03), ('Z7', 0.02)],
        [('D21', 0.89), ('D22', 0.07), ('D19', 0.04)],
    ]
    r = correct(slots, trie, log_prob, unigrams)
    assert 'Unknown' not in r.flat_corrected_seq
    assert r.unknowns_resolved
    assert all(u.proposed for u in r.unknowns_resolved)


def test_longer_sequence_segments_into_words(corrector):
    trie, log_prob, unigrams, _ = corrector
    slots = [
        [('Q3', 0.92), ('Q1', 0.05), ('Unknown', 0.03)],
        [('X1', 0.97), ('X2', 0.02), ('Unknown', 0.01)],
        [('G17', 0.88), ('G18', 0.08), ('Unknown', 0.04)],
        [('N35', 0.91), ('N36', 0.06), ('Z7', 0.03)],
        [('D21', 0.85), ('D22', 0.10), ('D19', 0.05)],
        [('X1', 0.94), ('X2', 0.04), ('Unknown', 0.02)],
    ]
    r = correct(slots, trie, log_prob, unigrams)
    assert len(r.flat_corrected_seq) == 6
    assert len(r.segmented_words) >= 2
    # segmentation covers the full sequence
    assert sum(len(w.codes) for w in r.segmented_words) == 6


def test_result_contract(corrector):
    trie, log_prob, unigrams, _ = corrector
    slots = [[('G17', 0.97)], [('I9', 0.94)], [('D21', 0.88)]]
    r = correct(slots, trie, log_prob, unigrams)
    assert isinstance(r.flat_corrected_seq, list)
    assert isinstance(r.flat_translit, str)
    assert isinstance(r.flat_translation, str)
    assert isinstance(r.had_fallback, bool)
    assert isinstance(r.score, float)
