"""Ported from the cartouche_matcher.py __main__ smoke tests."""
from __future__ import annotations

import pytest

from vision.cartouche_matcher import (
    apply_panel_consensus,
    load_royal_names,
    match_cartouche,
    normalize_code,
)


@pytest.mark.parametrize('raw,expected', [
    ('n35', 'N35'),
    ('aa15', 'Aa15'),
    ('Aa1', 'Aa1'),
    ('unknown', 'Unknown'),
    ('cartouche', 'cartouche'),
])
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected


def test_load_royal_names_missing_file(tmp_path):
    assert load_royal_names(tmp_path / 'nope.json') == {}


def test_perfect_wnjs_match(royal_names):
    slots = [
        [('E34', 0.70), ('E1', 0.05)],
        [('N35', 0.52), ('N37', 0.04)],
        [('M17', 0.65), ('F31', 0.55)],
        [('S29', 0.79), ('F31', 0.08)],
    ]
    r = match_cartouche(slots, royal_names)
    assert r is not None and r.name_key == 'wnjs'
    assert r.aligned_codes == ['E34', 'N35', 'M17', 'S29']


def test_missing_first_sign_deletion(royal_names):
    slots = [
        [('N35', 0.52), ('N37', 0.04)],
        [('M17', 0.65), ('F31', 0.55)],
        [('S29', 0.79), ('F31', 0.08)],
    ]
    r = match_cartouche(slots, royal_names)
    assert r is not None and r.name_key == 'wnjs'
    assert r.aligned_codes == ['N35', 'M17', 'S29']


def test_unknown_gap_consumes_missing_sign(royal_names):
    slots = [
        [('Unknown', 0.0)],
        [('N35', 0.52)],
        [('M17', 0.65)],
        [('S29', 0.79)],
    ]
    r = match_cartouche(slots, royal_names)
    assert r is not None and r.name_key == 'wnjs'
    assert r.aligned_codes == ['E34', 'N35', 'M17', 'S29']


def test_spurious_double_box_skipped(royal_names):
    slots = [
        [('N35', 0.52)],
        [('F31', 0.55)],          # double-boxed stroke (spurious)
        [('M17', 0.65)],
        [('S29', 0.79)],
    ]
    r = match_cartouche(slots, royal_names)
    assert r is not None and r.name_key == 'wnjs'
    assert r.aligned_codes[0] == 'N35'
    assert r.aligned_codes[2] == 'M17' and r.aligned_codes[3] == 'S29'


def test_nonsense_refused(royal_names):
    slots = [
        [('D21', 0.9)], [('G17', 0.9)], [('X1', 0.9)],
        [('O1', 0.9)], [('Q3', 0.9)], [('A1', 0.9)],
    ]
    assert match_cartouche(slots, royal_names) is None


def test_empty_slots_refused(royal_names):
    assert match_cartouche([], royal_names) is None


def test_panel_consensus_lifts_compatible_refusals(royal_names):
    # wnjs-only sub-lexicon: with the full 41-king lexicon the "weak"
    # fragment legitimately matches Thutmose III, defeating the scenario
    lex = {'wnjs': royal_names['wnjs']}
    good = [[('N35', 0.52)], [('M17', 0.65)], [('S29', 0.79)]]
    weak = [[('X1', 0.56)], [('S29', 0.60)]]              # compatible partial
    bad = [[('D21', 0.9)], [('G17', 0.9)], [('O1', 0.9)],
           [('Q3', 0.9)], [('A1', 0.9)], [('B1', 0.9)]]
    slots_pc = [good, good, weak, bad]
    results = [match_cartouche(s, lex) for s in slots_pc]
    assert results[0] is not None and results[1] is not None
    assert results[2] is None and results[3] is None

    out = apply_panel_consensus(results, slots_pc, lex)
    assert out[0][1] is False and out[1][1] is False      # originals untouched
    assert out[2][0] is not None and out[2][1] is True    # adopted, inferred
    assert out[2][0].name_key == 'wnjs'
    assert out[3][0] is None                              # bad stays refused


def test_panel_consensus_no_majority_noop(royal_names):
    """A single refused cartouche with no confident sibling stays refused."""
    bad = [[('D21', 0.9)], [('G17', 0.9)], [('O1', 0.9)],
           [('Q3', 0.9)], [('A1', 0.9)], [('B1', 0.9)]]
    results = [match_cartouche(bad, royal_names)]
    out = apply_panel_consensus(results, [bad], royal_names)
    assert out == [(None, False)]
