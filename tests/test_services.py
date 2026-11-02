"""Service-layer tests: LLM degradation paths + pure helpers. No network."""
from __future__ import annotations

import pytest

from app.schemas.transliterations import TextContext
from app.services.reverse import ReverseTranslationService, _clean_codes
from app.services.sphinx_chat import ChatService
from app.services.sphinx_inference import InferenceService
from app.services.transliteration_services import (
    TransliterationService,
    build_egyptologist_prompt,
    segment_into_chunks,
)
from tests.conftest import StubPipeline, make_raw_result


# ---------------------------------------------------------------------------
# Graceful degradation without ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

def test_transliteration_disabled_returns_error(no_api_key):
    svc = TransliterationService()
    assert not svc.enabled
    out = svc.transliterate_sequence(
        ['G17'], [0.9], [], [],
        direction='rtl', layout='columns', ctx=TextContext(),
    )
    assert out.error and 'ANTHROPIC_API_KEY' in out.error
    assert out.chunks == [] and out.n_chunks == 0


def test_reverse_disabled_returns_error(no_api_key):
    svc = ReverseTranslationService()
    assert not svc.enabled
    out = svc.translate('hello world')
    assert out.error and out.gardiner_codes == []


def test_chat_disabled_raises(no_api_key):
    """Chat's contract is distinct: it RAISES instead of returning error."""
    svc = ChatService()
    assert not svc.enabled
    with pytest.raises(RuntimeError):
        svc.chat('hello', [])


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_segment_into_chunks_cuts_at_boundaries():
    codes = [f'c{i}' for i in range(6)]
    confs = [0.5] * 6
    chunks = segment_into_chunks(codes, confs, boundary_hints=[2, 6])
    assert [len(c) for c in chunks] == [2, 4]
    assert chunks[0][0] == ('c0', 0.5)


def test_segment_into_chunks_splits_long_lines():
    codes = [f'c{i}' for i in range(25)]
    chunks = segment_into_chunks(codes, [0.5] * 25, [], max_signs=20)
    assert [len(c) for c in chunks] == [20, 5]


def test_segment_into_chunks_ignores_bad_hints():
    chunks = segment_into_chunks(['a', 'b'], [0.5, 0.5],
                                 boundary_hints=[0, 99])
    assert [len(c) for c in chunks] == [2]


def test_build_prompt_includes_cartouche_anchor_and_context():
    prompt = build_egyptologist_prompt(
        ['G17', 'N35'], [0.9, 0.3],
        TextContext(period='new_kingdom', site='Karnak'),
        direction='rtl', layout='columns',
        cartouche_names=['mn-xpr-ra — Thutmose III'],
        previous_context='previous line here',
        chunk_info='chunk 2/3',
    )
    assert 'mn-xpr-ra — Thutmose III' in prompt
    assert 'G17(HIGH:0.90)' in prompt and 'N35(LOW:0.30)' in prompt
    assert 'new kingdom' in prompt and 'Karnak' in prompt
    assert 'previous line here' in prompt


def test_build_prompt_renders_all_user_context_fields():
    """Every form field (selects + free text incl. King's reign) must land
    in the LLM prompt; blank strings render as 'unknown', never empty."""
    ctx = TextContext(period='new_kingdom', text_type='temple_wall',
                      support='granite', location_type='temple',
                      site='Karnak', dynasty='XVIII',
                      kings_reign='Thutmose III')
    prompt = build_egyptologist_prompt(
        ['G17'], [0.9], ctx, direction='rtl', layout='columns',
        cartouche_names=[], previous_context=None, chunk_info='chunk 1/1',
    )
    for line in ("- Period: new kingdom", "- Dynasty: XVIII",
                 "- King's reign: Thutmose III", "- Text type: temple wall",
                 "- Physical support: granite", "- Location type: temple",
                 "- Site: Karnak"):
        assert line in prompt, line
    # erased free-text field ('' from the client) -> 'unknown'
    p2 = build_egyptologist_prompt(
        ['G17'], [0.9],
        TextContext(kings_reign='', site='  ', dynasty=''),
        direction='rtl', layout='columns',
        cartouche_names=[], previous_context=None, chunk_info='chunk 1/1',
    )
    assert "- King's reign: unknown" in p2
    assert '- Site: unknown' in p2 and '- Dynasty: unknown' in p2


def test_build_prompt_separates_unresolved_cartouches():
    """UNRESOLVED entries must never appear under the 'authoritative' header."""
    prompt = build_egyptologist_prompt(
        ['G17'], [0.9], TextContext(),
        direction='rtl', layout='columns',
        cartouche_names=[
            'mn-xpr-ra — Thutmose III',
            '[UNRESOLVED cartouche — raw signs detected but no confident '
            'royal-name match: N5 Y5 L1]',
        ],
        previous_context=None, chunk_info='chunk 1/1',
    )
    auth = prompt.index('treat as authoritative')
    unres = prompt.index('NOT resolved to a known royal name')
    assert prompt.index('mn-xpr-ra') > auth
    assert auth < prompt.index('Thutmose III') < unres
    assert prompt.index('N5 Y5 L1') > unres
    # unresolved-only: no authoritative header at all
    p2 = build_egyptologist_prompt(
        ['G17'], [0.9], TextContext(), direction='rtl', layout='columns',
        cartouche_names=['[UNRESOLVED cartouche — no signs detected]'],
        previous_context=None, chunk_info='chunk 1/1',
    )
    assert 'treat as authoritative' not in p2
    assert 'NOT resolved' in p2
    # none at all
    p3 = build_egyptologist_prompt(
        ['G17'], [0.9], TextContext(), direction='rtl', layout='columns',
        cartouche_names=[], previous_context=None, chunk_info='chunk 1/1',
    )
    assert 'Contains cartouche: no' in p3


def test_clean_codes_drops_malformed():
    codes = ['G17', 'Aa15', 'NL3', 'D21a', 'not-a-code', 'g17 ', '', 123]
    out = _clean_codes(codes)
    assert 'G17' in out and 'Aa15' in out and 'NL3' in out and 'D21a' in out
    assert 'not-a-code' not in out and '' not in out and '123' not in out
    assert _clean_codes('not a list') == []


def test_inference_service_maps_raw_dict():
    svc = InferenceService(StubPipeline())
    import numpy as np
    resp = svc.run(np.zeros((64, 64, 3), dtype=np.uint8),
                   direction='rtl', layout='columns')
    raw = make_raw_result()
    assert resp.layout == 'columns' and resp.n_detections == 3
    assert resp.outer.correction.flat_corrected_seq == \
        raw['outer']['correction']['flat_corrected_seq']
    assert resp.cartouches[0].translit == 'wnjs'
    assert resp.annotated_image.startswith('data:image/jpeg;base64,')
    assert resp.transliteration is None       # translate=False default
