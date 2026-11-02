import ast
from pathlib import Path

import pytest

from app.core.config import settings
from app.schemas.transliterations import TextContext
from app.services import reverse, transliteration_services
from app.services.reverse import ReverseTranslationService
from app.services.sphinx_chat import SPHINX_PERSONA, ChatService
from app.services.transliteration_services import SYSTEM_PROMPT, TransliterationService
from tests.test_llm import StubMessages, _response, stub  # noqa: F401  (stub is a fixture)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def per_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the per-line path by disabling whole-inscription reading."""
    monkeypatch.setattr(settings, 'translit_whole_max_signs', 0)


def test_chat_uses_haiku_low_and_persona(stub: StubMessages) -> None:
    stub.response = _response(text='The sands remember Unas.')
    history = [{'role': 'assistant', 'content': 'Greetings, seeker.'},
               {'role': 'user', 'content': 'Who built Saqqara?'},
               {'role': 'assistant', 'content': 'Djoser, through Imhotep.'}]
    reply, message_id = ChatService().chat('And Unas?', history)
    assert reply == 'The sands remember Unas.' and len(message_id) == 32
    _, kw = stub.calls[0]
    assert kw['model'] == 'claude-haiku-5-5' and kw['output_config'] == {'effort': 'low'}
    assert kw['system'][0]['text'] == SPHINX_PERSONA
    # Leading assistant greeting dropped: Claude needs a user turn first.
    assert [m['role'] for m in kw['messages']] == ['user', 'assistant', 'user']
    assert kw['messages'][-1]['content'] == 'And Unas?'


def test_chat_disabled_raises(no_api_key: None) -> None:
    with pytest.raises(RuntimeError, match='ANTHROPIC_API_KEY'):
        ChatService().chat('hello', [])


def test_reverse_maps_parsed_reply(stub: StubMessages) -> None:
    reply = reverse._ReverseReply(
        normalized_english='the scribe', transliteration='sS',
        gardiner_codes=['Y3', 'A1', 'bogus!'],
        words=[reverse._WordReply(english='scribe', transliteration='sS',
                                  gardiner_codes=['Y3', 'A1'], literal='scribe', note='')],
        grammar_notes='Noun.', confidence='HIGH')
    stub.response = _response(parsed=reply, model='claude-sonnet-5-5')
    out = ReverseTranslationService().translate('the scribe', 'literary')
    assert out.error is None and out.confidence == 'HIGH'
    assert out.gardiner_codes == ['Y3', 'A1']           # malformed code sanitized away
    assert out.words[0].transliteration == 'sS' and out.model == 'claude-sonnet-5-5'
    method, kw = stub.calls[0]
    assert method == 'parse' and kw['output_format'] is reverse._ReverseReply
    assert kw['model'] == 'claude-sonnet-5-5' and kw['fallbacks'] == 'default'
    assert kw['system'][0]['text'] == reverse.SYSTEM_PROMPT
    assert 'the literary register' in kw['messages'][0]['content']


def test_reverse_failure_degrades(stub: StubMessages) -> None:
    stub.response = _response(stop_reason='refusal')
    out = ReverseTranslationService().translate('hello')
    assert out.error and 'refused' in out.error and out.gardiner_codes == []


def _chunk(text: str) -> transliteration_services._ChunkReading:
    """A canned per-chunk reading."""
    return transliteration_services._ChunkReading(
        transliteration=text, english_gloss=f'gloss {text}', linguistic_notes='',
        confidence='MEDIUM', period_note='')


def test_transliteration_per_chunk_calls(stub: StubMessages, per_line: None) -> None:
    stub.response = _response(parsed=_chunk('Htp-dj-nsw'))
    out = TransliterationService().transliterate_sequence(
        ['M23', 'X1', 'R4', 'G17', 'N35'], [0.9, 0.8, 0.4, 0.95, 0.6], [3], [],
        direction='rtl', layout='columns', ctx=TextContext())
    assert out.error is None and out.n_chunks == 2 and len(stub.calls) == 2
    assert out.full_transliteration == 'Htp-dj-nsw | Htp-dj-nsw'
    assert out.chunks[1].gardiner_codes == ['G17', 'N35'] and out.model == 'claude-haiku-5-5'
    _, kw = stub.calls[0]
    assert kw['output_config'] == {'effort': 'medium'} and 'fallbacks' not in kw
    assert kw['system'][0]['text'] == SYSTEM_PROMPT
    assert 'Segment 1 of 2' in kw['messages'][0]['content']
    # The second chunk sees the first one's reading as previous context.
    assert 'PREVIOUS SEGMENTS' in stub.calls[1][1]['messages'][0]['content']


def test_transliteration_failure_keeps_partial(stub: StubMessages, per_line: None) -> None:
    calls = {'n': 0}
    good = _response(parsed=_chunk('nb tAwy'))
    bad = _response(stop_reason='max_tokens')

    def flaky(**kwargs: object) -> object:
        calls['n'] += 1
        return good if calls['n'] == 1 else bad

    stub.parse = flaky  # type: ignore[method-assign]
    out = TransliterationService().transliterate_sequence(
        ['G17', 'N35', 'D21', 'X1'], [0.9] * 4, [2], [],
        direction='rtl', layout='columns', ctx=TextContext())
    assert out.error and 'segment 2/2' in out.error
    assert [c.transliteration for c in out.chunks] == ['nb tAwy']


NOISE_OLD = 'comparable to a worn wall read from a photograph — and you have full licence to emend it:'


def test_prompt_differs_from_space_only_in_noise_lines() -> None:
    # The Space copy (gpt-4o) keeps the old "worn wall" wording; the rest is shared.
    space = REPO_ROOT / 'space' / 'src' / 'llm.py'
    if not space.exists():
        pytest.skip('space/src/llm.py missing')
    tree = ast.parse(space.read_text())
    consts = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    old_head, old_tail = consts['SYSTEM_PROMPT'].split(NOISE_OLD)
    assert SYSTEM_PROMPT.startswith(old_head) and SYSTEM_PROMPT.endswith(old_tail)
    assert 'worn wall' not in SYSTEM_PROMPT and 'the noise comes from the detector' in SYSTEM_PROMPT


def test_app_never_imports_openai() -> None:
    for path in (REPO_ROOT / 'app').rglob('*.py'):
        tree = ast.parse(path.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(name.split('.')[0] == 'openai' for name in names), path


def test_transliteration_sends_raw_photo_once_cached(stub: StubMessages, per_line: None) -> None:
    import numpy as np
    stub.response = _response(parsed=_chunk('kA nxt xa m WAst'))
    bgr = np.full((2000, 1000, 3), 200, dtype=np.uint8)
    TransliterationService().transliterate_sequence(
        ['E1', 'D40', 'X1', 'G17'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext(), image_bgr=bgr)
    blocks = [kw['messages'][0]['content'] for _, kw in stub.calls]
    assert len(blocks) == 2 and all(b[0]['type'] == 'image' for b in blocks)
    # Same image bytes in both calls, so system + image form a cached prefix.
    assert blocks[0][0] == blocks[1][0] and blocks[0][0]['cache_control'] == {'type': 'ephemeral'}
    assert 'THE PHOTOGRAPH' in blocks[0][1]['text'] and 'read ltr' in blocks[0][1]['text']


def test_transliteration_without_photo_is_plain_text(stub: StubMessages, per_line: None) -> None:
    stub.response = _response(parsed=_chunk('nb'))
    TransliterationService().transliterate_sequence(
        ['V30'], [0.9], [], [], direction='rtl', layout='rows', ctx=TextContext())
    content = stub.calls[0][1]['messages'][0]['content']
    assert isinstance(content, str) and 'THE PHOTOGRAPH' not in content


def test_downscale_fits_640_and_never_upscales() -> None:
    import numpy as np
    from app.llm import images
    assert images.downscale(np.zeros((3000, 1500, 3), np.uint8)).shape[:2] == (640, 320)
    assert images.downscale(np.zeros((300, 200, 3), np.uint8)).shape[:2] == (300, 200)


def test_decode_data_url_roundtrip_and_rejects_garbage() -> None:
    import base64

    import cv2
    import numpy as np
    from app.llm import images
    ok, buf = cv2.imencode('.png', np.zeros((10, 20, 3), np.uint8))
    url = 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()
    assert images.decode_data_url(url).shape == (10, 20, 3)
    with pytest.raises(ValueError):
        images.decode_data_url('data:image/png;base64,bm90IGFuIGltYWdl')


def test_transliterate_endpoint_rejects_bad_image(client) -> None:  # noqa: ANN001
    r = client.post('/transliterate/', json={'codes': ['G17'], 'image': 'data:image/png;base64,eA=='})
    assert r.status_code == 422 and 'image' in r.json()['detail']


def _wall(*lines: str, full: str = 'Live the Horus, Strong Bull.', confidence: str = 'MEDIUM',
          ask: bool = False) -> transliteration_services._DraftWallReading:
    """A canned whole-inscription reading, one entry per transliteration given."""
    return transliteration_services._DraftWallReading(
        lines=[transliteration_services._LineReading(
            line=n, transliteration=t, english_gloss=f'gloss {t}', linguistic_notes='',
            confidence=confidence) for n, t in enumerate(lines, start=1)],
        full_translation=full, period_note='Dynasty XVIII.',
        image_notes=['Sunk relief with traces of red paint.', '  ', 'The cartouches sit on one axis.'],
        needs_second_opinion=ask, second_opinion_reason='two readings of line 1' if ask else '')


def test_whole_wall_is_one_call(stub: StubMessages) -> None:
    import numpy as np
    stub.response = _response(parsed=_wall('anx Hr kA-nxt', 'nsw-bjtj mn-xpr-ra'))
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'E1', 'S34', 'M23', 'L2'], [0.9, 0.9, 0.5, 0.9, 0.8, 0.3], [3],
        ['mn-xpr-ra — Thutmose III (interior: N5 Y5 L1)'],
        direction='ltr', layout='rows', ctx=TextContext(), image_bgr=np.zeros((50, 80, 3), np.uint8))
    assert len(stub.calls) == 1 and out.error is None and out.n_chunks == 2
    method, kw = stub.calls[0]
    assert method == 'parse' and kw['output_format'] is transliteration_services._DraftWallReading
    image, text = kw['messages'][0]['content']
    assert image['type'] == 'image'
    assert '[1] S34(HIGH:0.90)' in text['text'] and '[2] S34(HIGH:0.90)' in text['text']
    assert 'WHOLE-INSCRIPTION READING' in text['text'] and 'match each numbered line to its row' in text['text']
    assert [c.gardiner_codes for c in out.chunks] == [['S34', 'G5', 'E1'], ['S34', 'M23', 'L2']]
    assert out.full_translation == 'Live the Horus, Strong Bull.'
    assert out.chunks[0].period_note == 'Dynasty XVIII.' and out.chunks[1].period_note == ''


def test_whole_wall_missing_line_is_marked(stub: StubMessages) -> None:
    stub.response = _response(parsed=_wall('anx Hr kA-nxt'))
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'M23', 'L2'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext())
    assert out.chunks[1].transliteration == '' and out.chunks[1].confidence == 'LOW'
    assert 'No reading' in out.chunks[1].linguistic_notes


def test_whole_wall_truncated_falls_back_per_line(stub: StubMessages) -> None:
    replies = iter([_response(stop_reason='max_tokens'),
                    _response(parsed=_chunk('a')), _response(parsed=_chunk('b'))])
    stub.parse = lambda **kwargs: next(replies)  # type: ignore[method-assign]
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'M23', 'L2'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext())
    assert out.error is None and out.full_transliteration == 'a | b'


def test_long_wall_goes_per_line(stub: StubMessages, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'translit_whole_max_signs', 3)
    stub.response = _response(parsed=_chunk('x'))
    TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'M23', 'L2'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext())
    assert len(stub.calls) == 2 and stub.calls[0][1]['output_format'] is transliteration_services._ChunkReading


def test_whole_wall_refusal_reports_error(stub: StubMessages) -> None:
    stub.response = _response(stop_reason='refusal')
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5'], [0.9] * 2, [], [], direction='ltr', layout='rows', ctx=TextContext())
    assert out.error and 'whole inscription' in out.error and out.chunks == []


def test_confident_draft_is_not_escalated(stub: StubMessages) -> None:
    stub.response = _response(parsed=_wall('anx Hr kA-nxt', 'nsw-bjtj'))
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'M23', 'L2'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext())
    assert len(stub.calls) == 1 and not out.escalated
    assert out.escalation_reasons == ['haiku_confident'] and out.cost_usd is not None
    # Haiku is told it may ask for a second opinion.
    assert 'SECOND OPINION' in stub.calls[0][1]['messages'][0]['content']


def test_haiku_request_escalates_to_sonnet_review(stub: StubMessages) -> None:
    import numpy as np
    draft = _response(parsed=_wall('anx Hr kA-nxt', 'nsw-bjtj', ask=True))
    review = _response(parsed=transliteration_services._WallReading(**{
        k: v for k, v in _wall('anx Hr kA-nxt xa-m-WAst', 'nsw-bjtj').model_dump().items()
        if k in ('lines', 'full_translation', 'period_note')}, image_notes=['Sonnet saw the N28.']),
        model='claude-sonnet-5-5')
    replies = iter([draft, review])
    calls: list[dict] = []

    def parse(**kwargs: object) -> object:
        calls.append(kwargs)
        return next(replies)

    stub.parse = parse  # type: ignore[method-assign]
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'M23', 'L2'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext(), image_bgr=np.zeros((40, 40, 3), np.uint8))
    assert out.escalated and out.escalation_reasons == ['haiku_requested_second_opinion']
    assert out.model == 'claude-sonnet-5-5' and out.chunks[0].transliteration == 'anx Hr kA-nxt xa-m-WAst'
    assert out.image_notes == ['Sonnet saw the N28.']   # the reviewer's notes replace the draft's
    sonnet = calls[1]
    assert sonnet['model'] == 'claude-sonnet-5-5' and sonnet['output_format'] is transliteration_services._WallReading
    image, text = sonnet['messages'][0]['content']
    assert image['type'] == 'image' and image == calls[0]['messages'][0]['content'][0]
    assert 'FIRST READING' in text['text'] and 'two readings of line 1' in text['text']
    assert 'SECOND OPINION' not in text['text'] and '# type' not in text['text']


def test_failed_review_keeps_draft_with_note(stub: StubMessages) -> None:
    replies = iter([_response(parsed=_wall('anx', 'nb', confidence='LOW')),
                    _response(stop_reason='refusal')])
    stub.parse = lambda **kwargs: next(replies)  # type: ignore[method-assign]
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5', 'M23', 'L2'], [0.9] * 4, [2], [],
        direction='ltr', layout='rows', ctx=TextContext())
    assert not out.escalated and out.escalation_reasons == ['most_lines_low_confidence']
    assert out.escalation_note and 'Second opinion unavailable' in out.escalation_note
    assert out.full_transliteration == 'anx | nb' and out.error is None


def test_escalation_can_be_switched_off(stub: StubMessages, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'translit_escalation', False)
    stub.response = _response(parsed=_wall('anx', confidence='LOW'))
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5'], [0.9] * 2, [], [], direction='ltr', layout='rows', ctx=TextContext())
    assert len(stub.calls) == 1 and out.escalation_reasons == ['escalation_disabled']
    assert 'SECOND OPINION' not in stub.calls[0][1]['messages'][0]['content']


def test_whole_wall_returns_image_notes(stub: StubMessages) -> None:
    import numpy as np
    stub.response = _response(parsed=_wall('anx Hr kA-nxt'))
    out = TransliterationService().transliterate_sequence(
        ['S34', 'G5'], [0.9] * 2, [], [], direction='ltr', layout='rows', ctx=TextContext(),
        image_bgr=np.zeros((40, 40, 3), np.uint8))
    # Blank notes dropped, order kept.
    assert out.image_notes == ['Sunk relief with traces of red paint.', 'The cartouches sit on one axis.']
    text = stub.calls[0][1]['messages'][0]['content'][1]['text']
    assert 'IMAGE NOTES' in text and 'about the photograph' in text


def test_notes_without_photo_describe_the_inscription(stub: StubMessages) -> None:
    stub.response = _response(parsed=_wall('anx'))
    TransliterationService().transliterate_sequence(
        ['S34'], [0.9], [], [], direction='ltr', layout='rows', ctx=TextContext())
    assert 'no photograph is attached' in stub.calls[0][1]['messages'][0]['content']


def test_per_line_path_has_no_notes(stub: StubMessages, per_line: None) -> None:
    stub.response = _response(parsed=_chunk('anx'))
    out = TransliterationService().transliterate_sequence(
        ['S34'], [0.9], [], [], direction='ltr', layout='rows', ctx=TextContext())
    assert out.image_notes == []
