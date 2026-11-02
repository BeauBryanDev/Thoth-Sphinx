import ast
from pathlib import Path

import pytest

from app.schemas.transliterations import TextContext
from app.services import reverse, transliteration_services
from app.services.reverse import ReverseTranslationService
from app.services.sphinx_chat import SPHINX_PERSONA, ChatService
from app.services.transliteration_services import SYSTEM_PROMPT, TransliterationService
from tests.test_llm import StubMessages, _response, stub  # noqa: F401  (stub is a fixture)


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_transliteration_per_chunk_calls(stub: StubMessages) -> None:
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


def test_transliteration_failure_keeps_partial(stub: StubMessages) -> None:
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


def test_transliteration_sends_raw_photo_once_cached(stub: StubMessages) -> None:
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


def test_transliteration_without_photo_is_plain_text(stub: StubMessages) -> None:
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
