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


def test_prompts_kept_verbatim() -> None:
    # Byte-identical to the Space copy, which still runs on gpt-4o.
    space = REPO_ROOT / 'space' / 'src' / 'llm.py'
    if not space.exists():
        pytest.skip('space/src/llm.py missing')
    tree = ast.parse(space.read_text())
    consts = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert consts['SYSTEM_PROMPT'] == SYSTEM_PROMPT


def test_app_never_imports_openai() -> None:
    for path in (REPO_ROOT / 'app').rglob('*.py'):
        tree = ast.parse(path.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(name.split('.')[0] == 'openai' for name in names), path
