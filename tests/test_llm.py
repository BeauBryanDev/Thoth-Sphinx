import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

import app.llm.client as llm
from app.core.config import settings
from app.llm.pricing import estimate_cost


REPO_ROOT = Path(__file__).resolve().parents[1]


class Reply(BaseModel):
    """Toy structured-output schema."""
    reading : str
    notes   : list[str]


def _response(stop_reason: str = 'end_turn', text: str = 'hello', parsed: BaseModel | None = None,
              model: str = 'claude-haiku-5-5', iterations: list[Any] | None = None,
              stop_details: Any = None) -> SimpleNamespace:
    """A minimal stand-in for a BetaMessage."""
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=5000,
                            cache_creation_input_tokens=0, iterations=iterations)
    return SimpleNamespace(content=[SimpleNamespace(type='thinking', thinking=''),
                                    SimpleNamespace(type='text', text=text)],
                           stop_reason=stop_reason, stop_details=stop_details, usage=usage,
                           model=model, parsed_output=parsed, _request_id='req_test')


class StubMessages:
    """Records kwargs; returns a canned response or raises a canned error."""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response if response is not None else _response()
        self.error    = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, **kwargs: Any) -> Any:
        """Stub for client.beta.messages.create."""
        return self._run('create', kwargs)

    def parse(self, **kwargs: Any) -> Any:
        """Stub for client.beta.messages.parse."""
        return self._run('parse', kwargs)

    def _run(self, method: str, kwargs: dict[str, Any]) -> Any:
        """Record the call, then raise or return."""
        self.calls.append((method, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture()
def stub(monkeypatch: pytest.MonkeyPatch) -> StubMessages:
    """Patch get_client to a stub; returns its messages recorder."""
    messages = StubMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(settings, 'anthropic_api_key', 'sk-test')
    monkeypatch.setattr(settings, 'debug_prompts', False)
    monkeypatch.setattr(llm, 'get_client', lambda: client)
    return messages


def _status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    """Build an SDK status error without a network call."""
    request = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')
    return cls('boom', response=httpx2.Response(code, request=request), body=None)


MSG: list[dict[str, Any]] = [{'role': 'user', 'content': 'G17 X1 D21'}]


def test_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'anthropic_api_key', '')
    assert llm.get_client() is None and not llm.enabled()
    with pytest.raises(llm.LLMDisabled):
        llm.complete_text(feature='chat', model='claude-haiku-5-5', system='s',
                          messages=MSG, effort='low')


def test_text_request_shape(stub: StubMessages) -> None:
    text, usage = llm.complete_text(feature='chat', model='claude-haiku-5-5', system='persona',
                                    messages=MSG, effort='low')
    assert text == 'hello'
    method, kw = stub.calls[0]
    assert method == 'create'
    assert kw['model'] == 'claude-haiku-5-5'
    assert kw['output_config'] == {'effort': 'low'}
    assert kw['system'] == [{'type': 'text', 'text': 'persona',
                             'cache_control': {'type': 'ephemeral'}}]
    for banned in ('temperature', 'top_p', 'top_k', 'thinking', 'tool_choice'):
        assert banned not in kw
    # Haiku 5.5 has no server-side fallback.
    assert 'fallbacks' not in kw and 'betas' not in kw
    assert usage.input_tokens == 1000 and usage.cache_read == 5000
    assert usage.served_by == 'claude-haiku-5-5' and not usage.fallback_used
    assert usage.request_id == 'req_test'


def test_sonnet_opts_into_default_fallback(stub: StubMessages) -> None:
    stub.response = _response(model='claude-sonnet-5-5')
    llm.complete_text(feature='reverse', model='claude-sonnet-5-5', system='s',
                      messages=MSG, effort='medium')
    _, kw = stub.calls[0]
    assert kw['fallbacks'] == 'default'
    assert kw['betas'] == ['server-side-fallback-2026-07-01']


def test_fallback_detected_and_priced_at_serving_model(stub: StubMessages) -> None:
    stub.response = _response(model='claude-opus-4-8',
                              iterations=[SimpleNamespace(type='message'),
                                          SimpleNamespace(type='fallback_message')])
    _, usage = llm.complete_text(feature='reverse', model='claude-sonnet-5-5', system='s',
                                 messages=MSG, effort='medium')
    assert usage.fallback_used and usage.served_by == 'claude-opus-4-8'
    assert usage.cost_usd is None  # no price on file for the fallback model


def test_parsed_returns_schema_instance(stub: StubMessages) -> None:
    stub.response = _response(parsed=Reply(reading='nswt', notes=['n1']))
    out, _ = llm.complete_parsed(feature='translit', model='claude-haiku-5-5', system='s',
                                 messages=MSG, schema=Reply, effort='medium')
    assert isinstance(out, Reply) and out.reading == 'nswt'
    method, kw = stub.calls[0]
    assert method == 'parse' and kw['output_format'] is Reply


def test_parsed_none_raises(stub: StubMessages) -> None:
    stub.response = _response(parsed=None)
    with pytest.raises(llm.LLMError, match='Reply'):
        llm.complete_parsed(feature='translit', model='claude-haiku-5-5', system='s',
                            messages=MSG, schema=Reply, effort='medium')


def test_refusal_raises_with_category(stub: StubMessages) -> None:
    stub.response = _response(stop_reason='refusal',
                              stop_details=SimpleNamespace(category='cyber', explanation='no'))
    with pytest.raises(llm.LLMRefusal) as exc:
        llm.complete_text(feature='chat', model='claude-haiku-5-5', system='s',
                          messages=MSG, effort='low')
    assert exc.value.category == 'cyber'


def test_max_tokens_raises(stub: StubMessages) -> None:
    stub.response = _response(stop_reason='max_tokens')
    with pytest.raises(llm.LLMTruncated):
        llm.complete_text(feature='chat', model='claude-haiku-5-5', system='s',
                          messages=MSG, effort='low', max_tokens=10)


@pytest.mark.parametrize(('error', 'expected'), [
    (_status_error(anthropic.BadRequestError, 400), llm.LLMRequestError),
    (_status_error(anthropic.AuthenticationError, 401), llm.LLMRequestError),
    (_status_error(anthropic.NotFoundError, 404), llm.LLMRequestError),
    (_status_error(anthropic.RateLimitError, 429), llm.LLMUnavailable),
    (_status_error(anthropic.InternalServerError, 500), llm.LLMUnavailable),
    (anthropic.APIConnectionError(request=httpx2.Request('POST', 'https://api.anthropic.com')),
     llm.LLMUnavailable),
])
def test_sdk_errors_map_to_llm_errors(stub: StubMessages, error: Exception,
                                      expected: type[llm.LLMError]) -> None:
    stub.error = error
    with pytest.raises(expected):
        llm.complete_text(feature='chat', model='claude-haiku-5-5', system='s',
                          messages=MSG, effort='low')


def test_debug_dump_redacts_images(stub: StubMessages, monkeypatch: pytest.MonkeyPatch,
                                   tmp_path: Path) -> None:
    monkeypatch.setattr(settings, 'debug_prompts', True)
    monkeypatch.setattr(llm, 'DEBUG_DIR', tmp_path)
    image = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg',
                                         'data': 'A' * 500}}
    messages = [{'role': 'user', 'content': [image, {'type': 'text', 'text': 'read'}]}]
    llm.complete_text(feature='read', model='claude-haiku-5-5', system='s',
                      messages=messages, effort='low')
    dumped = json.loads(next(tmp_path.glob('*_read.json')).read_text())
    assert dumped['messages'][0]['content'][0]['source']['data'] == '<500 base64 chars>'
    assert dumped['reply'] == 'hello' and 'sk-test' not in json.dumps(dumped)
    # The caller's messages are untouched.
    assert image['source']['data'] == 'A' * 500


def test_estimate_cost() -> None:
    # 1000 in at $0.10/M + 200 out at $0.50/M + 5000 cache reads at $0.01/M
    assert estimate_cost('claude-haiku-5-5', 1000, 200, 5000) == pytest.approx(0.00025)
    assert estimate_cost('claude-sonnet-5-5', 1_000_000, 0) == pytest.approx(2.0)
    # Haiku long-context rate above 100K prompt tokens.
    assert estimate_cost('claude-haiku-5-5', 200_000, 0) == pytest.approx(0.10)
    assert estimate_cost('gpt-4o', 10, 10) is None


def test_llm_package_never_imports_openai() -> None:
    for path in (REPO_ROOT / 'app' / 'llm').glob('*.py'):
        tree = ast.parse(path.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(name.split('.')[0] == 'openai' for name in names), path
