import copy
import json
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from app.core import ratelimit
from app.core.config import settings
from app.llm.pricing import estimate_cost


log = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Server-side fallback on refusal; Claude API only, and Haiku 5.5 has none.
FALLBACK_BETA: str = 'server-side-fallback-2026-07-01'
FALLBACK_MODELS: frozenset[str] = frozenset({'claude-sonnet-5-5'})

DEBUG_DIR = Path('debug_prompts')


class LLMError(Exception):
    """Base class for every app/llm failure."""


class LLMDisabled(LLMError):
    """No ANTHROPIC_API_KEY configured."""


class LLMRefusal(LLMError):
    """The model (and any fallback) declined the request."""

    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f'refused ({category or "uncategorized"}): {explanation or ""}'.strip())
        self.category    = category
        self.explanation = explanation


class LLMTruncated(LLMError):
    """The reply hit max_tokens before finishing."""


class LLMUnavailable(LLMError):
    """Retryable failure: rate limit, overload, server error or network."""


class LLMBudgetSpent(LLMUnavailable):
    """Today's global LLM budget (settings.daily_budget_usd) is spent."""


class LLMRequestError(LLMError):
    """Non-retryable failure: bad request, auth, permission or unknown model."""


@dataclass(frozen=True)
class Usage:
    """Token counts, cost and latency of one call."""
    feature       : str
    model         : str          # model requested
    served_by     : str          # model that answered (differs after a fallback)
    input_tokens  : int
    output_tokens : int
    cache_read    : int
    cache_write   : int
    cost_usd      : float | None
    latency_ms    : int
    fallback_used : bool
    request_id    : str | None


@lru_cache(maxsize=1)
def _client_for(api_key: str, timeout: float) -> anthropic.Anthropic:
    """One SDK client per key and timeout."""
    return anthropic.Anthropic(api_key=api_key, timeout=timeout)


def get_client() -> anthropic.Anthropic | None:
    """The shared client, or None when no API key is set."""
    if not settings.anthropic_api_key:
        return None
    return _client_for(settings.anthropic_api_key, settings.llm_timeout)


def enabled() -> bool:
    """True when an API key is configured."""
    return bool(settings.anthropic_api_key)


def complete_text(*, feature: str, 
model: str, 
system: str,
messages: list[dict[str, Any]],
effort: str, 
max_tokens: int = 4096
) -> tuple[str, Usage]:
    """One call returning the reply's text."""
    response, usage = _call(feature=feature,
                            model=model,
                            system=system,
                            messages=messages, 
                            tools=None,
                            effort=effort,
                            max_tokens=max_tokens,
                            schema=None)

    text = ''.join(b.text for b in response.content if b.type == 'text')
    _dump(feature, model, system, messages, text, usage)

    return text, usage


def complete_parsed(*, feature: str, 
model: str, 
system: str,
messages: list[dict[str, Any]],
schema: type[T], 
effort: str, 
max_tokens: int = 8192
) -> tuple[T, Usage]:
    """One call whose reply is validated against a pydantic schema."""
    response, usage = _call(feature=feature,
                            model=model, 
                            system=system,
                            messages=messages,        
                            effort=effort, 
                            max_tokens=max_tokens,
                            schema=schema, 
                            tools=None)

    parsed = response.parsed_output

    _dump(feature, model, 
        system, messages,
        parsed.model_dump() if parsed is not None else None, usage ) 

    if parsed is None:
        raise LLMError(f'{feature}: reply did not match {schema.__name__}')

    return parsed, usage

# Helper  _CaLL
def _call(*, feature: str, 
            model: str, system: str, 
            messages: list[dict[str, Any]],
            effort: str, 
            max_tokens: int, 
            schema: type[BaseModel] | None,
            tools: list[dict[str, Any]] | None = None
            ) -> tuple[Any, Usage]:
    """Send one request and map stop reasons and SDK errors to LLMError."""
    client = get_client()
    if client is None:
        raise LLMDisabled('ANTHROPIC_API_KEY is not set')
    # Checked per call, so a tool loop or a review stops as soon as the day's budget is gone.
    if ratelimit.SPEND.exhausted(settings.daily_budget_usd):
        raise LLMBudgetSpent(f'{feature}: daily LLM budget of ${settings.daily_budget_usd:.2f} spent')

    # Static system prompt first and cached: it is identical across calls of a feature.
    kwargs: dict[str, Any] = {
        'model'         : model,
        'max_tokens'    : max_tokens,
        'system'        : [{'type': 'text', 
                            'text': system, 
                            'cache_control': {'type': 'ephemeral'}
        }],
        'messages'      : messages,
        'output_config' : {'effort': effort},
    }
    if tools:
        kwargs['tools'] = tools

    if model in FALLBACK_MODELS:
        kwargs['betas']     = [FALLBACK_BETA]
        kwargs['fallbacks'] = 'default'

    started = time.monotonic()
    try:
        if schema is None:
            response = client.beta.messages.create(**kwargs)
        else:
            response = client.beta.messages.parse(output_format=schema, **kwargs)

    except (anthropic.BadRequestError, anthropic.AuthenticationError,
            anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
        raise LLMRequestError(f'{feature}: {e.status_code} {e.message}') from e

    except anthropic.RateLimitError as e:
        raise LLMUnavailable(f'{feature}: rate limited') from e

    except anthropic.APIStatusError as e:
        if e.status_code >= 500:
            raise LLMUnavailable(f'{feature}: server error {e.status_code}') from e

        raise LLMRequestError(f'{feature}: {e.status_code} {e.message}') from e

    except anthropic.APIConnectionError as e:
        raise LLMUnavailable(f'{feature}: connection failed') from e

    usage = _usage(feature, model, response, int((time.monotonic() - started) * 1000))
    if usage.cost_usd is None:
        log.warning('no price for %s: this call is not counted against the daily budget', usage.served_by)
    ratelimit.SPEND.add(usage.cost_usd or 0.0)
    log.info('llm %s model=%s served_by=%s in=%d out=%d cache_read=%d cost=%s ms=%d',
                feature, model, usage.served_by, usage.input_tokens,
                usage.output_tokens,
                usage.cache_read, 
                usage.cost_usd, 
                usage.latency_ms)

    if response.stop_reason == 'refusal':
        details = response.stop_details
        _dump(feature, model, system, messages, None, usage)
        raise LLMRefusal(getattr(details, 'category', None), 
                        getattr(details, 'explanation', None))

    if response.stop_reason == 'max_tokens':
        _dump(feature, model, system, messages, None, usage)
        raise LLMTruncated(f'{feature}: reply hit max_tokens={max_tokens}')

    return response, usage


@dataclass(frozen=True)
class ToolCall:
    """One tool call a model made, as recorded for the response and the trace."""
    model      : str
    tool       : str
    input      : dict[str, Any]
    summary    : str
    ok         : bool
    latency_ms : int
    result     : dict[str, Any] | None   # compact result; None when the call failed


def cache_turn(message: dict[str, Any]) -> dict[str, Any]:
    """A text message as one block marked for caching: later tool turns and the next chat turn read it."""
    return {
        'role': message['role'],
        'content': [
            {
                'type': 'text',
                'text': message['content'],
                'cache_control': {'type': 'ephemeral'},
            },
        ],
    }


BUDGET_SPENT = 'Tool budget spent for this reading. Answer now with what you have.'

# Shared by every request: one turn's tool calls, and the evidence check, run side by side.
# Bounded, so each worker keeps at most one corpus DB connection (evidence_store.retrieve).
TOOL_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='sphinx-tool')
MAX_CALLS_PER_TURN = 12   # one search per line of a whole-wall read, plus a few lookups


def _assistant_content(response: Any) -> list[dict[str, Any]]:
    """The assistant turn to send back: every block unchanged, minus the SDK's parse cache."""
    blocks = []
    for block in response.content:
        data = block.to_dict() if hasattr(block, 'to_dict') else dict(vars(block))
        data.pop('parsed_output', None)
        blocks.append(data)
    return blocks


def complete_parsed_with_tools(*, feature: str, 
                            model: str, system: str,
                            messages: list[dict[str, Any]],
                            schema: type[T], effort: str,
                            tools: Sequence[Any],
                            max_tool_turns: int,
                            trace: Any = None,
                            max_tokens: int = 16000
                            ) -> tuple[T, list[Usage], list[ToolCall]]:
    """Hand-written tool loop : the model may call tools, then answers in the schema."""
    by_name = {t.name: t for t in tools}
    definitions = [t.definition() for t in tools]
    history = list(messages)
    usages: list[Usage] = []
    calls: list[ToolCall] = []

    # A budget of round trips, not calls: one turn's calls run side by side.
    for turn in range(max_tool_turns + 2):
        response, usage = _call(feature=feature, 
                                model=model, 
                                system=system, 
                                messages=history,
                                effort=effort, 
                                max_tokens=max_tokens,
                                schema=schema,
                                tools=definitions)

        usages.append(usage)
        uses = [b for b in response.content if getattr(b, 'type', None) == 'tool_use']
        if trace is not None:
            trace.add_usage(usage, f'turn {turn + 1}: ' + (
                f'calls {", ".join(u.name for u in uses)}' if uses else 'final answer'))

        if response.stop_reason != 'tool_use' or not uses:
            parsed = response.parsed_output
            _dump(feature, model, system, history,
                  parsed.model_dump() if parsed is not None else None, usage)

            if parsed is None:
                raise LLMError(f'{feature}: reply did not match {schema.__name__}')

            return parsed, usages, calls

        results = _run_tools(
            model,
            uses,
            by_name,
            calls,
            MAX_CALLS_PER_TURN if turn < max_tool_turns else 0,
            trace,
        )

        history = history + [{'role': 'assistant', 
                            'content': _assistant_content(response)},
                             {'role': 'user', 
                             'content': results}
                             ]

    raise LLMError(f'{feature}: no final answer after {max_tool_turns + 2} turns')


def _run_tools(
    model: str,
    uses: list[Any],
    by_name: dict[str, Any],
    calls: list[ToolCall],
    room: int,
    trace: Any,
) -> list[dict[str, Any]]:
    """Run up to `room` of this turn's tool calls in parallel; results keep the model's order."""
    outcomes = list(TOOL_POOL.map(lambda use: _execute(use, by_name), uses[:room]))
    outcomes += [_REFUSED] * (len(uses) - len(outcomes))
    return [
        _run_tool(model, use, outcome, calls, trace)
        for use, outcome in zip(uses, outcomes)
    ]


# (content, ok, result, summary, latency_ms) of one tool call
_Outcome = tuple[str, bool, dict[str, Any] | None, str, int]
_REFUSED: _Outcome = (BUDGET_SPENT, False, None, 'refused: tool budget spent', 0)


def _execute(
    use: Any,
    by_name: dict[str, Any],
) -> _Outcome:
    """Call one tool; errors become an outcome, never an exception."""
    if use.name not in by_name:
        return f'Unknown tool {use.name}.', False, None, 'unknown tool', 0
    started = time.monotonic()
    try:
        result = by_name[use.name].run(**dict(use.input or {}))
        content, ok = json.dumps(result, ensure_ascii=False, default=str), True
        summary = _summarize(use.name, result)
    except Exception as e:  # tool errors go back to the model, never swallowed (spec §12)
        log.warning('tool %s failed: %s', use.name, e)
        content, ok, result = f'Tool error: {e}', False, None
        summary = f'error: {e}'
    return content, ok, result, summary, int((time.monotonic() - started) * 1000)


def _run_tool(
    model: str,
    use: Any,
    outcome: _Outcome,
    calls: list[ToolCall],
    trace: Any,
) -> dict[str, Any]:
    """Record one tool call (calls list, trace) and return its tool_result block."""
    tool_input = dict(use.input or {})
    content, ok, result, summary, latency = outcome

    calls.append(ToolCall(model=model, tool=use.name, 
                          input=tool_input, summary=summary,
                          ok=ok, latency_ms=latency, 
                          result=result))
                          
    if trace is not None:
        trace.add('tool_call', use.name, summary, latency_ms=latency,
                  detail={'called_by': model, 'input': tool_input, 'ok': ok}
                  )
    block: dict[str, Any] = {
                            'type': 'tool_result', 
                            'tool_use_id': use.id, 
                            'content': content
                            }

    if not ok:
        block['is_error'] = True

    return block


def _summarize(name: str, result: dict[str, Any]) -> str:
    """Trace line for a tool result; agent_tools knows the shapes."""
    from app.llm.agent_tools import summarize  # local import: agent_tools imports the DB layer
    return summarize(name, result)


def _usage(feature: str, 
model: str, 
response: Any, 
latency_ms: int
) -> Usage:
    """Build a Usage record from a response."""
    u = response.usage
    iterations = getattr(u, 'iterations', None) or []
    fallback_used = any(getattr(it, 'type', None) == 'fallback_message' for it in iterations)
    served_by = getattr(response, 'model', None) or model
    input_tokens  = u.input_tokens or 0
    output_tokens = u.output_tokens or 0
    cache_read    = u.cache_read_input_tokens or 0
    cache_write   = u.cache_creation_input_tokens or 0
    # Priced at the serving model's rate; after a fallback this is an estimate across hops.
    cost = estimate_cost(served_by, input_tokens,
                        output_tokens, cache_read, 
                        cache_write)

    return Usage(feature=feature, model=model, served_by=served_by,
                 input_tokens=input_tokens, output_tokens=output_tokens,
                 cache_read=cache_read, cache_write=cache_write, cost_usd=cost,
                 latency_ms=latency_ms, fallback_used=fallback_used,
                 request_id=getattr(response, '_request_id', None))


def _redact_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy of messages with base64 image data replaced by its length."""
    out = copy.deepcopy(messages)
    for msg in out:
        content = msg.get('content')
        if not isinstance(content, list):
            continue

        for block in content:
            source = block.get('source') if isinstance(block, dict) else None
            if isinstance(source, dict) and source.get('type') == 'base64':
                source['data'] = f'<{len(source.get("data", ""))} base64 chars>'

    return out


def _dump(feature: str, 
model: str, 
system: str, 
messages: list[dict[str, Any]],
reply: Any, usage: Usage
) -> None:
    """Write the exchange to debug_prompts/ when SPHINX_DEBUG_PROMPTS is on."""
    if not settings.debug_prompts:
        return
    try:
        DEBUG_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path = DEBUG_DIR / f'{stamp}_{feature}.json'

        payload = {
            'feature': feature, 
            'model': model, 
            'system': system,
            'messages': _redact_images(messages),
            'reply': reply, 
            'usage': asdict(usage)
             }
        path.write_text(json.dumps(payload, ensure_ascii=False, 
                                    indent=2, default=str))

    except OSError as e:
        log.warning('debug_prompts dump failed: %s', e)
