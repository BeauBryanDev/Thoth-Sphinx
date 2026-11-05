
from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel

from app import llm
from app.llm import agent_tools
from app.llm.prompts import load
from app.core import guards
from app.core.config import settings
from app.schemas.books import BookPassage
from app.schemas.transliterations import ToolCallOut
from app.services.riddle_sessions import RIDDLE_SESSIONS

logger = logging.getLogger('sphinxeyes.chat')

MAX_HISTORY_MESSAGES = 20      # keep the context window (and cost) bounded
MAX_PASSAGES         = 5       # book passages shown beside the chat
MAX_PROMPT_CHARS     = 4000
MAX_REPLY_TOKENS     = 16000   # thinking counts toward it; replies stay short via the persona

# SphinxChat — conversational endpoint persona (Phase 6).

# Thot-Sphinx: a wise arcane creature made by Thoth to stand as the
# sacred guardian of temple secrets and knowledge. Expert companion for
# everything Ancient Egypt. Runs on Claude Haiku through app/llm/;
# degrades gracefully when no API key is configured.

SPHINX_PERSONA = load('sphinx_persona')   # app/llm/prompts/
CHAT_TOOLS_RULES = '\n\n' + load('chat_tools')   # appended: the persona stays word for word
CHAT_TOOLS = (agent_tools.LOOKUP_ENTITY, agent_tools.SEARCH_BOOKS)


class _ChatAnswer(BaseModel):
    """Thot-Sphinx's answer, in markdown."""
    reply : str


def book_passages(calls: list[llm.ToolCall]) -> list[BookPassage]:
    """Passages from this reply's search_books calls, first seen first, no repeats, at most MAX_PASSAGES."""
    seen: dict[str, BookPassage] = {}
    for call in calls:
        if call.tool != 'search_books' or not call.ok or not call.result:
            continue
        weak = bool(call.result.get('no_book_support'))
        for passage in call.result.get('passages', []):
            seen.setdefault(
                passage['chunk_id'],
                BookPassage(**passage, weak=weak),
            )
    return list(seen.values())[:MAX_PASSAGES]


class ChatService:
    """Stateless chat turn: history comes from the client each call."""

    @property
    def enabled(self) -> bool:
        return llm.enabled()

    def chat(
        self,
        prompt  : str,
        history : list[dict],          # [{'role': 'user'|'assistant', 'content': str}, ...]
        riddle_session : str | None = None,
    ) -> tuple[str, str, list[ToolCallOut], list[BookPassage], str | None]:
        """Returns (reply, message_id, tool_calls, book_passages, riddle_session). Raises RuntimeError when disabled, LLMError on failure."""
        if not self.enabled:
            raise RuntimeError('ANTHROPIC_API_KEY not configured — chat disabled.')

        trimmed = [
            guards.chat_turn(
                m['role'],
                str(m['content'])[:MAX_PROMPT_CHARS],
            )
            for m in history[-MAX_HISTORY_MESSAGES:]
            if m.get('role') in ('user', 'assistant') and m.get('content')
        ]
        # Claude needs the conversation to open with a user turn (the UI greeting is an assistant turn).
        while trimmed and trimmed[0]['role'] != 'user':
            trimmed.pop(0)
        messages = trimmed + [
            llm.cache_turn(guards.chat_turn('user', prompt[:MAX_PROMPT_CHARS])),   # read back next turn
        ]

        session_id, game = RIDDLE_SESSIONS.get(riddle_session)
        game.begin_turn()   # one riddle action per seeker message

        # No temperature on current Claude models: the persona carries the tone.
        answer, _, calls = llm.complete_parsed_with_tools(
            feature        = 'chat',
            model          = settings.haiku_model,
            system         = SPHINX_PERSONA + CHAT_TOOLS_RULES + guards.USER_TEXT_RULE,
            messages       = messages,
            schema         = _ChatAnswer,
            effort         = settings.chat_effort,
            tools          = CHAT_TOOLS + agent_tools.riddle_tools(game),
            max_tool_turns = settings.chat_tool_budget,
            max_tokens     = MAX_REPLY_TOKENS,
        )
        tool_calls = [
            ToolCallOut(
                model=c.model,
                tool=c.tool,
                input=c.input,
                summary=c.summary,
                ok=c.ok,
                latency_ms=c.latency_ms,
            )
            for c in calls
        ]
        stored = RIDDLE_SESSIONS.keep(session_id, game)
        return (
            answer.reply,
            uuid.uuid4().hex,
            tool_calls,
            book_passages(calls),
            session_id if stored else None,
        )
