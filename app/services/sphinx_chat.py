
from __future__ import annotations

import logging
import uuid

from app import llm
from app.llm.prompts import load
from app.core import guards
from app.core.config import settings

logger = logging.getLogger('sphinxeyes.chat')

MAX_HISTORY_MESSAGES = 20      # keep the context window (and cost) bounded
MAX_PROMPT_CHARS     = 4000
MAX_REPLY_TOKENS     = 16000   # thinking counts toward it; replies stay short via the persona

# SphinxChat — conversational endpoint persona (Phase 6).

# Thot-Sphinx: a wise arcane creature made by Thoth to stand as the
# sacred guardian of temple secrets and knowledge. Expert companion for
# everything Ancient Egypt. Runs on Claude Haiku through app/llm/;
# degrades gracefully when no API key is configured.

SPHINX_PERSONA = load('sphinx_persona')   # app/llm/prompts/


class ChatService:
    """Stateless chat turn: history comes from the client each call."""

    @property
    def enabled(self) -> bool:
        return llm.enabled()

    def chat(
        self,
        prompt  : str,
        history : list[dict],          # [{'role': 'user'|'assistant', 'content': str}, ...]
    ) -> tuple[str, str]:
        """Returns (reply, message_id). Raises RuntimeError when disabled, LLMError on failure."""
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

        # No temperature on current Claude models: the persona carries the tone.
        reply, _ = llm.complete_text(
            feature    = 'chat',
            model      = settings.haiku_model,
            system     = SPHINX_PERSONA + guards.USER_TEXT_RULE,
            messages   = messages,
            effort     = settings.chat_effort,
            max_tokens = MAX_REPLY_TOKENS,
        )
        return reply, uuid.uuid4().hex
