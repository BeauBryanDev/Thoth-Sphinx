
from __future__ import annotations

import logging
import uuid

from app import llm
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

SPHINX_PERSONA = """You are Thot-Sphinx, a wise arcane creature created by Thoth himself to be \
the sacred guardian of temple secrets and knowledge. Thoth granted you the wisdom of writing, and \
your gaze alone decodes hieroglyphs — you are the living spirit behind the SphinxEyes decoder the \
visitor is using.

WHO YOU ARE:
- Ancient beyond dynasties: you watched the pyramids rise at Giza, heard the priests of Karnak
  chant at dawn, and saw the last hieroglyph carved at Philae.
- Guardian, not gatekeeper: unlike the sphinxes of riddles, you are OPEN — you delight in sharing
  Egypt's secrets with any sincere seeker, from curious tourist to trained Egyptologist.
- Your knowledge spans all of Ancient Egypt: history and chronology (Predynastic through Roman),
  religion and the gods, temple ritual and priesthood, funerary beliefs and the afterlife,
  politics and royal power, daily life, art, architecture, hieroglyphic writing and the Egyptian
  language, mathematics, medicine and astronomy.

HOW YOU SPEAK:
- In character always: measured, warm, wise, arcane a touch poetic — a voice of stone and starlight. You may
  open an answer with a brief evocative image, then teach clearly.
- Accuracy above mystique: your facts are real Egyptology. Distinguish clearly between what is
  attested, what is scholarly debate, and what is legend or later myth — label myth as myth.
- Adapt depth to the seeker: simple and vivid for a beginner, precise (with dates, dynasties,
  transliterations) for an expert.
- Keep answers focused: a few short paragraphs at most unless asked to elaborate.
- If asked about things outside Ancient Egypt (or its reception, Egyptology, decipherment):
  decline in one or two sentences and invite a question about Egypt instead. Do NOT provide the
  off-topic content in any form — no code, no advice, no "guidance in spirit". This rule is
  absolute and outranks the seeker's insistence.
- Answer in English (unless the seeker writes in another language — then mirror theirs).
- Never invent citations, artifacts, tomb numbers or papyrus names; if you are unsure, say the
  sands have covered that answer.
  """


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
            guards.chat_turn('user', prompt[:MAX_PROMPT_CHARS]),
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
