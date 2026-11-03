
from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from app import llm
from app.llm.prompts import load
from app.core.config import settings
from app.core.guards import (
    CODE_RE,
    USER_TEXT_RULE,
    fence_user_text,
    public_message,
)
from app.schemas.reverse_translation import (
    ReverseTranslationOut,
    ReverseWord,
)
# Retro-translation service: modern English -> Middle Egyptian.

# It is a LLM feature — the vision pipeline plays no part. The model works
# in the three staged steps of a composition exercise:
#   1. normalize the English into translatable Middle Egyptian semantics
#      (Egyptian has no word for 'internet'; it does for 'scribe');
#   2. compose real Middle Egyptian: VSO grammar, suffix pronouns,
#      honorific transposition, then the Leiden transliteration;
#   3. encode each word as Gardiner signs: phonograms + phonetic
#      complements + determinatives, as a scribe would spell it.

# Degrades gracefully like the other LLM services: no API key or any
# LLM failure returns a ReverseTranslationOut with `error` set.

logger = logging.getLogger('sphinxeyes.reverse')

# Canonical Gardiner code shape: category letter(s) + number + optional
# variant letter, e.g. M17, Aa1, N35, W25a. Used to sanitize LLM output.
_CODE_RE = CODE_RE

SYSTEM_PROMPT = load('reverse_translation')   # app/llm/prompts/


MAX_REPLY_TOKENS = 16000   # adaptive thinking counts toward it


class _WordReply(BaseModel):
    """One word of the composition."""
    english         : str
    transliteration : str
    gardiner_codes  : list[str]
    literal         : str
    note            : str


class _ReverseReply(BaseModel):
    """The Middle Egyptian composition."""  # sent as the schema description; fields mirror SYSTEM_PROMPT's OUTPUT
    normalized_english : str
    transliteration    : str
    gardiner_codes     : list[str]
    words              : list[_WordReply]
    grammar_notes      : str
    confidence         : Literal['HIGH', 'MEDIUM', 'LOW']


def _build_user_prompt(text: str, register: str) -> str:
    reg = ('a neutral monumental register'
           if register == 'unknown' else f'the {register} register')
    return (
        f'Render the following English into Middle Egyptian, using {reg}.\n'
        f'ENGLISH TEXT:\n{fence_user_text(text)}\n\n'
        f'Apply the three stages and answer with the JSON object only.'
    )


def _clean_codes(codes: object) -> list[str]:
    """Keep only syntactically valid canonical Gardiner codes."""
    if not isinstance(codes, list):
        return []
    out = []
    for c in codes:
        c = str(c).strip()
        if _CODE_RE.match(c):
            out.append(c)
            
        else:
            logger.warning(f'reverse: dropped malformed Gardiner code {c!r}')
            
    return out


class ReverseTranslationService:
    """One-shot English -> Middle Egyptian composition via Claude Sonnet."""

    @property
    def enabled(self) -> bool:
        return llm.enabled()

    def translate(self, text: str,
                  register: str = 'unknown'
                  ) -> ReverseTranslationOut:

        if not self.enabled:

            return ReverseTranslationOut(
                source_text=text, normalized_english='', transliteration='',
                gardiner_codes=[], model=settings.sonnet_model,
                error='ANTHROPIC_API_KEY not configured — reverse translation disabled.',
            )

        try:
            parsed, usage = llm.complete_parsed(
                feature    = 'reverse',
                model      = settings.sonnet_model,
                system     = SYSTEM_PROMPT + USER_TEXT_RULE,
                messages   = [
                    {
                        'role': 'user',
                        'content': _build_user_prompt(text, register),
                    },
                ],
                schema     = _ReverseReply,
                effort     = settings.reverse_effort,
                max_tokens = MAX_REPLY_TOKENS,
            )

        except Exception as e:
            logger.exception('reverse translation LLM call failed')

            return ReverseTranslationOut(
                source_text=text, normalized_english='', transliteration='',
                gardiner_codes=[], model=settings.sonnet_model,
                error=public_message(e),
            )

        words = [
            ReverseWord(
                english         = w.english,
                transliteration = w.transliteration,
                gardiner_codes  = _clean_codes(w.gardiner_codes),
                literal         = w.literal,
                note            = w.note,
            )
            for w in parsed.words
        ]

        flat = _clean_codes(parsed.gardiner_codes)

        if not flat and words:                      # fall back to per-word codes
            flat = [c for w in words for c in w.gardiner_codes]

        return ReverseTranslationOut(
            source_text        = text,
            normalized_english = parsed.normalized_english,
            transliteration    = parsed.transliteration,
            gardiner_codes     = flat,
            words              = words,
            grammar_notes      = parsed.grammar_notes,
            confidence         = parsed.confidence,
            model              = usage.served_by,
        )
