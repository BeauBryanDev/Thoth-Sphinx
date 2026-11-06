import logging
import re
import time
from typing import Literal

from pydantic import BaseModel

from app import llm
from app.core.config import settings
from app.core.guards import public_message
from app.llm import tools
from app.llm.prompts import load
from app.schemas.transliterations import SemanticTranslation, ToolCallOut, TranslationPassage

logger = logging.getLogger('sphinxeyes.refine_translation')

# Second translation: search the published translations with the LLM's English, then smooth it against them.

SYSTEM_PROMPT = load('translation_polish')
MAX_PASSAGE_TEXT = 3000           # characters of each spell sent to the editor (whole spell, small-to-big)
PYRAMID_KINGS = {                 # cartouche English name -> king column of english_translations
    'Unas': 'UNIS',
    'Teti': 'TETI',
    'Pepi I ': 'PEPI I',
    'Merenre': 'MERENRE',
    'Pepi II': 'PEPI II',
}


class _Refined(BaseModel):
    """The editor's reply."""
    translation  : str
    match        : Literal['same_text', 'related', 'none']
    used_passage : str
    note         : str


def pyramid_king(cartouche_names: list[str]) -> str | None:
    """The first matched cartouche whose king owns a pyramid with texts, as the table spells it."""
    for name in cartouche_names:
        if name.startswith('[UNRESOLVED') or ' — ' not in name:
            continue
        english = name.split(' — ', 1)[1] + ' '
        for label, king in PYRAMID_KINGS.items():
            if english.startswith(label):
                return king
    return None


def clean_query(translation: str) -> str:
    """The LLM translation without doubt marks and ellipses, which only add noise to the embedding."""
    text = re.sub(r'\(\?\)|\[\.\.\.\]|…|\.\.\.', ' ', translation)
    return re.sub(r'\s+', ' ', text).strip()


def _passages_block(passages: list[TranslationPassage]) -> str:
    """The passages as the editor reads them."""
    return '\n\n'.join(
        f'[{p.passage_id}] {p.citation} (relevance {p.rerank_score if p.rerank_score is not None else p.similarity})\n'
        f'{(p.spell_text or p.text)[:MAX_PASSAGE_TEXT]}'
        for p in passages
    )


def refine(translation: str, cartouche_names: list[str]) -> tuple[SemanticTranslation, ToolCallOut | None]:
    """Search, then refine; failures land in .error and never break the reading."""
    started = time.monotonic()
    query = clean_query(translation)
    king = pyramid_king(cartouche_names)
    out = SemanticTranslation(query=query, king_filter=king)
    call = None
    try:
        search_started = time.monotonic()
        result = tools.search_translations(query, king=king, limit=settings.semantic_top_k)
        out.search_mode = result['mode']
        out.passages = [TranslationPassage(**p) for p in result['passages']]
        call = ToolCallOut(
            model='pipeline',
            tool='search_translations',
            input={'query': query[:200], 'king': king},
            summary=f'{len(out.passages)} published passages'
                    + (f', best: {out.passages[0].citation}' if out.passages else ''),
            ok=True,
            latency_ms=int((time.monotonic() - search_started) * 1000),
        )
    except Exception as e:  # database down: the refinement still runs on the translation alone
        logger.warning('translation search failed: %s', e)
        call = ToolCallOut(model='pipeline', tool='search_translations', input={'king': king},
                           summary=f'search failed: {public_message(e)}', ok=False, latency_ms=0)
    prompt = (f'FIRST TRANSLATION:\n{translation}\n\n'
              + (f'KING OF THE CARTOUCHE: {king}\n\n' if king else '')
              + 'PUBLISHED PASSAGES:\n' + (_passages_block(out.passages) or '(none found)'))
    try:
        refined, usage = llm.complete_parsed(
            feature='semantic_translation',
            model=settings.semantic_model,
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user', 'content': prompt}],
            schema=_Refined,
            effort=settings.semantic_effort,
            max_tokens=4096,
        )
        out.translation = refined.translation
        out.match = refined.match
        out.used_passage = refined.used_passage or None
        out.note = refined.note
        out.model = usage.served_by
        out.cost_usd = usage.cost_usd
    except Exception as e:  # the raw LLM translation still ships
        logger.warning('semantic translation failed: %s', e)
        out.error = public_message(e)
    out.latency_ms = int((time.monotonic() - started) * 1000)
    return out, call
