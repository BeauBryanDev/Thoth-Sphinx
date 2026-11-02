
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


# Controlled vocabularies. 'unknown' is always first = default.
Period = Literal[
    'unknown', 'old_kingdom', 'first_intermediate', 'middle_kingdom',
    'second_intermediate', 'new_kingdom', 'third_intermediate',
    'late_period', 'ptolemaic', 'roman',
]

TextType = Literal[
    'unknown', 'stela', 'temple_wall', 'tomb_wall', 'papyrus',
    'sarcophagus', 'obelisk', 'statue', 'offering_table', 'pyramid_texts',
]

Support = Literal[
    'unknown', 'limestone', 'sandstone', 'granite', 'papyrus',
    'wood', 'plaster', 'faience', 'metal',
]

LocationType = Literal[
    'unknown', 'pyramid', 'temple', 'tomb', 'museum', 'open_site',
]

# Pydantic contracts for the LLM transliteration stage (Phase 6).

# TextContext carries the archaeological metadata the user supplies about
# the uploaded image. EVERY field defaults to 'unknown' — a tourist who
# knows nothing about the artifact can still get a reading; an Egyptologist
# who fills everything in gets a much better one.
class TextContext(BaseModel):
    """User-supplied archaeological context. All optional, all default unknown."""
    period          : Period       = 'unknown'
    text_type       : TextType     = 'unknown'
    support         : Support      = 'unknown'
    location_type   : LocationType = 'unknown'
    site            : str          = 'unknown'   # free text: "Karnak", "Saqqara"...
    dynasty         : str          = 'unknown'   # free text: "IV", "XVIII"...
    kings_reign     : str          = 'unknown'   # free text: "Thutmose III"...


class ChunkTransliteration(BaseModel):
    """One LLM reading for one chunk of the sign sequence."""
    chunk_index      : int
    gardiner_codes   : list[str]
    transliteration  : str
    english_gloss    : str
    linguistic_notes : str = ''
    confidence       : Literal['HIGH', 'MEDIUM', 'LOW'] = 'LOW'
    period_note      : str = ''
    is_cartouche     : bool = False


class TransliterateRequest(BaseModel):
    """
    Body for POST /transliterate/ — the LLM-only second stage.
    The frontend echoes back what /predict/ returned (stateless server):
    corrected codes, per-slot top-1 confidences, line boundaries and any
    matched cartouches, plus the user-filled archaeological context.
    """
    codes           : list[str]
    confidences     : list[float] = []       # empty -> assume 0.5 everywhere
    boundary_hints  : list[int]   = []
    cartouche_names : list[str]   = []       # e.g. "mn-xpr-ra — Thutmose III (...)"
    direction       : Literal['rtl', 'ltr'] = 'rtl'
    layout          : Literal['rows', 'columns'] = 'rows'
    context         : TextContext = TextContext()
    image           : Optional[str] = None   # raw upload as a data URL; the LLM sees it at 640 px


class TransliterationOut(BaseModel):
    """Full-image transliteration assembled from all chunks."""
    chunks               : list[ChunkTransliteration]
    full_transliteration : str
    full_translation     : str
    model                : str
    n_chunks             : int
    error                : Optional[str] = None   # set if the LLM stage failed
