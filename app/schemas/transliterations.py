
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.core.guards import MAX_CARTOUCHES, MAX_CODES, CartoucheName, ContextText, GardinerCode


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
    site            : ContextText  = 'unknown'   # free text: "Karnak", "Saqqara"...
    dynasty         : ContextText  = 'unknown'   # free text: "IV", "XVIII"...
    kings_reign     : ContextText  = 'unknown'   # free text: "Thutmose III"...


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


class HumanCorrection(BaseModel):
    """A sign the user checked against the stone and corrected (human in the loop)."""
    line      : int = Field(..., ge=1)        # 1-based line of the previous reading
    position  : int = Field(..., ge=1)        # 1-based sign position within that line
    from_code : GardinerCode                  # what the detector had
    to_code   : GardinerCode                  # what the user confirmed


class TransliterateRequest(BaseModel):
    """
    Body for POST /transliterate/ — the LLM-only second stage.
    The frontend echoes back what /predict/ returned (stateless server):
    corrected codes, per-slot top-1 confidences, line boundaries and any
    matched cartouches, plus the user-filled archaeological context.
    """
    # Capped and pattern-checked: every item reaches the prompt, and the count sets the cost.
    codes           : list[GardinerCode] = Field(..., max_length=MAX_CODES)
    confidences     : list[float] = Field(default_factory=list, max_length=MAX_CODES)  # empty -> 0.5 everywhere
    boundary_hints  : list[int]   = Field(default_factory=list, max_length=MAX_CODES)
    cartouche_names : list[CartoucheName] = Field(default_factory=list, max_length=MAX_CARTOUCHES)  # "mn-xpr-ra — Thutmose III (...)"
    direction       : Literal['rtl', 'ltr'] = 'rtl'
    layout          : Literal['rows', 'columns'] = 'rows'
    context         : TextContext = TextContext()
    image           : Optional[str] = None   # raw upload as a data URL; the LLM sees it at 640 px
    human_corrections : list[HumanCorrection] = Field(default_factory=list, max_length=50)


class CorpusMatch(BaseModel):
    """The closest corpus sentence for a line (Thesaurus Linguae Aegyptiae)."""
    sentence_id      : str
    rating           : str                     # strong | possible | weak | unrated
    transliteration  : str = ''
    translation      : str = ''
    translation_lang : Literal['en', 'de'] = 'de'
    date             : Optional[str] = None
    text_name        : Optional[str] = None
    matched_by       : list[str] = []          # search channels: sign, translit


class LineEvidence(BaseModel):
    """What the corpus says about one line of the reading."""
    line           : int                       # 1-based, as in the prompt
    verdict        : str                       # strong | possible | none | too_short | unrated | error
    query_translit : Optional[str] = None      # the Unicode query actually searched
    best           : Optional[CorpusMatch] = None
    searched_by    : str                       # a model id, or 'evidence_check' when code searched


class ToolCallOut(BaseModel):
    """One tool call a model made during the read."""
    model      : str
    tool       : str
    input      : dict = {}
    summary    : str
    ok         : bool
    latency_ms : int


class TraceEvent(BaseModel):
    """One step of the read, in order (spec §11). No secrets, no image data."""
    step          : int
    t_ms          : int                        # since the read started
    kind          : str                        # detection | llm_call | tool_call | router | escalation | evidence_check | final | error
    actor         : str                        # model id, tool name, router, pipeline
    summary       : str
    latency_ms    : Optional[int] = None
    input_tokens  : Optional[int] = None
    output_tokens : Optional[int] = None
    cost_usd      : Optional[float] = None
    detail        : dict = {}


class TransliterationOut(BaseModel):
    """Full-image transliteration assembled from all chunks."""
    chunks               : list[ChunkTransliteration]
    full_transliteration : str
    full_translation     : str
    model                : str
    n_chunks             : int
    error                : Optional[str] = None   # set if the LLM stage failed
    escalated            : bool = False           # a Sonnet second opinion produced this reading
    escalation_reasons   : list[str] = []         # router reason codes (app/llm/router.py)
    escalation_note      : Optional[str] = None   # e.g. the second opinion failed; Haiku's reading kept
    cost_usd             : Optional[float] = None # estimated, all calls of this read
    image_notes          : list[str] = []         # Claude's notes on the photograph (whole-wall reads only)
    evidence             : list[LineEvidence] = []   # closest corpus match per line
    tool_calls           : list[ToolCallOut] = []    # every tool call the models made
    trace                : list[TraceEvent] = []     # the whole read, step by step
