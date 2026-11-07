from __future__ import annotations

from typing import Annotated, Literal, Optional, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.core.guards import (
    MAX_CARTOUCHES,
    MAX_CODES,
    MAX_HISTORY_TURNS,
    CartoucheName,
    ReadingId,
    GardinerCode,
    NoteText,
    TurnText,
)
from app.schemas.books import BookPassage
from app.schemas.transliterations import LineEvidence, TextContext, ToolCallOut

# Schemas for POST /chat/reading/ — chat with Thot-Sphinx about the reading on screen.
# The server keeps the reading by reading_id (app/services/reading_sessions.py). The client's
# snapshot is only a fallback for a reading the server has lost, and it is capped field by field.

MAX_LINES = 60
MAX_CODES_PER_LINE = 200

Short = Annotated[str, StringConstraints(max_length=64)]
LineText = Annotated[str, StringConstraints(max_length=2000)]
FullText = Annotated[str, StringConstraints(max_length=8000)]


class ReadingLine(BaseModel):
    """One physical line of the reading: the YOLO codes and Claude's reading of them."""
    line            : int = Field(..., ge=1)
    codes           : list[GardinerCode] = Field(default_factory=list, max_length=MAX_CODES_PER_LINE)
    transliteration : LineText = ''
    gloss           : LineText = ''
    confidence      : Short = 'LOW'


class ReadingSnapshot(BaseModel):
    """The prediction under discussion, as the frontend holds it."""
    lines                : list[ReadingLine] = Field(default_factory=list, max_length=MAX_LINES)
    full_transliteration : FullText = ''
    full_translation     : FullText = ''
    cartouches           : list[CartoucheName] = Field(default_factory=list, max_length=MAX_CARTOUCHES)
    evidence             : list[LineEvidence] = Field(default_factory=list, max_length=MAX_LINES)
    image_notes          : list[NoteText] = Field(default_factory=list, max_length=20)
    model                : Short = ''
    escalated            : bool = False
    escalation_reasons   : list[Short] = Field(default_factory=list, max_length=20)
    second_opinion_reason: NoteText = ''   # the first reader's own words, from the router step of the trace
    direction            : Literal['rtl', 'ltr'] = 'rtl'
    layout               : Literal['rows', 'columns'] = 'rows'
    context              : TextContext = TextContext()

    @model_validator(mode='after')
    def _total_codes(self) -> Self:
        """The whole reading holds at most MAX_CODES signs, like a /transliterate/ request."""
        total = sum(len(ln.codes) for ln in self.lines)
        if total > MAX_CODES:
            raise ValueError(f'reading has {total} signs; at most {MAX_CODES}')
        return self


class ReadingChatTurn(BaseModel):
    role    : Literal['user', 'assistant']
    content : TurnText


class ReadingChatRequest(BaseModel):
    prompt  : str = Field(..., min_length=1, max_length=4000)
    history : list[ReadingChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    reading_id : Optional[ReadingId] = None        # the server's copy wins when it is still alive
    reading : Optional[ReadingSnapshot] = None     # fallback after a restart or expiry
    image   : Optional[str] = None   # raw upload as a data URL; the model sees it at 640 px


class SignCorrection(BaseModel):
    """A sign the agent proposes to change; the user decides whether to apply it."""
    line      : int                  # 1-based, as in the snapshot
    position  : int                  # 1-based position of the sign within that line
    from_code : str                  # the code there now
    to_code   : str                  # the proposed Gardiner code
    reason    : str


class ReadingChatResponse(BaseModel):
    reply       : str                          # markdown
    message_id  : str
    corrections : list[SignCorrection] = []    # validated against the snapshot
    tool_calls  : list[ToolCallOut] = []       # corpus calls made this turn
    book_passages : list[BookPassage] = []     # top book passages search_books retrieved
    model       : str = ''
    cost_usd    : Optional[float] = None
    reading_id  : Optional[str] = None             # set when the server's copy of the reading was used
    reading_version : Optional[int] = None
    state_source : Literal['server', 'client'] = 'client'   # whose copy of the reading the agent read
