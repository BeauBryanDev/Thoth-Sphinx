from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.transliterations import LineEvidence, TextContext, ToolCallOut

# Schemas for POST /chat/reading/ — chat with Thot-Sphinx about the reading on screen.
# Stateless like /transliterate/: the client echoes the reading back every turn.

MAX_LINES = 60
MAX_CODES_PER_LINE = 200


class ReadingLine(BaseModel):
    """One physical line of the reading: the YOLO codes and Claude's reading of them."""
    line            : int = Field(..., ge=1)
    codes           : list[str] = Field(default_factory=list, max_length=MAX_CODES_PER_LINE)
    transliteration : str = ''
    gloss           : str = ''
    confidence      : str = 'LOW'


class ReadingSnapshot(BaseModel):
    """The prediction under discussion, as the frontend holds it."""
    lines                : list[ReadingLine] = Field(default_factory=list, max_length=MAX_LINES)
    full_transliteration : str = ''
    full_translation     : str = ''
    cartouches           : list[str] = []
    evidence             : list[LineEvidence] = []
    image_notes          : list[str] = []
    model                : str = ''
    escalated            : bool = False
    escalation_reasons   : list[str] = []
    second_opinion_reason: str = ''      # the first reader's own words, from the router step of the trace
    direction            : Literal['rtl', 'ltr'] = 'rtl'
    layout               : Literal['rows', 'columns'] = 'rows'
    context              : TextContext = TextContext()


class ReadingChatTurn(BaseModel):
    role    : Literal['user', 'assistant']
    content : str


class ReadingChatRequest(BaseModel):
    prompt  : str = Field(..., min_length=1, max_length=4000)
    history : list[ReadingChatTurn] = []
    reading : ReadingSnapshot
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
    model       : str = ''
    cost_usd    : Optional[float] = None
