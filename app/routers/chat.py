
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import guards
from app.core.guards import MAX_HISTORY_TURNS, TurnText
from app.schemas.books import BookPassage
from app.schemas.transliterations import ToolCallOut
from app.services.sphinx_chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

# POST /chat/ — converse with Thot-Sphinx, guardian of temple knowledge.

# Stateless like the rest of the API: the client sends the conversation
# history each turn  — the LLM is stateless and remembers nothing.
class ChatTurn(BaseModel):
    role    : Literal['user', 'assistant']
    content : TurnText


class ChatRequest(BaseModel):
    prompt  : str = Field(..., min_length=1, max_length=4000)
    history : list[ChatTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    riddle_session : str | None = Field(None, pattern=r'^[0-9a-f]{32}$')   # echoed from the last reply


class ChatResponse(BaseModel):
    reply      : str
    message_id : str
    tool_calls : list[ToolCallOut] = []   # knowledge-graph lookups and book searches made for this reply
    book_passages : list[BookPassage] = []   # top book passages search_books retrieved
    riddle_session : str | None = None   # set once a riddle trial has begun; send it back next turn


@router.post("/", response_model=ChatResponse)
async def chat(body: ChatRequest, _: str = Depends(guards.check_llm)) -> ChatResponse:

    service = ChatService()

    if not service.enabled:

        raise HTTPException(status_code=503,
                            detail="Chat is unavailable: ANTHROPIC_API_KEY not configured.")

    try:
        reply, message_id, tool_calls, book_passages, riddle_session = service.chat(
            body.prompt,
            [t.model_dump() for t in body.history],
            body.riddle_session,
        )
    except Exception as e:

        raise guards.upstream_error('chat', e) from e

    return ChatResponse(
        reply=reply,
        message_id=message_id,
        tool_calls=tool_calls,
        book_passages=book_passages,
        riddle_session=riddle_session,
    )
