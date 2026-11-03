
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core import guards
from app.core.guards import MAX_HISTORY_TURNS, TurnText
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


class ChatResponse(BaseModel):
    reply      : str
    message_id : str


@router.post("/", response_model=ChatResponse)
async def chat(body: ChatRequest, _: str = Depends(guards.check_llm)) -> ChatResponse:

    service = ChatService()

    if not service.enabled:

        raise HTTPException(status_code=503,
                            detail="Chat is unavailable: ANTHROPIC_API_KEY not configured.")

    try:
        reply, message_id = service.chat(
            body.prompt,
            [t.model_dump() for t in body.history],
        )
    except Exception as e:

        raise guards.upstream_error('chat', e) from e

    return ChatResponse(reply=reply, message_id=message_id)
