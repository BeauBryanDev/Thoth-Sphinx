
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.sphinx_chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

# POST /chat/ — converse with Thot-Sphinx, guardian of temple knowledge.

# Stateless like the rest of the API: the client sends the conversation
# history each turn  — the LLM is stateless and remembers nothing.
class ChatTurn(BaseModel):
    role    : Literal['user', 'assistant']
    content : str


class ChatRequest(BaseModel):
    prompt  : str = Field(..., min_length=1, max_length=4000)
    history : list[ChatTurn] = []


class ChatResponse(BaseModel):
    reply      : str
    message_id : str


@router.post("/", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:

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

        raise HTTPException(status_code=502, detail=f"Chat backend error: {e}")

    return ChatResponse(reply=reply, message_id=message_id)
