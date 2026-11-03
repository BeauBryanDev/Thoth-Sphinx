from fastapi import APIRouter, Depends, HTTPException

from app import llm
from app.core import guards
from app.llm import images
from app.schemas.reading_chat import ReadingChatRequest, ReadingChatResponse
from app.services.reading_chat import ReadingChatService

router = APIRouter(prefix='/chat/reading', tags=['chat'])

# POST /chat/reading/ — talk with Thot-Sphinx about the reading on screen.
# Stateless: the client sends the reading snapshot and the conversation every turn,
# so the agent never starts from scratch and the server stores nothing.


@router.post('/', response_model=ReadingChatResponse)
def reading_chat(body: ReadingChatRequest, _: str = Depends(guards.check_llm)) -> ReadingChatResponse:
    service = ReadingChatService()
    if not service.enabled:
        raise HTTPException(status_code=503, detail='Chat is unavailable: ANTHROPIC_API_KEY not configured.')
    if not body.reading.lines:
        raise HTTPException(status_code=422, detail='reading has no lines')

    image_bgr = None
    if body.image:
        try:
            image_bgr = images.decode_data_url(body.image)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f'image: {e}') from e

    try:
        return service.reply(body.prompt, body.history, body.reading, image_bgr)
        
    except llm.LLMError as e:
        raise guards.upstream_error('reading_chat', e) from e
