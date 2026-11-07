from fastapi import APIRouter, Depends, HTTPException

from app import llm
from app.core import guards
from app.llm import images
from app.schemas.reading_chat import ReadingChatRequest, ReadingChatResponse
from app.services.reading_chat import ReadingChatService
from app.services.reading_sessions import READING_SESSIONS

router = APIRouter(prefix='/chat/reading', tags=['chat'])

# POST /chat/reading/ — talk with Thot-Sphinx about the reading on screen.
# The reading comes from the server session (reading_id) when it is alive; the client's
# snapshot is the fallback. The conversation itself is still sent by the client every turn.


@router.post('/', response_model=ReadingChatResponse)
def reading_chat(body: ReadingChatRequest, _: str = Depends(guards.check_llm)) -> ReadingChatResponse:
    service = ReadingChatService()
    if not service.enabled:
        raise HTTPException(status_code=503, detail='Chat is unavailable: ANTHROPIC_API_KEY not configured.')
    state = READING_SESSIONS.get(body.reading_id)
    if state is not None:
        reading, source = state.snapshot, 'server'
    elif body.reading is not None:
        reading, source = body.reading, 'client'
    else:
        raise HTTPException(status_code=410, detail='This reading has expired; decode the image again.')
    if not reading.lines:
        raise HTTPException(status_code=422, detail='reading has no lines')

    image_bgr = None
    if body.image:
        try:
            image_bgr = images.decode_data_url(body.image)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f'image: {e}') from e

    try:
        reply = service.reply(body.prompt, body.history, reading, image_bgr,
                              corrections=state.corrections if state else [])
        
    except llm.LLMError as e:
        raise guards.upstream_error('reading_chat', e) from e
    reply.state_source = source
    if state is not None:
        reply.reading_id, reply.reading_version = body.reading_id, state.version
    return reply
