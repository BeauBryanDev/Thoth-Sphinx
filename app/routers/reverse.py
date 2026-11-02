
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.reverse_translation import (
    ReverseTranslateRequest,
    ReverseTranslationOut,
)
from app.services.reverse import ReverseTranslationService

router = APIRouter(prefix='/reverse', tags=['reverse'])

# POST /reverse/ — retro-translation: modern English -> Middle Egyptian.

# Pure LLM feature (Claude Sonnet composes the Egyptian; the CV pipeline is
# not involved). Stateless like the rest of the API.

@router.post('/', response_model=ReverseTranslationOut)
async def reverse_translate(body: ReverseTranslateRequest) -> ReverseTranslationOut:

    service = ReverseTranslationService()

    if not service.enabled:
        raise HTTPException(
            status_code=503,
            detail='Reverse translation unavailable: ANTHROPIC_API_KEY not configured.',
        )

    result = service.translate(body.text, body.register)

    if result.error:
        raise HTTPException(status_code=502,
                            detail=f'Reverse translation failed: {result.error}')

    return result
