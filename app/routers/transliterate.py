
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.ratelimit import Cooldown, client_key
from app.llm import images
from app.schemas.transliterations import TransliterateRequest, TransliterationOut
from app.services.transliteration_services import TransliterationService

router = APIRouter(prefix="/transliterate", tags=["transliterate"])

# Apply & re-read is the expensive human-in-the-loop path: once per client per window.
REREAD_LIMIT = Cooldown(settings.reread_cooldown_s)

# Split from /predict/ for latency hiding: the frontend fires detection
# as soon as the user picks a reading direction, lets them fill the
# archaeological-context form while the pipeline runs, then calls this
# endpoint with the detection output + context when they hit Decode.

@router.post("/", response_model=TransliterationOut)
async def transliterate(body: TransliterateRequest, request: Request) -> TransliterationOut:
# POST /transliterate/ — the LLM-only second stage.
    if not body.codes:
        raise HTTPException(status_code=422, detail="codes must not be empty")

    confs = body.confidences
    if len(confs) != len(body.codes):
        # missing/misaligned confidences -> neutral MED for all signs
        confs = [0.5] * len(body.codes)

    image_bgr = None
    if body.image:
        try:
            image_bgr = images.decode_data_url(body.image)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"image: {e}") from e

    # A re-read with the user's corrections is rate limited; a first decode is not.
    key = None
    if body.human_corrections:
        REREAD_LIMIT.window_s = settings.reread_cooldown_s
        key = client_key(request.client.host if request.client else None)
        wait = REREAD_LIMIT.reserve(key)
        if wait is not None:
            raise HTTPException(
                status_code=429,
                detail=f"One correction re-read per {settings.reread_cooldown_s // 60} minutes. "
                       f"Try again in {(wait + 59) // 60} min.",
                headers={"Retry-After": str(wait)},
            )

    try:
        result = TransliterationService().transliterate_sequence(
            body.codes,
            confs,
            body.boundary_hints,
            body.cartouche_names,
            direction = body.direction,
            layout    = body.layout,
            ctx       = body.context,
            image_bgr = image_bgr,
            human_corrections = body.human_corrections,
        )
    except Exception:
        if key:
            REREAD_LIMIT.release(key)   # a crash must not burn the user's hour
        raise
    if key and result.error:
        REREAD_LIMIT.release(key)       # nor a failed LLM call
    return result
