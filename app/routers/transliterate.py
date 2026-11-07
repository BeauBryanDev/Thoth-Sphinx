from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core import guards
from app.core.config import settings
from app.core.ratelimit import Cooldown
from app.llm import images
from app.schemas.transliterations import TransliterateRequest, TransliterationOut
from app.services.reading_sessions import READING_SESSIONS, snapshot_from
from app.services.transliteration_services import TransliterationService

router = APIRouter(prefix="/transliterate", tags=["transliterate"])

# Apply & re-read is the expensive human-in-the-loop path: once per client per window.
REREAD_LIMIT = Cooldown(settings.reread_cooldown_s)

# Split from /predict/ for latency hiding: the frontend fires detection
# as soon as the user picks a reading direction, lets them fill the
# archaeological-context form while the pipeline runs, then calls this
# endpoint with the detection output + context when they hit Decode.

@router.post("/", response_model=TransliterationOut)
async def transliterate(body: TransliterateRequest,
                        key: str = Depends(guards.check_llm)) -> TransliterationOut:
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

    # A re-read with the user's corrections has its own cooldown on top of the daily quota.
    reread = bool(body.human_corrections)
    if reread:
        REREAD_LIMIT.window_s = settings.reread_cooldown_s
        wait = REREAD_LIMIT.reserve(key)
        if wait is not None:
            raise guards.refuse(
                429, "reread",
                f"One correction re-read per {settings.reread_cooldown_s // 60} minutes. "
                f"Try again in {(wait + 59) // 60} min.",
                wait,
            )

    def refund() -> None:
        """A crash or a failed LLM call must not burn the user's quota."""
        guards.release_translit(key)
        if reread:
            REREAD_LIMIT.release(key)

    try:
        guards.reserve_translit(key)
    except HTTPException:
        if reread:
            REREAD_LIMIT.release(key)
        raise

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
        refund()
        raise
    if result.error:
        refund()
        return result
    # Agent state: the chat reads the server's copy; a re-read keeps the reading's ID.
    snapshot = snapshot_from(result, body)
    if snapshot is not None:
        result.reading_id, result.reading_version = READING_SESSIONS.save(
            body.reading_id, snapshot, body.human_corrections)
    return result
