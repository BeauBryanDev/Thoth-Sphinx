
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.llm import images
from app.schemas.transliterations import TransliterateRequest, TransliterationOut
from app.services.transliteration_services import TransliterationService

router = APIRouter(prefix="/transliterate", tags=["transliterate"])

# Split from /predict/ for latency hiding: the frontend fires detection
# as soon as the user picks a reading direction, lets them fill the
# archaeological-context form while the pipeline runs, then calls this
# endpoint with the detection output + context when they hit Decode.

@router.post("/", response_model=TransliterationOut)
async def transliterate(body: TransliterateRequest) -> TransliterationOut:
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

    return TransliterationService().transliterate_sequence(
        body.codes,
        confs,
        body.boundary_hints,
        body.cartouche_names,
        direction = body.direction,
        layout    = body.layout,
        ctx       = body.context,
        image_bgr = image_bgr,
    )
