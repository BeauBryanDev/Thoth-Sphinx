from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from pydantic import ValidationError

from app.core import guards
from app.core.config import settings
from app.routers.deps import get_pipeline
from app.schemas.predict import PredictResponse
from app.schemas.transliterations import TextContext
from app.services.sphinx_inference import InferenceService
from app.utils.image_utils import decode_upload, validate_image

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/", response_model=PredictResponse)
async def predict(
    request   : Request,
    file      : UploadFile      = File(..., description="Raw hieroglyph image (JPEG/PNG/WebP)"),
    direction : str             = Form("rtl", description="Reading direction: rtl or ltr"),
    layout    : str             = Form(...,  description="rows | columns (required — auto-detect is unreliable)"),
    preset    : str             = Form("none", description="Enhancement preset: none (recommended) | default | aggressive | gentle"),
    # LLM transliteration stage + archaeological context.
    # Everything defaults to 'unknown' — a naive tourist can leave it all blank.
    # Default FALSE: the web flow runs detection here, then calls
    # POST /transliterate/ separately (latency hiding — user fills the
    # context form while detection runs). True = one-shot for API users.
    translate     : bool = Form(False,     description="Also run the LLM transliteration stage (one-shot mode)"),
    period        : str  = Form("unknown", description="Historical period, e.g. new_kingdom"),
    text_type     : str  = Form("unknown", description="stela | temple_wall | tomb_wall | papyrus | ..."),
    support       : str  = Form("unknown", description="Physical support: limestone | sandstone | papyrus | ..."),
    location_type : str  = Form("unknown", description="pyramid | temple | tomb | museum | open_site"),
    site          : str  = Form("unknown", description="Site/location, e.g. Karnak, Saqqara"),
    dynasty       : str  = Form("unknown", description="Dynasty, e.g. IV, XVIII"),
    kings_reign   : str  = Form("unknown", description="King's reign, e.g. Thutmose III"),
    pipeline                    = Depends(get_pipeline),
) -> PredictResponse:
    
    if direction not in ("rtl", "ltr"):
        raise HTTPException(status_code=422, detail="direction must be 'rtl' or 'ltr'")
    
    if layout not in ("rows", "columns"):

        raise HTTPException(status_code=422, detail="layout must be 'rows' or 'columns'")
    
    if preset not in ("none", "default", "aggressive", "gentle"):

        raise HTTPException(status_code=422, detail="preset must be 'none', 'default', 'aggressive', or 'gentle'")

    data = await file.read()
    
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        
        raise HTTPException(status_code=413,
                            
                            detail=f"File exceeds {settings.max_upload_mb} MB limit")

    bgr = decode_upload(data)

    validate_image(bgr, max_mb=settings.max_upload_mb)

    try:
        context = TextContext(
            period        = period,
            text_type     = text_type,
            support       = support,
            location_type = location_type,
            site          = site or "unknown",
            dynasty       = dynasty or "unknown",
            kings_reign   = kings_reign or "unknown",
        )
    except ValidationError as e:

        raise HTTPException(status_code=422, detail=str(e))

    # Detection alone costs no tokens; one-shot transliteration counts like /transliterate/.
    key = None
    if translate:
        key = guards.check_llm(request)
        guards.reserve_translit(key)

    try:
        response = InferenceService(pipeline).run(
            bgr,
            direction = direction,
            layout    = layout,
            preset    = preset,
            translate = translate,
            context   = context,
        )
    except Exception:
        if key:
            guards.release_translit(key)
        raise
    if key and (response.transliteration is None or response.transliteration.error):
        guards.release_translit(key)
    return response
