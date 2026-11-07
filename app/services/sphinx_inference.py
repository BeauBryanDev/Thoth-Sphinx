from __future__ import annotations

import base64
from typing import Optional

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.cartouches import CartoucheOut
from app.schemas.corrections import CorrectionOut, SegmentedWordOut, ResolvedUnknownOut
from app.schemas.predict import PredictResponse, OuterOut
from app.schemas.transliterations import TextContext
from app.services.transliteration_services import TransliterationService


class InferenceService:
    """
    Single responsibility: call SphinxPipeline.run() and translate its raw
    dict output into the PredictResponse schema.
    No image decoding, no HTTP concerns, no artifact loading here.
    """

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    def run(
        self,
        bgr       : np.ndarray,
        *,
        direction : str           = 'rtl',
        layout    : Optional[str] = None,
        preset    : str           = 'none',
        translate : bool          = False,
        context   : Optional[TextContext] = None,
    ) -> PredictResponse:
        # preset='none' skips enhancement entirely — A/B testing showed
        # enhance() deletes real signs and hallucinates fakes on clean images.
        raw = self._pipeline.run(
            bgr,
            direction   = direction,
            layout      = layout,
            use_enhance = preset != 'none',
            preset      = preset if preset != 'none' else 'default',
            annotate    = True,
            use_corrector = settings.use_corrector,
        )
        response = self._map(raw)
        response.annotated_image = self._encode_annotated(
            raw.get('annotated_bgr'))

        if translate:
            response.transliteration = TransliterationService().transliterate(
                raw, context or TextContext(), image_bgr=bgr,   # raw upload, pre-enhance
            )
        return response

 
    # Private mapping helpers

    @staticmethod
    def _encode_annotated(bgr: Optional[np.ndarray]) -> Optional[str]:
        """Annotated ndarray -> JPEG data URL (None if unavailable)."""
        if bgr is None:
            return None
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return None
        return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')

    @staticmethod
    def _map_correction(c: dict) -> CorrectionOut:
        return CorrectionOut(
            segmented_words   = [SegmentedWordOut(**w) for w in c['segmented_words']],
            unknowns_resolved = [ResolvedUnknownOut(**u) for u in c['unknowns_resolved']],
            flat_corrected_seq = c['flat_corrected_seq'],
            flat_translit      = c['flat_translit'],
            flat_translation   = c['flat_translation'],
            score              = c['score'],
            had_fallback       = c['had_fallback'],
        )

    @staticmethod
    def _map_cartouche(c: dict) -> CartoucheOut:
        return CartoucheOut(
            bbox          = tuple(c['bbox']),
            n_members     = c['n_members'],
            inferred      = c['inferred'],
            translit      = c.get('translit'),
            english       = c.get('english'),
            spelling      = c.get('spelling'),
            score         = c.get('score'),
            aligned_codes = c.get('aligned_codes'),
            verified      = c.get('verified'),
            interior_codes = [slot[0][0] for slot in c.get('slots', [])
                              if slot] or None,
        )

    def _map(self, raw: dict) -> PredictResponse:
        " Map the raw pipeline output to the schema. "
        outer_raw = raw['outer']
        outer = OuterOut(
            slots          = outer_raw['slots'],
            boundary_hints = outer_raw['boundary_hints'],
            n_synthetic    = outer_raw['n_synthetic'],
            correction     = self._map_correction(outer_raw['correction']),
        )
        return PredictResponse(
            layout       = raw['layout'],
            direction    = raw['direction'],
            image_shape  = tuple(raw['image_shape']),
            n_detections = raw['n_detections'],
            n_cartouches = raw['n_cartouches'],
            outer        = outer,
            cartouches   = [self._map_cartouche(c) for c in raw['cartouches']],
        )
