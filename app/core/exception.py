
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger('sphinxeyes.exception')


class InferenceError(Exception):
    """Pipeline ran but could not produce a result for this image."""


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(InferenceError)
    async def inference_error_handler(request: Request, exc: InferenceError):
        return JSONResponse(
            status_code = 422,
            content     = {'error': 'inference_failed', 'detail': str(exc)},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        # Log which field failed (input values left out: they can be whole images).
        errors = [{k: e.get(k) for k in ('loc', 'type', 'msg')} for e in exc.errors()]
        logger.warning(f'422 on {request.method} {request.url.path}: {errors}')
        return JSONResponse(status_code=422, content={'detail': errors})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        error_id = uuid.uuid4().hex[:8]
        logger.exception(f'Unhandled error [{error_id}] on {request.method} {request.url.path}')
        return JSONResponse(
            status_code = 500,
            content     = {
                'error'    : 'internal_error',
                'detail'   : 'Unexpected server error. Quote the error_id when reporting.',
                'error_id' : error_id,
            },
        )
