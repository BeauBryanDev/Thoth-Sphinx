
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
 
from app.core.config import settings
from app.core.exception import register_exception_handlers
from app.core.lifespan import lifespan
from app.routers import predict, health, transliterate, chat, reverse, reading_chat
 
 
def create_app() -> FastAPI:
    app = FastAPI(
        title       = 'SphinxEyes',
        version     = settings.pipeline_version,
        description = (
            'Archaeological computer vision API for Middle Egyptian '
            'hieroglyph detection and transliteration.'
        ),
        lifespan    = lifespan,
    )
 
    # CORS: open during development; on the VPS set CORS_ORIGINS='["https://<your-domain>"]'.
    # No cookies are used (the bearer token is a header), so credentials stay off.
    # The guard headers are exposed so the browser can show "try again at HH:MM".
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = settings.cors_origins,
        allow_credentials = False,
        allow_methods     = ['*'],
        allow_headers     = ['*'],
        expose_headers    = ['Retry-After', 'X-Guard'],
    )
 
    register_exception_handlers(app)

    # Routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(transliterate.router)
    app.include_router(chat.router)
    app.include_router(reading_chat.router)
    app.include_router(reverse.router)
 
    return app
 
 
app = create_app()
 