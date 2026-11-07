from fastapi import Request
from vision.pipeline import SphinxPipeline


def get_pipeline(request: Request) -> SphinxPipeline:
    """Dependency: reads the pipeline loaded by lifespan onto app.state."""
    return request.app.state.pipeline
