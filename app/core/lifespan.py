import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
 
from fastapi import FastAPI
 
from app.core.config import settings
 
# Domain core lives at repo root, not under app/
# pipeline.py is the standalone module that wraps the full inference flow
from vision.pipeline import SphinxPipeline
# the pipeline module stays out of the app/ namespace for testing
 
logger = logging.getLogger('sphinxeyes.lifespan')
 
 
REQUIRED_ARTIFACTS: list[tuple[str, Path]] = [
    
    ('onnx_path',     settings.onnx_path),
    ('class_map',     settings.class_map),
    ('trie_pkl',      settings.trie_pkl),
    ('bbaw_parquet',  settings.bbaw_parquet),
    ('confusion_csv', settings.confusion_csv),
    
]


def _validate_artifacts() -> None:
    """
    Verifies every required artifact exists on disk before pipeline boot.
    Raises FileNotFoundError listing every missing path. Stopping early
    here is intentional: a half-loaded pipeline is worse than no pipeline.
    """
    missing = [
        f'  {name:14s} -> {path}'
        for name, path in REQUIRED_ARTIFACTS
        
        if not path.exists()
    ]
    if missing:
        joined = '\n'.join(missing)
        raise FileNotFoundError(
            f'SphinxEyes cannot start. Missing artifacts:\n{joined}\n'
            f'Verify the paths in .env or place the files at the expected '
            f'locations.'
        )
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Boot sequence:
        1. Validate every artifact path exists
        2. Construct SphinxPipeline (loads ONNX, trie, bigram, sub-cost matrix)
        3. Attach to app.state.pipeline
        4. Yield (server runs)
        5. On shutdown: log and let GC release resources
    """
    logger.info('SphinxEyes starting up')
    logger.info(f'Pipeline version: {settings.pipeline_version}')
    logger.info(f'ONNX path       : {settings.onnx_path}')
    logger.info(f'imgsz           : {settings.imgsz}')
 
    #  artifact validation
    try:
        _validate_artifacts()
        
    except FileNotFoundError as e:
        
        logger.error(str(e))
        
        raise
 
    # pipeline construction
    t0 = time.perf_counter()
    try:
        pipeline = SphinxPipeline(
            onnx_path     = settings.onnx_path,
            class_map     = settings.class_map,
            trie_pkl      = settings.trie_pkl,
            bbaw_parquet  = settings.bbaw_parquet,
            confusion_csv = settings.confusion_csv,
            imgsz         = settings.imgsz,
        )
    except Exception as e:
        
        logger.exception('SphinxPipeline construction failed')
        
        raise RuntimeError(
            
            f'SphinxPipeline failed to construct: {e}'
        ) from e
 
    # /health reads this via getattr(pipeline, 'version', 'v4')
    pipeline.version = settings.pipeline_version

    elapsed = time.perf_counter() - t0
    logger.info(f'Pipeline loaded in {elapsed:.2f}s')
 
    # Quick sanity check: confirm session is alive and classes are loaded
    try:
        
        n_classes  = len(pipeline.class_names)
        onnx_shape = pipeline.session.get_outputs()[0].shape
        logger.info(f'ONNX output shape: {onnx_shape}')
        logger.info(f'Class count     : {n_classes}')
        
        if n_classes != 150:
            
            logger.warning(
                
                f'Expected 150 classes, got {n_classes}. '
                f'Verify class_map matches the trained model.'
            )
            
    except Exception as e:
        
        logger.exception('Pipeline sanity check failed')
        
        raise RuntimeError(
            
            f'Pipeline loaded but sanity check failed: {e}'
            
        ) from e
 
    # attach
    app.state.pipeline = pipeline
    logger.info('SphinxEyes ready to serve requests')
 
    # yield to the server
    yield
 
    #  shutdown
    logger.info('SphinxEyes shutting down')
    # ONNX Runtime releases its session on GC; no manual close needed.
    # If pipeline.py ever opens a DB pool or HTTP client, close it here.
    app.state.pipeline = None
 