"""
post_process_utils.py — thin Phase-5 wrapper.

spatial_logic.py owns all ONNX post-processing today (sigmoid, NMS, top-3).
This module exists as the correct future home if spatial_logic is ever split
for testability. At Phase 5 it re-exports the relevant helpers so callers
import from one place.
"""
from vision.spatial_logic import (   # noqa: F401
    postprocess_onnx,
    CONF_THRESHOLD,
    NMS_IOU,
)
