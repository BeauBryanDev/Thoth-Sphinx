# YOLO-style annotated copy of the photo: one labelled box per detection.
# Line and label size scale with the image resolution.

from __future__ import annotations

import cv2
import numpy as np

from vision.spatial_primitives import UNKNOWN_CLASS, Detection

# BGR colours.
CARTOUCHE_COLOR = (0, 190, 255)
UNKNOWN_COLOR = (160, 160, 160)
SIGN_COLOR = (80, 200, 60)
LABEL_TEXT_COLOR = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
# Shorter image side per pixel of line thickness, and per unit of font scale.
THICKNESS_DIVISOR = 640
FONT_SCALE_DIVISOR = 1600
LABEL_MARGIN = 4
# Raw slot scores can pass 1.0; the label never shows more than this.
MAX_SHOWN_SCORE = 0.998


def _box_style(detection: Detection, thickness: int) -> tuple[tuple, int]:
    """Colour and line thickness: gold and thicker for a cartouche."""
    if detection.is_cartouche():
        return CARTOUCHE_COLOR, thickness * 2

    if detection.cls.lower() == UNKNOWN_CLASS:
        return UNKNOWN_COLOR, thickness

    return SIGN_COLOR, thickness


def _draw_label(
    image: np.ndarray,
    label: str,
    box: tuple[int, int, int, int],
    color: tuple,
    font_scale: float,
    thickness: int,
) -> None:
    """Filled label above the box, or below it when there is no room."""
    x1, y1, _, y2 = box
    (text_w, text_h), baseline = cv2.getTextSize(
        label, FONT, font_scale, thickness
    )

    text_y = y1 - LABEL_MARGIN
    if y1 - text_h - baseline - LABEL_MARGIN < 0:
        text_y = y2 + text_h + baseline + LABEL_MARGIN

    cv2.rectangle(
        image,
        (x1, text_y - text_h - baseline),
        (x1 + text_w + 2, text_y + baseline),
        color,
        -1,
    )
    cv2.putText(
        image,
        label,
        (x1 + 1, text_y),
        FONT,
        font_scale,
        LABEL_TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_detections(bgr: np.ndarray, 
                    detections: list[Detection]
                    ) -> np.ndarray:
    """Copy of the image with a class and confidence box per detection."""
    out = bgr.copy()
    short_side = min(out.shape[:2])
    thickness = max(1, round(short_side / THICKNESS_DIVISOR))
    font_scale = short_side / FONT_SCALE_DIVISOR

    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox)
        cls, score = detection.top3[0]
        color, box_thickness = _box_style(detection, thickness)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)
        _draw_label(
            out,
            f'{cls} {min(score, MAX_SHOWN_SCORE):.2f}',
            (x1, y1, x2, y2),
            color,
            font_scale,
            thickness,
        )

    return out
