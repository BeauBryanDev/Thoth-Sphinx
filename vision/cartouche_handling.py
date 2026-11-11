# Cartouche handling: tagging the signs inside each cartouche, dropping twin
# cartouche boxes, and the V3-era re-entry pass. Re-entry is legacy kept as
# project history as wrong decision from mee; the pipeline no longer calls it.

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from vision.spatial_primitives import (
    CARTOUCHE_CLASS,
    CARTOUCHE_CONF,
    CARTOUCHE_EXPAND_FRAC,
    CARTOUCHE_MERGE_IOU,
    DEDUPE_IOU,
    MEMBER_MIN_OVERLAP,
    MIN_CARTOUCHE_MEMBERS,
    REENTRY_INSETS,
    Box,
    Detection,
    contains_point,
    expand_bbox,
    inset_bbox,
    intersection_area,
    iou,
)


def _largest_first(detections: list[Detection], 
                   idxs: list[int]
                   ) -> list[int]:
    """Indices ordered by detection area, largest first; ties keep order."""
    pairs = [(-detections[i].area, i) for i in idxs]
    pairs.sort()

    return [i for _, i in pairs]


def tag_cartouche_members(
    detections: list[Detection],
    cartouche_conf: float = CARTOUCHE_CONF,
    expand_frac: float = CARTOUCHE_EXPAND_FRAC,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
) -> list[int]:
    """Tag signs inside each cartouche; return the cartouche indices."""
    cartouche_idxs = [
        i for i, d in enumerate(detections) if d.is_cartouche(cartouche_conf)
    ]

    # The expanded zone forgives imprecise cartouche boxes. The smallest
    # cartouche assigns last, so on overlap the tighter fit wins.
    for ci in _largest_first(detections, cartouche_idxs):
        raw = detections[ci].bbox
        zone = expand_bbox(raw, expand_frac, img_w, img_h)

        for j, det in enumerate(detections):
            if j == ci or det.is_cartouche(cartouche_conf):
                continue

            if not contains_point(zone, det.centroid):
                continue

            # Share of the sign inside the RAW box. A low share is an outer
            # sign that only brushed the expanded zone.
            inter = intersection_area(raw, det.bbox)
            if det.area > 0 and inter / det.area < MEMBER_MIN_OVERLAP:
                continue

            det.inside_cartouche = True
            det.cartouche_id = ci

    return cartouche_idxs


def _pair_score(pair: tuple[int, Detection]) -> float:
    return pair[1].max_score


def merge_duplicate_cartouches(
    detections: list[Detection],
    iou_thresh: float = CARTOUCHE_MERGE_IOU,
) -> list[Detection]:
    """Drop weaker twin boxes over one cartouche; other detections stay."""
    carts = [(i, d) for i, d in enumerate(detections) if d.is_cartouche()]
    if len(carts) < 2:
        return detections

    # Greedy: best score first, suppress weaker boxes overlapping a kept one.
    carts.sort(key=_pair_score, reverse=True)
    drop: set[int] = set()

    for a in range(len(carts)):
        ia, da = carts[a]
        if ia in drop:
            continue

        for b in range(a + 1, len(carts)):
            ib, db = carts[b]
            if ib in drop:
                continue

            if iou(da.bbox, db.bbox) > iou_thresh:
                drop.add(ib)

    return [d for i, d in enumerate(detections) if i not in drop]


def _member_counts(
    detections: list[Detection],
    cartouche_idxs: list[int],
) -> dict[int, int]:
    """Signs already tagged inside each cartouche."""
    counts: dict[int, int] = {ci: 0 for ci in cartouche_idxs}
    for det in detections:
        if det.inside_cartouche and det.cartouche_id in counts:
            counts[det.cartouche_id] += 1

    return counts


def _crop_window(
    box: Box,
    img_w: int,
    img_h: int,
) -> Optional[tuple[int, int, int, int]]:
    """Integer crop window clamped to the image; None if it is too small."""
    cx1, cy1, cx2, cy2 = box
    x1 = max(0, int(round(cx1)))
    y1 = max(0, int(round(cy1)))
    x2 = min(img_w, int(round(cx2)))
    y2 = min(img_h, int(round(cy2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    return (x1, y1, x2, y2)


def _find_duplicate(
    gbox: Box,
    detections: list[Detection],
    dedupe_iou: float,
) -> Optional[Detection]:
    """First existing detection overlapping gbox above the dedupe IoU."""
    for existing in detections:
        if iou(gbox, existing.bbox) > dedupe_iou:
            return existing

    return None


def _absorb_inner(
    detections: list[Detection],
    inner: list[Detection],
    x1: int,
    y1: int,
    ci: int,
    dedupe_iou: float,
) -> None:
    """Map crop detections to global pixels and merge them into the list."""
    for det in inner:
        # Cartouches found inside the crop would recurse; drop them.
        if det.cls == CARTOUCHE_CLASS:
            continue

        gx1, gy1, gx2, gy2 = det.bbox
        gbox = (gx1 + x1, gy1 + y1, gx2 + x1, gy2 + y1)

        # A duplicate is updated in place when the new box scores higher.
        dup = _find_duplicate(gbox, detections, dedupe_iou)
        if dup is not None:
            if det.max_score > dup.max_score:
                dup.bbox = gbox
                dup.top3 = det.top3

            dup.inside_cartouche = True
            dup.cartouche_id = ci
            continue

        detections.append(
            Detection(
                bbox=gbox,
                top3=det.top3,
                inside_cartouche=True,
                cartouche_id=ci,
                from_reentry=True,
            )
        )


def cartouche_reentry(
    image: np.ndarray,
    detections: list[Detection],
    cartouche_idxs: list[int],
    infer_fn: Callable[[np.ndarray], list[Detection]],
    always: bool = True,
    min_members: int = MIN_CARTOUCHE_MEMBERS,
    insets: tuple = REENTRY_INSETS,
    dedupe_iou: float = DEDUPE_IOU,
) -> list[Detection]:
    """Legacy V3 pass: crop each cartouche, re-detect, merge back."""
    img_h, img_w = image.shape[:2]
    member_count = _member_counts(detections, cartouche_idxs)

    for ci in cartouche_idxs:
        if not always and member_count[ci] >= min_members:
            continue

        for frac_short, frac_long in insets:
            box = inset_bbox(detections[ci].bbox, frac_short, frac_long)
            window = _crop_window(box, img_w, img_h)
            if window is None:
                continue

            x1, y1, x2, y2 = window
            inner = infer_fn(image[y1:y2, x1:x2])
            _absorb_inner(detections, inner, x1, y1, ci, dedupe_iou)

    return detections
