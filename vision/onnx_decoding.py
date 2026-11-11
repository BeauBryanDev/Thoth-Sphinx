# ONNX side of the spatial layer: raw tensor to detections (NMS and top-k)
# and the injected inference function built around an ONNX session.

from __future__ import annotations

from typing import Callable

import numpy as np

from vision.spatial_primitives import (
    CARTOUCHE_CLASS,
    CARTOUCHE_CONF,
    CONF_THRESHOLD,
    NMS_IOU,
    REENTRY_CONF,
    TOP_K,
    Detection,
    iou,
)


def nms_class_agnostic(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = NMS_IOU,
) -> list[int]:
    """Greedy class-agnostic NMS over [M, 4] boxes; best score first."""
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    suppressed = np.zeros(len(order), dtype=bool)

    for rank, i in enumerate(order):
        if suppressed[rank]:
            continue

        keep.append(int(i))
        bi = boxes_xyxy[i]

        for rank2 in range(rank + 1, len(order)):
            if suppressed[rank2]:
                continue

            if iou(tuple(bi), tuple(boxes_xyxy[order[rank2]])) > iou_thresh:
                suppressed[rank2] = True

    return keep


def _anchor_mask(
    scores: np.ndarray,
    max_scores: np.ndarray,
    class_names: list[str],
    conf_thresh: float,
) -> np.ndarray:
    """Anchors kept by score, plus weak ones whose best class is cartouche."""
    mask = max_scores >= conf_thresh
    if CARTOUCHE_CLASS not in class_names:
        return mask

    cart_row = class_names.index(CARTOUCHE_CLASS)
    is_cartouche = scores.argmax(axis=0) == cart_row
    weak_ok = scores[cart_row] >= CARTOUCHE_CONF

    return mask | (is_cartouche & weak_ok)


def _top_classes(
    col: np.ndarray,
    class_names: list[str],
    k: int,
) -> list[tuple[str, float]]:
    """The k best (class_name, score) pairs of one anchor, best first."""
    top_idx = np.argpartition(col, -k)[-k:]
    top_idx = top_idx[np.argsort(col[top_idx])[::-1]]

    return [(class_names[int(j)], float(col[j])) for j in top_idx]


def postprocess_onnx(
    raw: np.ndarray,
    class_names: list[str],
    conf_thresh: float = CONF_THRESHOLD,
    iou_thresh: float = NMS_IOU,
    top_k: int = TOP_K,
) -> list[Detection]:
    """Decode a [1, 154, N] or [154, N] nms=False export into Detections."""
    if raw.ndim == 3:
        raw = raw[0]

    n_classes = len(class_names)
    assert raw.shape[0] == 4 + n_classes, (
        f"channel mismatch: tensor has {raw.shape[0]}, "
        f"expected {4 + n_classes}"
    )

    # Rows 0 to 3 are cx, cy, w, h; the rest are independent sigmoid scores.
    boxes = raw[:4, :]
    scores = raw[4:, :]
    max_scores = scores.max(axis=0)

    mask = _anchor_mask(scores, max_scores, class_names, conf_thresh)
    if not mask.any():
        return []

    boxes_f = boxes[:, mask]
    scores_f = scores[:, mask]
    max_f = max_scores[mask]

    cx, cy, w, h = boxes_f
    xyxy = np.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        axis=1,
    )

    # Top-k runs on the NMS survivors only, never on all anchors.
    keep = nms_class_agnostic(xyxy, max_f, iou_thresh)
    k = min(top_k, n_classes)

    detections: list[Detection] = []
    for i in keep:
        top3 = _top_classes(scores_f[:, i], class_names, k)
        bbox = tuple(float(v) for v in xyxy[i])
        detections.append(Detection(bbox=bbox, top3=top3))

    return detections


def letterbox(
    bgr: np.ndarray,
    imgsz: int = 1024,
    pad_value: int = 114,
) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving square resize: (canvas, scale, dx, dy)."""
    import cv2

    h, w = bgr.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(bgr, (nw, nh))

    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    dx, dy = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized

    return canvas, scale, dx, dy


def make_onnx_infer_fn(
    session,
    class_names: list[str],
    conf_thresh: float = REENTRY_CONF,
    iou_thresh: float = NMS_IOU,
    imgsz: int = 1024,
    mode: str = 'letterbox',
) -> Callable[[np.ndarray], list[Detection]]:
    """Build infer_fn: BGR image in, Detections in image pixels out."""
    import cv2

    # letterbox matches the V4 training at 1024 and reproduces the Colab raw
    # YOLO output exactly. stretch is kept only for legacy callers.
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    def run(canvas: np.ndarray) -> list[Detection]:
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        raw = session.run([out_name], {inp_name: x})[0]

        return postprocess_onnx(raw, class_names, conf_thresh, iou_thresh)

    def infer_stretch(bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        dets = run(cv2.resize(bgr, (imgsz, imgsz)))
        sx, sy = w / imgsz, h / imgsz
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

        return dets

    def infer_letterbox(bgr: np.ndarray) -> list[Detection]:
        canvas, scale, dx, dy = letterbox(bgr, imgsz)
        dets = run(canvas)
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = (
                (x1 - dx) / scale,
                (y1 - dy) / scale,
                (x2 - dx) / scale,
                (y2 - dy) / scale,
            )

        return dets

    if mode == 'stretch':
        return infer_stretch

    if mode == 'letterbox':
        return infer_letterbox

    raise ValueError(f"mode must be 'stretch' or 'letterbox', got {mode!r}")
