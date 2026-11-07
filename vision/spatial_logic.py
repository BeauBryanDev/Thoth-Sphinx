#!/usr/bin/env python3
"""
spatial_logic.py — Phase 2 spatial layer for SphinxEyes (steps 1 + 2).

Role in the pipeline
--------------------
    ONNXRuntime raw output [1, 154, N]
        --> postprocess_onnx()       (conf filter -> NMS on max score -> top-3)
        --> List[Detection]          (bbox, centroid, top3, area)
        --> tag_cartouche_members()  (PRIMARY: centroid containment)
        --> cartouche_reentry()      (FALLBACK: inset crop + re-inference,
                                      only when containment found < 2 members)
        --> [next: quadrat clustering -> reading order -> sphinx_corrector]

Design decisions
----------------
    1. NMS is CLASS-AGNOSTIC. One physical glyph predicted as two confusable
       classes must collapse to ONE Detection; the alternatives survive in
       top3. Per-class NMS would emit duplicate boxes for the corrector.

    2. Top-3 extraction runs only on NMS survivors (~dozens), never on the
       full ~21k anchors (fragile breakpoint #7 in CLAUDE.md).

    3. Cartouche interiors come from RE-ENTRY, always. V3 was intentionally
       trained with no labels inside cartouches (curriculum decision), so
       the model is systematically blind there at global resolution. Every
       cartouche gets an inset crop + second inference; inside the crop the
       bracket context is gone and the model sees an ordinary sign column —
       its training regime. Containment tagging still runs first: it
       catches stray interior detections the model emits anyway, and the
       re-entry dedupe reconciles them.

    4. Re-entry inference is injected as `infer_fn(crop) -> list[Detection]`
       (bboxes in crop coordinates). Tests mock it; production passes the
       ONNX wrapper. Inner-pass `cartouche` detections are dropped to break
       recursion (fragile breakpoint #3); inner detections duplicating an
       existing global one (IoU > DEDUPE_IOU) update it in place when they
       score higher, instead of appending a twin.

Usage
-----
    from vision.spatial_logic import Detection, postprocess_onnx, \
        tag_cartouche_members, cartouche_reentry

    raw  = sess.run([out.name], {inp.name: x})[0]      # [1, 154, N]
    dets = postprocess_onnx(raw, class_names)
    cartouche_idxs = tag_cartouche_members(dets)
    dets = cartouche_reentry(img, dets, cartouche_idxs, infer_fn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CARTOUCHE_CLASS         = 'cartouche'
CONF_THRESHOLD          = 0.15   # min max-class score to keep an anchor
NMS_IOU                 = 0.50   # class-agnostic NMS threshold
TOP_K                   = 3      # alternatives kept per detection
CARTOUCHE_CONF          = 0.075  # min score to treat a det as a cartouche
                                 # (lowered 0.25->0.05: the V9 detector under-
                                 # scores cartouches even on trained images;
                                 # missing the cartouche loses the royal name)
                                 # (was 0.50; real cartouches on weathered
                                 # stone surface at ~0.28 — sandstone_wall)
CARTOUCHE_MERGE_IOU     = 0.30   # two 'cartouche' boxes overlapping above
                                 # this collapse to the higher-scoring one.
                                 # Lower than the 0.50 global NMS on purpose:
                                 # lowering CARTOUCHE_CONF admits weak twin
                                 # boxes over ONE real cartouche (IoU ~0.3-0.5)
                                 # that NMS leaves alone. Cartouche-only.
CARTOUCHE_EXPAND_FRAC   = 0.10   # expand cartouche bbox before containment
MEMBER_MIN_OVERLAP      = 0.20   # min fraction of a sign's area overlapping
                                 # the RAW cartouche bbox to count as member
                                 # (blocks adjacent outer signs the expanded
                                 # zone would otherwise swallow)
# Re-entry crop insets as (short_axis_frac, long_axis_frac) per side.
# Two complementary passes whose results are unioned via the dedupe step:
#   shallow (6%/6%)  — keeps signs hugging the bracket ends
#   deep    (6%/10%) — cuts bracket curve + tie-knot, zooms interior more
# Empirically (test_image_vn5, 6 cartouches): each pass alone misses signs
# the other finds; the union covers 6/6 with the best score per sign.
REENTRY_INSETS          = ((0.06, 0.06), (0.06, 0.10))
REENTRY_CONF            = 0.20   # conf threshold for the re-entry pass —
                                 # lower than global: interior signs were
                                 # never labeled in training, scores run low
MIN_CARTOUCHE_MEMBERS   = 2      # legacy heuristic (always=False only)
DEDUPE_IOU              = 0.50   # inner det vs global det dedupe threshold

# Step 3 — quadrat clustering
DUP_OVERLAP             = 0.55   # intersection/min-area above which two
                                 # boxes are the SAME physical sign
QUADRAT_ALIGN           = 0.50   # min projection-overlap ratio along the
                                 # reading axis to share a quadrat
QUADRAT_GAP_FRAC        = 0.60   # max perpendicular gap (x median size)
QUADRAT_SIZE_RATIO      = (0.5, 2.0)   # sqrt-area ratio guard (plan rule)
QUADRAT_MAX_SIGNS       = 4      # split components larger than this
QUADRAT_MAX_EXTENT      = 2.5    # max merged cross-extent (x median sign
                                 # size). A merged pair spanning more has
                                 # crossed into the neighbouring row/column
                                 # (sandstone_wall: g5+l2 spanned 2.9x).
                                 # NOT lower: two equal stacked signs are
                                 # ~2.2x, must stay mergeable.

# Step 4 — line assembly + gap insertion
LINE_GAP_FRAC           = 0.60   # cross-axis jump (x median) = new line
MISSING_GAP_FRAC        = 1.50   # reading-axis gap (x median step) above
                                 # which synthetic Unknown slots go in
MAX_GAP_INSERTS         = 2      # max synthetic slots per gap
UNKNOWN_CLASS           = 'unknown'   # YOLO-space name; glue maps to the
                                      # trie's 'Unknown' token


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """One detected sign. bbox is (x1, y1, x2, y2) in global image pixels."""
    bbox             : tuple[float, float, float, float]
    top3             : list[tuple[str, float]]   # [(class_name, score)] desc
    inside_cartouche : bool = False
    cartouche_id     : Optional[int] = None      # index of parent cartouche
    from_reentry     : bool = False              # came from the fallback pass

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cls(self) -> str:
        return self.top3[0][0]

    @property
    def max_score(self) -> float:
        return self.top3[0][1]

    def is_cartouche(self, conf: float = CARTOUCHE_CONF) -> bool:
        return self.cls == CARTOUCHE_CLASS and self.max_score > conf


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def iou(a: tuple, b: tuple) -> float:
    """IoU of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def expand_bbox(
    bbox: tuple, frac: float, img_w: Optional[float] = None,
    img_h: Optional[float] = None,
) -> tuple:
    """Grow a bbox by `frac` of its own size on each side; clamp to image."""
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
    if img_w is not None:
        x1, x2 = max(0.0, x1), min(float(img_w), x2)
    if img_h is not None:
        y1, y2 = max(0.0, y1), min(float(img_h), y2)
    return (x1, y1, x2, y2)


def inset_bbox(
    bbox: tuple, frac: float, frac_long: Optional[float] = None,
) -> tuple:
    """
    Shrink a bbox by `frac` of its own size on each side (re-entry crop).

    If `frac_long` is given, the LONG axis of the box is inset by that
    fraction instead. For a cartouche the bracket curve and the tie-knot
    sit at the long-axis ends, so cutting deeper there removes them while
    keeping the signs (which span the short axis nearly edge to edge).
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    fl = frac if frac_long is None else frac_long
    if h >= w:                      # vertical cartouche: long axis = y
        dx, dy = w * frac, h * fl
    else:                           # horizontal cartouche: long axis = x
        dx, dy = w * fl, h * frac
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def contains_point(bbox: tuple, pt: tuple) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


# ---------------------------------------------------------------------------
# Step 1 — ONNX postprocess: raw tensor -> List[Detection]
# ---------------------------------------------------------------------------

def nms_class_agnostic(
    boxes_xyxy: np.ndarray,    # [M, 4]
    scores    : np.ndarray,    # [M]
    iou_thresh: float = NMS_IOU,
) -> list[int]:
    """Greedy class-agnostic NMS. Returns kept indices, score-descending."""
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


def postprocess_onnx(
    raw         : np.ndarray,         # [1, 4+C, N] or [4+C, N]
    class_names : list[str],
    conf_thresh : float = CONF_THRESHOLD,
    iou_thresh  : float = NMS_IOU,
    top_k       : int   = TOP_K,
) -> list[Detection]:
    """
    Decode the V3 ONNX output (nms=False export) into Detection objects.

    Layout per anchor column: rows 0-3 = (cx, cy, w, h) in input-image
    pixels; rows 4..4+C-1 = independent sigmoid class scores.

    Order of operations (fragile breakpoint #7): conf-filter on max class
    score -> class-agnostic NMS -> top-k extraction on survivors only.
    """
    if raw.ndim == 3:
        raw = raw[0]
    C = len(class_names)
    assert raw.shape[0] == 4 + C, \
        f"channel mismatch: tensor has {raw.shape[0]}, expected {4 + C}"

    boxes  = raw[:4, :]              # [4, N]  cx, cy, w, h
    scores = raw[4:, :]              # [C, N]

    max_scores = scores.max(axis=0)  # [N]
    mask = max_scores >= conf_thresh

    # Cartouche bypass: the V9 detector chronically UNDER-scores cartouches
    # (OOD scale/context — see degubbing_cartouches.md), and dropping the
    # cartouche box loses the royal name entirely. So keep any anchor whose
    # TOP class is 'cartouche' down to CARTOUCHE_CONF, even when that is
    # below the global conf gate. Only the cartouche class gets this relief.
    try:
        cart_row = class_names.index(CARTOUCHE_CLASS)
        cart_argmax = scores.argmax(axis=0) == cart_row
        mask = mask | (cart_argmax & (scores[cart_row] >= CARTOUCHE_CONF))
    except ValueError:
        pass  # no cartouche class in this model — nothing to relax

    if not mask.any():
        return []

    boxes_f  = boxes[:, mask]
    scores_f = scores[:, mask]
    max_f    = max_scores[mask]

    cx, cy, w, h = boxes_f
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep = nms_class_agnostic(xyxy, max_f, iou_thresh)

    detections: list[Detection] = []
    k = min(top_k, C)
    for i in keep:
        col = scores_f[:, i]
        top_idx = np.argpartition(col, -k)[-k:]
        top_idx = top_idx[np.argsort(col[top_idx])[::-1]]
        top3 = [(class_names[int(j)], float(col[j])) for j in top_idx]
        detections.append(Detection(bbox=tuple(float(v) for v in xyxy[i]),
                                    top3=top3))
    return detections


# ---------------------------------------------------------------------------
# Step 2a — Cartouche containment tagging (PRIMARY path)
# ---------------------------------------------------------------------------

def tag_cartouche_members(
    detections    : list[Detection],
    cartouche_conf: float = CARTOUCHE_CONF,
    expand_frac   : float = CARTOUCHE_EXPAND_FRAC,
    img_w         : Optional[float] = None,
    img_h         : Optional[float] = None,
) -> list[int]:
    """
    Tag every detection whose centroid sits inside an (expanded) cartouche
    bbox with inside_cartouche=True and the cartouche's index.

    The expansion compensates for the model's imprecise cartouche boxes;
    centroid containment (vs bbox-IoU) keeps signs touching the bracket
    correctly tagged. A detection inside two overlapping cartouches is
    assigned to the smaller one (tighter fit wins).

    Guard (2026-07-19): a sign must ALSO overlap the RAW cartouche bbox by
    >= MEMBER_MIN_OVERLAP of its own area. The expansion alone swallowed
    outer-text signs sitting just above the cartouche (padding_cartouche_1:
    the two X1 of nsw-bity, ~6% overlap, vanished from the outer sequence).
    Bracket-clipped true members overlap far more (~38% in the self-test).

    Returns the indices of the cartouche detections themselves.
    """
    cartouche_idxs = [
        i for i, d in enumerate(detections) if d.is_cartouche(cartouche_conf)
    ]
    # Smaller cartouches assign last -> tighter fit wins on overlap
    for ci in sorted(cartouche_idxs, key=lambda i: -detections[i].area):
        raw = detections[ci].bbox
        zone = expand_bbox(raw, expand_frac, img_w, img_h)
        for j, det in enumerate(detections):
            if j == ci or det.is_cartouche(cartouche_conf):
                continue
            if not contains_point(zone, det.centroid):
                continue
            # overlap of the sign's own area with the RAW cartouche box
            ox = max(0.0, min(raw[2], det.bbox[2]) - max(raw[0], det.bbox[0]))
            oy = max(0.0, min(raw[3], det.bbox[3]) - max(raw[1], det.bbox[1]))
            if det.area > 0 and (ox * oy) / det.area < MEMBER_MIN_OVERLAP:
                continue
            det.inside_cartouche = True
            det.cartouche_id = ci
    return cartouche_idxs


# ---------------------------------------------------------------------------
# Step 2b — Cartouche re-entry (FALLBACK path)
# ---------------------------------------------------------------------------

def cartouche_reentry(
    image          : np.ndarray,                          # HxWx3 global image
    detections     : list[Detection],
    cartouche_idxs : list[int],
    infer_fn       : Callable[[np.ndarray], list[Detection]],
    always         : bool  = True,
    min_members    : int   = MIN_CARTOUCHE_MEMBERS,
    insets         : tuple = REENTRY_INSETS,
    dedupe_iou     : float = DEDUPE_IOU,
) -> list[Detection]:
    """
    Inset-crop each cartouche, re-run inference on the crop(s), and map the
    results back to global coordinates.

    V3 was INTENTIONALLY trained with no labels inside cartouches
    (curriculum-learning decision), so the model is systematically blind
    there at global resolution. Re-entry is therefore the PRIMARY mechanism
    for cartouche interiors — `always=True` re-enters every cartouche.
    Set always=False to fall back to the legacy heuristic (re-enter only
    when containment tagged fewer than `min_members` signs).

    Each cartouche is cropped once per (short_frac, long_frac) pair in
    `insets` and the passes are UNIONED: results of earlier passes are
    appended before later passes run, so the dedupe step reconciles them,
    keeping the higher-scoring reading of each sign. Shallow insets keep
    bracket-hugging signs; deep insets remove the bracket curve / tie-knot
    and zoom the interior. Inside the crop the bracket context is gone, so
    the model sees an ordinary sign column — its training regime.

    infer_fn contract: takes an HxWx3 crop, returns list[Detection] with
    bboxes in CROP pixel coordinates (any internal resize is its business —
    see make_onnx_infer_fn for the reference implementation).

    Inner-pass rules:
      - inner `cartouche` detections are dropped (breaks recursion,
        fragile breakpoint #3 — bracket leakage shows up as this class)
      - an inner det overlapping an existing det (IoU > dedupe_iou)
        updates that det in place if it scores higher; never appended twice
      - surviving inner dets are tagged inside_cartouche / from_reentry

    Returns the (extended) detection list. Re-entry results never trigger
    another re-entry.
    """
    img_h, img_w = image.shape[:2]

    member_count: dict[int, int] = {ci: 0 for ci in cartouche_idxs}
    for det in detections:
        if det.inside_cartouche and det.cartouche_id in member_count:
            member_count[det.cartouche_id] += 1

    for ci in cartouche_idxs:
        if not always and member_count[ci] >= min_members:
            continue

        for frac_short, frac_long in insets:
            cx1, cy1, cx2, cy2 = inset_bbox(
                detections[ci].bbox, frac_short, frac_long
            )
            x1 = max(0, int(round(cx1)));  y1 = max(0, int(round(cy1)))
            x2 = min(img_w, int(round(cx2)));  y2 = min(img_h, int(round(cy2)))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue                  # degenerate crop, skip

            crop = image[y1:y2, x1:x2]
            inner = infer_fn(crop)

            for det in inner:
                if det.cls == CARTOUCHE_CLASS:
                    continue              # recursion / bracket leakage
                gx1, gy1, gx2, gy2 = det.bbox
                gbox = (gx1 + x1, gy1 + y1, gx2 + x1, gy2 + y1)

                # Dedupe against every det so far (including earlier
                # passes' output): update in place if better
                dup = None
                for existing in detections:
                    if iou(gbox, existing.bbox) > dedupe_iou:
                        dup = existing
                        break
                if dup is not None:
                    if det.max_score > dup.max_score:
                        dup.bbox = gbox
                        dup.top3 = det.top3
                    dup.inside_cartouche = True
                    dup.cartouche_id = ci
                    continue

                detections.append(Detection(
                    bbox             = gbox,
                    top3             = det.top3,
                    inside_cartouche = True,
                    cartouche_id     = ci,
                    from_reentry     = True,
                ))

    return detections


# ---------------------------------------------------------------------------
# Step 3 — DSU quadrat clustering
# ---------------------------------------------------------------------------

class DSU:
    """Union-Find with path compression + union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def overlap_ratio(a: tuple, b: tuple) -> float:
    """Intersection / min(area). Better than IoU for the double-box case:
    a thin box fully inside a taller box scores ~1.0 here but low IoU."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(min(area_a, area_b), 1e-9)


def merge_duplicate_boxes(
    detections     : list[Detection],
    overlap_thresh : float = DUP_OVERLAP,
) -> list[Detection]:
    """
    Collapse multiple detections of the SAME physical sign into one slot
    (fragile breakpoint #5). Survives NMS because the duplicate boxes have
    IoU < 0.5 (e.g. a thin f31 box inside a taller s29 box on one stroke).

    Two detections merge iff intersection/min-area > overlap_thresh, both
    are non-cartouche, and they share the same cartouche context. DSU
    handles transitivity. The highest-scoring member keeps its bbox and
    identity; top-3s of the group are unioned (max score per code, top 3).

    `cartouche_id` indices are remapped to the returned list's positions
    (cartouche detections never merge, so they always survive).
    """
    n = len(detections)
    dsu = DSU(n)
    for i in range(n):
        di = detections[i]
        if di.cls == CARTOUCHE_CLASS:
            continue
        for j in range(i + 1, n):
            dj = detections[j]
            if dj.cls == CARTOUCHE_CLASS:
                continue
            if (di.inside_cartouche, di.cartouche_id) != \
               (dj.inside_cartouche, dj.cartouche_id):
                continue
            if overlap_ratio(di.bbox, dj.bbox) > overlap_thresh:
                dsu.union(i, j)

    merged: list[Detection] = []
    seen_root: set[int] = set()
    for i in range(n):                      # preserve original order
        root = dsu.find(i)
        if root in seen_root:
            continue
        seen_root.add(root)
        group = [detections[k] for k in range(n) if dsu.find(k) == root]
        base = max(group, key=lambda d: d.max_score)
        if len(group) > 1:
            scores: dict[str, float] = {}
            for d in group:
                for c, s in d.top3:
                    scores[c] = max(scores.get(c, 0.0), s)
            base.top3 = sorted(scores.items(),
                               key=lambda kv: kv[1], reverse=True)[:3]
        merged.append(base)

    # Remap cartouche_id (old index -> new index of the same object)
    new_idx = {id(obj): k for k, obj in enumerate(merged)}
    for d in merged:
        if d.cartouche_id is not None:
            d.cartouche_id = new_idx[id(detections[d.cartouche_id])]
    return merged


def merge_duplicate_cartouches(
    detections : list[Detection],
    iou_thresh : float = CARTOUCHE_MERGE_IOU,
) -> list[Detection]:
    """
    Collapse twin cartouche boxes over the SAME physical cartouche.

    `merge_duplicate_boxes` deliberately never merges cartouches, and the
    global class-agnostic NMS only fires above IoU 0.50 — so when a lowered
    CARTOUCHE_CONF admits a second, weaker box over one real cartouche at
    IoU ~0.3-0.5, both survive and the panel reads two cartouches where
    there is one. This pass dedups cartouche-vs-cartouche only, at a lower
    IoU, keeping the higher-scoring box. Non-cartouche detections pass
    through untouched and in place.
    """
    carts = [(i, d) for i, d in enumerate(detections) if d.is_cartouche()]
    if len(carts) < 2:
        return detections

    # Greedy: score-descending, suppress lower-scoring cartouches that
    # overlap a kept one above the threshold.
    carts.sort(key=lambda t: t[1].max_score, reverse=True)
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


@dataclass
class Quadrat:
    """One visual block of 1-4 signs sharing a slot in the reading order."""
    members: list[Detection]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (min(d.bbox[0] for d in self.members),
                min(d.bbox[1] for d in self.members),
                max(d.bbox[2] for d in self.members),
                max(d.bbox[3] for d in self.members))

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def ordered(self, direction: str = 'rtl') -> list[Detection]:
        """
        Within-quadrat reading order: top-to-bottom bands, then ltr/rtl
        inside each band. Band break = y-centroid jump > 0.5 x median
        member height.
        """
        if len(self.members) <= 1:
            return list(self.members)
        med_h = float(np.median([d.height for d in self.members]))
        by_y = sorted(self.members, key=lambda d: d.centroid[1])
        bands: list[list[Detection]] = [[by_y[0]]]
        for d in by_y[1:]:
            band_y = np.mean([m.centroid[1] for m in bands[-1]])
            if d.centroid[1] - band_y > 0.5 * med_h:
                bands.append([d])
            else:
                bands[-1].append(d)
        out: list[Detection] = []
        for band in bands:
            band.sort(key=lambda d: d.centroid[0], reverse=(direction == 'rtl'))
            out.extend(band)
        return out


def _axis_overlap(a: tuple, b: tuple, axis: int) -> float:
    """Projection-overlap ratio of two bboxes on x (axis=0) or y (axis=1),
    normalized by the smaller extent."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    inter = min(a[hi], b[hi]) - max(a[lo], b[lo])
    if inter <= 0:
        return 0.0
    return inter / max(min(a[hi] - a[lo], b[hi] - b[lo]), 1e-9)


def _split_component(
    members   : list[Detection],
    stack_axis: int,                  # 0 = x (columns layout), 1 = y (rows)
    max_signs : int,
) -> list[list[Detection]]:
    """Recursively split an oversized component at its largest gap along
    the stack axis."""
    if len(members) <= max_signs:
        return [members]
    members = sorted(members, key=lambda d: d.centroid[stack_axis])
    gaps = [members[k + 1].centroid[stack_axis] - members[k].centroid[stack_axis]
            for k in range(len(members) - 1)]
    cut = int(np.argmax(gaps)) + 1
    return (_split_component(members[:cut], stack_axis, max_signs)
            + _split_component(members[cut:], stack_axis, max_signs))


def cluster_quadrats(
    detections    : list[Detection],
    layout        : str   = 'columns',          # 'rows' | 'columns'
    align_overlap : float = QUADRAT_ALIGN,
    gap_frac      : float = QUADRAT_GAP_FRAC,
    size_ratio    : tuple = QUADRAT_SIZE_RATIO,
    max_signs     : int   = QUADRAT_MAX_SIGNS,
) -> list[Quadrat]:
    """
    Group detections into quadrats via DSU connected components.

    ANTI-CHAINING RULE: two signs share a quadrat only if they stack
    PERPENDICULAR to the reading axis —
      layout='rows'    (horizontal reading): vertically stacked signs
                       (x-projection overlap > align_overlap, y-gap small)
      layout='columns' (vertical reading):   side-by-side signs
                       (y-projection overlap > align_overlap, x-gap small)
    Signs adjacent ALONG the reading axis never merge, so a crowded row
    can't chain into one giant component (the failure mode of the naive
    centroid-distance rule).

    Additional guards: sqrt-area ratio within `size_ratio`; components
    larger than `max_signs` split at their largest perpendicular gap.
    Cartouche-class detections never merge (each is its own quadrat).
    Caller chooses the subset: outer text = not inside_cartouche;
    cartouche interiors = per-cartouche member lists.

    Returns quadrats sorted by centroid (y, then x) for determinism;
    line-level reading order is step 4's job.
    """
    if not detections:
        return []
    # axis along which quadrat-mates align = reading axis
    read_axis  = 0 if layout == 'rows' else 1     # x for rows, y for columns
    stack_axis = 1 - read_axis

    sign_dets = [d for d in detections if d.cls != CARTOUCHE_CLASS]
    med_stack = (float(np.median([(d.width if stack_axis == 0 else d.height)
                                  for d in sign_dets]))
                 if sign_dets else 1.0)

    n = len(detections)
    dsu = DSU(n)
    for i in range(n):
        di = detections[i]
        if di.cls == CARTOUCHE_CLASS:
            continue
        for j in range(i + 1, n):
            dj = detections[j]
            if dj.cls == CARTOUCHE_CLASS:
                continue
            if _axis_overlap(di.bbox, dj.bbox, read_axis) < align_overlap:
                continue
            lo, hi = (0, 2) if stack_axis == 0 else (1, 3)
            gap = max(di.bbox[lo], dj.bbox[lo]) - min(di.bbox[hi], dj.bbox[hi])
            if gap > gap_frac * med_stack:
                continue
            # Cross-line guard: a merged pair spanning more than
            # QUADRAT_MAX_EXTENT sign-sizes on the stack axis has leaked
            # into the neighbouring row/column, even if the gap is tiny
            # (adjacent rows can sit closer than intra-quadrat stacks).
            extent = max(di.bbox[hi], dj.bbox[hi]) - min(di.bbox[lo], dj.bbox[lo])
            if extent > QUADRAT_MAX_EXTENT * med_stack:
                continue
            r = (di.area / max(dj.area, 1e-9)) ** 0.5
            if not (size_ratio[0] <= r <= size_ratio[1]):
                continue
            dsu.union(i, j)

    quadrats: list[Quadrat] = []
    for idxs in dsu.groups().values():
        members = [detections[k] for k in idxs]
        for part in _split_component(members, stack_axis, max_signs):
            quadrats.append(Quadrat(members=part))
    quadrats.sort(key=lambda q: (q.centroid[1], q.centroid[0]))
    return quadrats


# ---------------------------------------------------------------------------
# Step 4 — line assembly: quadrats -> reading order + boundary hints
# ---------------------------------------------------------------------------

@dataclass
class ReadingOrder:
    """Final spatial-layer output, ready for sphinx_corrector.correct()."""
    slots           : list[list[tuple[str, float]]]   # top-3 per slot
    boundary_hints  : list[int]          # slot indices where a line ends
                                         # (exclusive end — matches the j
                                         # convention in viterbi_segment)
    slot_detections : list[Optional[Detection]]   # None = synthetic Unknown
    lines           : list[list[Quadrat]]

    @property
    def n_synthetic(self) -> int:
        return sum(1 for d in self.slot_detections if d is None)


def _group_lines(
    quadrats : list[Quadrat],
    cross    : int,                      # cross axis: 0=x (columns), 1=y (rows)
    gap_frac : float,
) -> list[list[Quadrat]]:
    """1-D cluster quadrats on the cross axis using a RUNNING-MEAN center
    (comparing to the last element drifts on slanted photos — the bug in
    the old order_signs.py)."""
    med = float(np.median([(q.bbox[2] - q.bbox[0]) if cross == 0
                           else (q.bbox[3] - q.bbox[1]) for q in quadrats]))
    qs = sorted(quadrats, key=lambda q: q.centroid[cross])
    lines: list[list[Quadrat]] = [[qs[0]]]
    for q in qs[1:]:
        mean_c = float(np.mean([m.centroid[cross] for m in lines[-1]]))
        if abs(q.centroid[cross] - mean_c) > gap_frac * med:
            lines.append([q])
        else:
            lines[-1].append(q)
    return lines


def assemble_reading_order(
    quadrats         : list[Quadrat],
    layout           : str   = 'columns',     # 'rows' | 'columns'
    direction        : str   = 'rtl',         # 'ltr' | 'rtl'
    line_gap_frac    : float = LINE_GAP_FRAC,
    missing_gap_frac : float = MISSING_GAP_FRAC,
    max_inserts      : int   = MAX_GAP_INSERTS,
    extent           : Optional[tuple] = None,
    single_line      : bool  = False,
) -> ReadingOrder:
    """
    Assemble quadrats into final reading order.

    1. Group quadrats into lines on the cross axis (columns: x; rows: y).
    2. Order lines: columns follow `direction` (rtl = rightmost column
       first); rows always read top-down.
    3. Walk each line along the reading axis (columns: top-down; rows:
       per `direction`), emitting each quadrat's members via
       Quadrat.ordered(direction) — one slot (top-3) per sign.
    4. SYNTHETIC UNKNOWN INSERTION (fragile breakpoint #1): when the
       edge-gap between consecutive quadrats in a line exceeds
       `missing_gap_frac` x median quadrat step, YOLO probably dropped
       sign(s) there — insert round(gap/step) Unknown slots (capped at
       `max_inserts`) so the Unknown-resolver / royal-name matcher can
       fill them. With `extent` (e.g. the inset cartouche bbox), leading
       and trailing gaps are checked too — a missing FIRST sign (the
       Unas e34 case) is only detectable against a known extent.
    5. After each line, append len(slots) to boundary_hints (exclusive
       end index — the corrector's LAYOUT_BONUS convention).

    `single_line=True` skips line grouping and treats every quadrat as
    one line. REQUIRED for cartouche interiors: narrow signs are not
    x-aligned, so line grouping splits the interior into fake columns and
    the per-line extent checks then flood it with spurious Unknowns.

    Returns a ReadingOrder. Slots use YOLO-space class names; the glue
    layer normalizes via cartouche_matcher.normalize_code.
    """
    if not quadrats:
        return ReadingOrder([], [], [], [])

    read_axis = 0 if layout == 'rows' else 1
    cross     = 1 - read_axis
    lo, hi    = (0, 2) if read_axis == 0 else (1, 3)

    med_step = float(np.median(
        [q.bbox[hi] - q.bbox[lo] for q in quadrats]))

    lines = ([list(quadrats)] if single_line
             else _group_lines(quadrats, cross, line_gap_frac))
    # Line order: columns follow direction; rows always top-down
    if layout == 'columns' and direction == 'rtl':
        lines.sort(key=lambda ln: -float(np.mean([q.centroid[0] for q in ln])))
    elif layout == 'columns':
        lines.sort(key=lambda ln: float(np.mean([q.centroid[0] for q in ln])))
    else:
        lines.sort(key=lambda ln: float(np.mean([q.centroid[1] for q in ln])))

    descending = (layout == 'rows' and direction == 'rtl')

    slots           : list[list[tuple[str, float]]] = []
    slot_detections : list[Optional[Detection]]     = []
    boundary_hints  : list[int]                     = []

    def emit_unknowns(gap: float) -> None:
        if gap <= missing_gap_frac * med_step:
            return
        # one missing sign ≈ one med_step of empty space
        n = min(max_inserts, max(1, int(gap // med_step)))
        for _ in range(n):
            slots.append([(UNKNOWN_CLASS, 0.0)])
            slot_detections.append(None)

    for line in lines:
        line.sort(key=lambda q: q.centroid[read_axis], reverse=descending)

        for k, q in enumerate(line):
            if k == 0:
                if extent is not None:
                    lead = (extent[hi] - q.bbox[hi] if descending
                            else q.bbox[lo] - extent[lo])
                    emit_unknowns(lead)
            else:
                prev = line[k - 1]
                gap = (prev.bbox[lo] - q.bbox[hi] if descending
                       else q.bbox[lo] - prev.bbox[hi])
                emit_unknowns(gap)
            for d in q.ordered(direction):
                slots.append(list(d.top3))
                slot_detections.append(d)

        if extent is not None and line:
            last = line[-1]
            trail = (last.bbox[lo] - extent[lo] if descending
                     else extent[hi] - last.bbox[hi])
            emit_unknowns(trail)

        boundary_hints.append(len(slots))

    return ReadingOrder(
        slots           = slots,
        boundary_hints  = boundary_hints,
        slot_detections = slot_detections,
        lines           = lines,
    )


# ---------------------------------------------------------------------------
# Reference infer_fn for production / re-entry (ONNXRuntime + letterbox)
# ---------------------------------------------------------------------------

def letterbox(
    bgr: np.ndarray, imgsz: int = 1024, pad_value: int = 114,
) -> tuple[np.ndarray, float, int, int]:
    """
    Aspect-preserving resize onto a square canvas (Ultralytics-style).
    Returns (canvas, scale, dx, dy) where original = (model - d) / scale.

    NOTE: for THIS model, plain stretch outperforms letterbox on re-entry
    crops (6/6 vs 3-5/6 cartouches covered on test_image_vn5). The training
    data went through ETL/batch_resize.py's in-place 224x224 squash, so the
    model learned aspect-distorted signs — stretch matches that
    distribution. Letterbox is kept for experiments and future models
    trained on aspect-preserved data.
    """
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
    session,                          # onnxruntime.InferenceSession
    class_names : list[str],
    conf_thresh : float = REENTRY_CONF,
    iou_thresh  : float = NMS_IOU,
    imgsz       : int   = 1024,
    mode        : str   = 'letterbox',  # 'letterbox' | 'stretch'
) -> Callable[[np.ndarray], list[Detection]]:
    """
    Build an infer_fn satisfying the cartouche_reentry contract: BGR crop
    in -> list[Detection] with bboxes in crop pixel coordinates out.
    Also usable for the global pass (pass conf_thresh=CONF_THRESHOLD).

    mode='letterbox' matches the V4 training distribution (Ultralytics
    trained at 1024 with letterbox) and reproduces Colab raw-YOLO output
    exactly (A/B on grand_glyphs.jpeg: 0 missing / 0 extra vs 5 extra
    for stretch). Use it for the global pass. mode='stretch' kept for
    legacy callers; the old 224x224-squash rationale is obsolete.
    """
    import cv2
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    def _run(canvas: np.ndarray) -> list[Detection]:
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        raw = session.run([out_name], {inp_name: x})[0]
        return postprocess_onnx(raw, class_names, conf_thresh, iou_thresh)

    def infer_stretch(bgr: np.ndarray) -> list[Detection]:
        h, w = bgr.shape[:2]
        dets = _run(cv2.resize(bgr, (imgsz, imgsz)))
        sx, sy = w / imgsz, h / imgsz
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        return dets

    def infer_letterbox(bgr: np.ndarray) -> list[Detection]:
        canvas, scale, dx, dy = letterbox(bgr, imgsz)
        dets = _run(canvas)
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            d.bbox = ((x1 - dx) / scale, (y1 - dy) / scale,
                      (x2 - dx) / scale, (y2 - dy) / scale)
        return dets

    if mode == 'stretch':
        return infer_stretch
    if mode == 'letterbox':
        return infer_letterbox
    raise ValueError(f"mode must be 'stretch' or 'letterbox', got {mode!r}")

