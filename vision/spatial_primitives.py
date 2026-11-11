# Shared building blocks of the spatial layer: tuning constants, the
# Detection record and plain box geometry. The other spatial modules
# import from here; spatial_logic.py re-exports everything.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

Box = tuple[float, float, float, float]
Point = tuple[float, float]
Slot = list[tuple[str, float]]

CARTOUCHE_CLASS = 'cartouche'

# Minimum best-class score to keep an anchor.
CONF_THRESHOLD = 0.15

# Class-agnostic NMS overlap limit.
NMS_IOU = 0.50

# Alternatives kept per detection.
TOP_K = 3

# Minimum score to treat a detection as a cartouche. Lowered from 0.25: the
# V9 detector under-scores cartouches, and a missed one loses the royal name.
CARTOUCHE_CONF = 0.075

# Two cartouche boxes above this IoU collapse to the higher score. Lower than
# the global NMS on purpose: the low CARTOUCHE_CONF admits weak twin boxes
# (IoU 0.3 to 0.5) over one real cartouche that NMS leaves alone.
CARTOUCHE_MERGE_IOU = 0.30

# Grow the cartouche box by this fraction before the containment test.
CARTOUCHE_EXPAND_FRAC = 0.10

# Minimum share of a sign's area inside the raw cartouche box to count as a
# member. Blocks outer signs that the expanded zone would swallow.
MEMBER_MIN_OVERLAP = 0.20

# Legacy re-entry: crop insets as (short axis, long axis) per side. Two passes
# whose results are unioned: shallow keeps signs hugging the bracket ends,
# deep cuts the bracket curve and the tie knot.
REENTRY_INSETS = ((0.06, 0.06), (0.06, 0.10))

# Legacy re-entry: lower score floor, interior signs had no training labels.
REENTRY_CONF = 0.20

# Legacy re-entry: only re-enter below this many members (always=False).
MIN_CARTOUCHE_MEMBERS = 2

# Legacy re-entry: IoU above which an inner detection duplicates a global one.
DEDUPE_IOU = 0.50

# Intersection over the smaller area above which two boxes are one sign.
DUP_OVERLAP = 0.55

# Two overlapping boxes with an IoU inside this range are two real signs (a
# sun over a duck's back), not one sign boxed twice.
DISTINCT_IOU = (0.10, 0.30)

# Both boxes must score above this to stay distinct; a weaker box is noise.
DISTINCT_MIN_CONF = 0.56

# Minimum projection overlap along the reading axis to share a quadrat.
QUADRAT_ALIGN = 0.50

# Maximum perpendicular gap, in median sign sizes.
QUADRAT_GAP_FRAC = 0.60

# Allowed square-root area ratio between two signs of one quadrat.
QUADRAT_SIZE_RATIO = (0.5, 2.0)

# Components with more signs than this are split.
QUADRAT_MAX_SIGNS = 4

# Maximum merged cross extent, in median sign sizes. A wider pair has crossed
# into the next row or column (sandstone_wall g5+l2 spanned 2.9). Do not lower:
# two equal stacked signs span about 2.2 and must still merge.
QUADRAT_MAX_EXTENT = 2.5

# Trim each sign's cross-axis extent by this fraction per side before the
# coverage profile, so boxes touching across a line gap do not hide it.
BAND_SHRINK = 0.10

# Coverage at or under this fraction of the peak is a gap between two lines.
BAND_VALLEY_FRAC = 0.10

# Cross-axis jump, in median sizes, that starts a new line.
LINE_GAP_FRAC = 0.60

# Reading-axis gap, in median steps, above which Unknown slots are inserted.
MISSING_GAP_FRAC = 1.50

# Maximum synthetic Unknown slots per gap.
MAX_GAP_INSERTS = 2

# YOLO-space name; the glue layer maps it to the trie token Unknown.
UNKNOWN_CLASS = 'unknown'


@dataclass
class Detection:
    """One detected sign; bbox is (x1, y1, x2, y2) in image pixels."""

    bbox: Box
    # (class_name, score) pairs, best first.
    top3: list[tuple[str, float]]
    inside_cartouche: bool = False
    # Index of the parent cartouche detection.
    cartouche_id: Optional[int] = None
    # Legacy: came from the re-entry pass.
    from_reentry: bool = False

    @property
    def centroid(self) -> Point:
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
        """Top-1 class name."""
        return self.top3[0][0]

    @property
    def max_score(self) -> float:
        """Top-1 score."""
        return self.top3[0][1]

    def is_cartouche(self, conf: float = CARTOUCHE_CONF) -> bool:
        return self.cls == CARTOUCHE_CLASS and self.max_score > conf


class HasCentroid(Protocol):
    @property
    def centroid(self) -> Point: ...


def centroid_on(item: HasCentroid, axis: int) -> float:
    """Centroid coordinate of a detection or quadrat on axis 0 (x) or 1 (y)."""
    return item.centroid[axis]


def axis_edges(axis: int) -> tuple[int, int]:
    """Bbox indices of the low and high edge on axis 0 (x) or 1 (y)."""
    if axis == 0:
        return (0, 2)

    return (1, 3)


def intersection_area(a: Box, b: Box) -> float:
    """Area of the overlap rectangle of two boxes; 0.0 if they are apart."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    return iw * ih


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    inter = intersection_area(a, b)
    if inter == 0.0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter)


def overlap_ratio(a: Box, b: Box) -> float:
    """Intersection over the smaller area; near 1.0 for a box inside another."""
    inter = intersection_area(a, b)
    if inter == 0.0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / max(min(area_a, area_b), 1e-9)


def axis_overlap(a: Box, b: Box, axis: int) -> float:
    """Projection overlap on axis 0 (x) or 1 (y), over the smaller extent."""
    lo, hi = axis_edges(axis)
    inter = min(a[hi], b[hi]) - max(a[lo], b[lo])
    if inter <= 0:
        return 0.0

    return inter / max(min(a[hi] - a[lo], b[hi] - b[lo]), 1e-9)


def expand_bbox(
    bbox: Box,
    frac: float,
    img_w: Optional[float] = None,
    img_h: Optional[float] = None,
) -> Box:
    """Grow a bbox by frac of its own size on each side; clamp to image."""
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy

    if img_w is not None:
        x1, x2 = max(0.0, x1), min(float(img_w), x2)

    if img_h is not None:
        y1, y2 = max(0.0, y1), min(float(img_h), y2)

    return (x1, y1, x2, y2)


def inset_bbox(
    bbox: Box,
    frac: float,
    frac_long: Optional[float] = None,
) -> Box:
    """Shrink a bbox by frac per side; frac_long applies to the longer axis."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    fl = frac if frac_long is None else frac_long

    # A vertical box has its long axis on y, a horizontal one on x.
    if h >= w:
        dx, dy = w * frac, h * fl
    else:
        dx, dy = w * fl, h * frac

    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def contains_point(bbox: Box, pt: Point) -> bool:
    """True if the point lies inside the box, edges included."""
    x1, y1, x2, y2 = bbox

    return x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2
