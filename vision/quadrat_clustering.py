# Grouping signs: union-find, duplicate box merging, text line bands and
# quadrat clustering (small blocks of signs that share one reading slot).

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from operator import attrgetter, itemgetter

import numpy as np

from vision.spatial_primitives import (
    BAND_SHRINK,
    BAND_VALLEY_FRAC,
    CARTOUCHE_CLASS,
    DISTINCT_IOU,
    DISTINCT_MIN_CONF,
    DUP_OVERLAP,
    QUADRAT_ALIGN,
    QUADRAT_GAP_FRAC,
    QUADRAT_MAX_EXTENT,
    QUADRAT_MAX_SIGNS,
    QUADRAT_SIZE_RATIO,
    Box,
    Detection,
    Point,
    axis_edges,
    axis_overlap,
    centroid_on,
    iou,
    overlap_ratio,
)


class DSU:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

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


def _are_duplicates(a: Detection, b: Detection, overlap_thresh: float) -> bool:
    """True if two detections box the same physical sign."""
    if a.cls == CARTOUCHE_CLASS or b.cls == CARTOUCHE_CLASS:
        return False

    # Outer text never merges with a cartouche interior, nor across two.
    context_a = (a.inside_cartouche, a.cartouche_id)
    context_b = (b.inside_cartouche, b.cartouche_id)
    if context_a != context_b:
        return False

    if overlap_ratio(a.bbox, b.bbox) <= overlap_thresh:
        return False

    # Two confident boxes that only partly overlap are neighbours.
    pair_iou = iou(a.bbox, b.bbox)
    confident = min(a.max_score, b.max_score) > DISTINCT_MIN_CONF
    if DISTINCT_IOU[0] < pair_iou < DISTINCT_IOU[1] and confident:
        return False

    return True


def _collapse_group(group: list[Detection]) -> Detection:
    """Best-scoring member keeps its box; top-3 lists are unioned."""
    base = max(group, key=attrgetter('max_score'))
    if len(group) == 1:
        return base

    scores: dict[str, float] = {}
    for d in group:
        for c, s in d.top3:
            scores[c] = max(scores.get(c, 0.0), s)

    ranked = sorted(scores.items(), key=itemgetter(1), reverse=True)
    base.top3 = ranked[:3]

    return base


def _remap_cartouche_ids(
    merged: list[Detection],
    original: list[Detection],
) -> None:
    """Point cartouche_id at the new list positions (cartouches survive)."""
    new_idx = {id(obj): k for k, obj in enumerate(merged)}
    for d in merged:
        if d.cartouche_id is not None:
            d.cartouche_id = new_idx[id(original[d.cartouche_id])]


def merge_duplicate_boxes(
    detections: list[Detection],
    overlap_thresh: float = DUP_OVERLAP,
) -> list[Detection]:
    """Collapse detections of one physical sign into one; cartouches stay."""
    n = len(detections)
    dsu = DSU(n)

    # The union-find makes the merge transitive across chains of boxes.
    for i in range(n):
        for j in range(i + 1, n):
            if _are_duplicates(detections[i], detections[j], overlap_thresh):
                dsu.union(i, j)

    merged: list[Detection] = []
    for idxs in dsu.groups().values():
        group = [detections[k] for k in idxs]
        merged.append(_collapse_group(group))

    _remap_cartouche_ids(merged, detections)

    return merged


@dataclass
class Quadrat:
    """One visual block of 1 to 4 signs sharing a slot in the reading order."""

    members: list[Detection]

    @property
    def bbox(self) -> Box:
        return (
            min(d.bbox[0] for d in self.members),
            min(d.bbox[1] for d in self.members),
            max(d.bbox[2] for d in self.members),
            max(d.bbox[3] for d in self.members),
        )

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def ordered(self, direction: str = 'rtl') -> list[Detection]:
        """Reading order inside the block: top bands first, then sideways."""
        if len(self.members) <= 1:
            return list(self.members)

        # A new band starts when the y centroid jumps more than half the
        # median member height below the mean of the current band.
        med_h = float(np.median([d.height for d in self.members]))
        by_y = sorted(self.members, key=partial(centroid_on, axis=1))
        bands: list[list[Detection]] = [[by_y[0]]]

        for d in by_y[1:]:
            band_y = np.mean([m.centroid[1] for m in bands[-1]])
            if d.centroid[1] - band_y > 0.5 * med_h:
                bands.append([d])
            else:
                bands[-1].append(d)

        out: list[Detection] = []
        descending = direction == 'rtl'
        for band in bands:
            band.sort(key=partial(centroid_on, axis=0), reverse=descending)
            out.extend(band)

        return out


def _split_component(
    members: list[Detection],
    stack_axis: int,
    max_signs: int,
) -> list[list[Detection]]:
    """Recursively split an oversized component at its largest gap."""
    if len(members) <= max_signs:
        return [members]

    members = sorted(members, key=partial(centroid_on, axis=stack_axis))
    gaps = []
    for k in range(len(members) - 1):
        nxt = members[k + 1].centroid[stack_axis]
        gaps.append(nxt - members[k].centroid[stack_axis])

    cut = int(np.argmax(gaps)) + 1
    left = _split_component(members[:cut], stack_axis, max_signs)
    right = _split_component(members[cut:], stack_axis, max_signs)

    return left + right


def _coverage(
    signs: list[Detection],
    cross: int,
    shrink: float,
) -> tuple[np.ndarray, int]:
    """Sign coverage per pixel on the cross axis, and the first pixel."""
    lo, hi = axis_edges(cross)
    start = int(min(d.bbox[lo] for d in signs))
    cov = np.zeros(int(max(d.bbox[hi] for d in signs)) - start + 1)

    for d in signs:
        pad = shrink * (d.bbox[hi] - d.bbox[lo])
        a = int(d.bbox[lo] + pad) - start
        b = int(d.bbox[hi] - pad) - start
        cov[a:b + 1] += d.height if cross == 0 else d.width

    return cov, start


def _valley_flips(
    cov: np.ndarray,
    valley_frac: float,
    start: int,
) -> np.ndarray:
    """Pixels where coverage enters or leaves a valley inside the text."""
    low = cov <= valley_frac * cov.max()
    filled = np.flatnonzero(~low)
    first = int(filled[0])
    last = int(filled[-1])

    # Inside the text block every valley is one (enter, exit) pair of flips.
    return np.flatnonzero(np.diff(low[first:last + 1])) + first + start


def _bridged(
    signs: list[Detection],
    a: float,
    b: float,
    cross: int,
    med: float,
) -> bool:
    """True if a full-size sign lies across the valley: one line, not two."""
    lo, hi = axis_edges(cross)
    for d in signs:
        spans = d.bbox[lo] <= a and d.bbox[hi] >= b
        if spans and d.bbox[hi] - d.bbox[lo] >= med:
            return True

    return False


def line_bands(
    detections: list[Detection],
    cross: int,
    shrink: float = BAND_SHRINK,
    valley_frac: float = BAND_VALLEY_FRAC,
) -> list[float]:
    """Cross-axis cuts between text lines: empty valleys of sign coverage."""
    signs = [d for d in detections if d.cls != CARTOUCHE_CLASS]
    if len(signs) < 2:
        return []

    lo, hi = axis_edges(cross)
    cov, start = _coverage(signs, cross, shrink)
    flips = _valley_flips(cov, valley_frac, start)
    med = float(np.median([d.bbox[hi] - d.bbox[lo] for d in signs]))

    cuts = []
    for a, b in zip(flips[::2], flips[1::2]):
        if not _bridged(signs, a, b, cross, med):
            cuts.append((a + b) / 2.0)

    return cuts


def band_of(det: Detection, cuts: list[float], cross: int) -> int:
    """Index of the line band holding the detection's centroid."""
    return int(np.searchsorted(cuts, det.centroid[cross]))


def _median_stack_size(detections: list[Detection], stack_axis: int) -> float:
    """Median sign size on the stack axis; 1.0 if only cartouches exist."""
    sizes = []
    for d in detections:
        if d.cls == CARTOUCHE_CLASS:
            continue

        sizes.append(d.width if stack_axis == 0 else d.height)

    if not sizes:
        return 1.0

    return float(np.median(sizes))


def _quadrat_mates(
    a: Detection,
    b: Detection,
    read_axis: int,
    med_stack: float,
    align_overlap: float,
    gap_frac: float,
    size_ratio: tuple[float, float],
) -> bool:
    """True if two signs are aligned, close and alike enough to share a slot."""
    if axis_overlap(a.bbox, b.bbox, read_axis) < align_overlap:
        return False

    lo, hi = axis_edges(1 - read_axis)
    gap = max(a.bbox[lo], b.bbox[lo]) - min(a.bbox[hi], b.bbox[hi])
    if gap > gap_frac * med_stack:
        return False

    # Cross-line guard: a pair spanning more than QUADRAT_MAX_EXTENT sign
    # sizes has leaked into the next row or column, even with a tiny gap.
    extent = max(a.bbox[hi], b.bbox[hi]) - min(a.bbox[lo], b.bbox[lo])
    if extent > QUADRAT_MAX_EXTENT * med_stack:
        return False

    ratio = (a.area / max(b.area, 1e-9)) ** 0.5

    return size_ratio[0] <= ratio <= size_ratio[1]


def _centroid_yx(q: Quadrat) -> tuple[float, float]:
    return (q.centroid[1], q.centroid[0])


def cluster_quadrats(
    detections: list[Detection],
    layout: str = 'columns',
    align_overlap: float = QUADRAT_ALIGN,
    gap_frac: float = QUADRAT_GAP_FRAC,
    size_ratio: tuple[float, float] = QUADRAT_SIZE_RATIO,
    max_signs: int = QUADRAT_MAX_SIGNS,
    split_lines: bool = False,
) -> list[Quadrat]:
    """Group detections into quadrats with DSU, sorted by centroid (y, x).
    The caller passes the subset: outer text or one cartouche's members."""
    if not detections:
        return []

    # layout is 'rows' or 'columns'. Quadrat-mates stack perpendicular to
    # the reading axis; neighbours along it never merge, so a crowded row
    # cannot chain into one giant component.
    read_axis = 0 if layout == 'rows' else 1
    stack_axis = 1 - read_axis
    med_stack = _median_stack_size(detections, stack_axis)

    # split_lines (outer text only) cuts the wall into line bands so a
    # quadrat never reaches into the next row or column. Cartouche
    # interiors leave it off: side-by-side signs there have a real gap.
    cuts = line_bands(detections, stack_axis) if split_lines else []
    bands = [band_of(d, cuts, stack_axis) for d in detections]

    # Cartouches never merge: each stays alone in its own quadrat.
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

            if bands[i] != bands[j]:
                continue

            mates = _quadrat_mates(
                di,
                dj,
                read_axis,
                med_stack,
                align_overlap,
                gap_frac,
                size_ratio,
            )
            if mates:
                dsu.union(i, j)

    quadrats: list[Quadrat] = []
    for idxs in dsu.groups().values():
        members = [detections[k] for k in idxs]
        for part in _split_component(members, stack_axis, max_signs):
            quadrats.append(Quadrat(members=part))

    quadrats.sort(key=_centroid_yx)

    return quadrats
