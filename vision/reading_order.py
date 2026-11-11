# Reading order: group quadrats into lines, order the lines and the signs
# inside them, and insert synthetic Unknown slots where signs went missing.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import numpy as np

from vision.quadrat_clustering import Quadrat
from vision.spatial_primitives import (
    CARTOUCHE_CLASS,
    LINE_GAP_FRAC,
    MAX_GAP_INSERTS,
    MISSING_GAP_FRAC,
    UNKNOWN_CLASS,
    Box,
    Detection,
    Slot,
    axis_edges,
    centroid_on,
)


@dataclass
class ReadingOrder:
    """Spatial-layer output, ready for sphinx_corrector.correct()."""

    # Top-3 (class, score) pairs per slot.
    slots: list[Slot]
    # Slot index where each line ends (exclusive), the corrector convention.
    boundary_hints: list[int]
    # Source detection per slot; None marks a synthetic Unknown.
    slot_detections: list[Optional[Detection]]
    lines: list[list[Quadrat]]

    @property
    def n_synthetic(self) -> int:
        return sum(1 for d in self.slot_detections if d is None)


def _group_lines(
    quadrats: list[Quadrat],
    cross: int,
    gap_frac: float,
) -> list[list[Quadrat]]:
    """Cluster quadrats on the cross axis against a running-mean center."""
    lo, hi = axis_edges(cross)
    sizes = []
    for q in quadrats:
        sizes.append(q.bbox[hi] - q.bbox[lo])

    med = float(np.median(sizes))
    qs = sorted(quadrats, key=partial(centroid_on, axis=cross))
    lines: list[list[Quadrat]] = [[qs[0]]]

    # A running mean avoids the drift of comparing with the last element
    # on slanted photos.
    for q in qs[1:]:
        mean_c = float(np.mean([m.centroid[cross] for m in lines[-1]]))
        if abs(q.centroid[cross] - mean_c) > gap_frac * med:
            lines.append([q])
        else:
            lines[-1].append(q)

    return lines


def _line_center(line: list[Quadrat], axis: int) -> float:
    return float(np.mean([q.centroid[axis] for q in line]))


def _sort_lines(
    lines: list[list[Quadrat]],
    layout: str,
    direction: str,
) -> None:
    """Order lines: columns follow direction, rows always read top-down."""
    if layout == 'columns' and direction == 'rtl':
        lines.sort(key=partial(_line_center, axis=0), reverse=True)
    elif layout == 'columns':
        lines.sort(key=partial(_line_center, axis=0))
    else:
        lines.sort(key=partial(_line_center, axis=1))


def _median_step(quadrats: list[Quadrat], lo: int, hi: int) -> float:
    """Median quadrat length on the reading axis, cartouches excluded."""
    sign_quads = []
    for q in quadrats:
        if q.members[0].cls != CARTOUCHE_CLASS:
            sign_quads.append(q)

    # A cartouche is several signs long and would skew the step.
    if not sign_quads:
        sign_quads = quadrats

    return float(np.median([q.bbox[hi] - q.bbox[lo] for q in sign_quads]))


def _gap_before(
    first: Quadrat,
    extent: Box,
    lo: int,
    hi: int,
    descending: bool,
) -> float:
    """Empty space between the known extent start and the first quadrat."""
    if descending:
        return extent[hi] - first.bbox[hi]

    return first.bbox[lo] - extent[lo]


def _gap_between(
    prev: Quadrat,
    nxt: Quadrat,
    lo: int,
    hi: int,
    descending: bool,
) -> float:
    """Empty space between two consecutive quadrats of a line."""
    if descending:
        return prev.bbox[lo] - nxt.bbox[hi]

    return nxt.bbox[lo] - prev.bbox[hi]


def _gap_after(
    last: Quadrat,
    extent: Box,
    lo: int,
    hi: int,
    descending: bool,
) -> float:
    """Empty space between the last quadrat and the known extent end."""
    if descending:
        return last.bbox[lo] - extent[lo]

    return extent[hi] - last.bbox[hi]


def _unknown_slots(
    gap: float,
    med_step: float,
    missing_gap_frac: float,
    max_inserts: int,
) -> list[Slot]:
    """Synthetic Unknown slots for a gap; none if the gap is small."""
    if gap <= missing_gap_frac * med_step:
        return []

    # One missing sign is about one median step of empty space.
    count = min(max_inserts, max(1, int(gap // med_step)))

    return [[(UNKNOWN_CLASS, 0.0)] for _ in range(count)]


def _pad_unknowns(
    order: ReadingOrder,
    gap: float,
    unknowns: Callable[[float], list[Slot]],
) -> None:
    new = unknowns(gap)
    order.slots.extend(new)
    order.slot_detections.extend([None] * len(new))


def _slot_scores(det: Detection) -> Slot:
    """Scores of one slot; a cartouche marker keeps only its top-1."""
    if det.cls == CARTOUCHE_CLASS:
        return list(det.top3[:1])

    return list(det.top3)


def assemble_reading_order(
    quadrats: list[Quadrat],
    layout: str = 'columns',
    direction: str = 'rtl',
    line_gap_frac: float = LINE_GAP_FRAC,
    missing_gap_frac: float = MISSING_GAP_FRAC,
    max_inserts: int = MAX_GAP_INSERTS,
    extent: Optional[Box] = None,
    single_line: bool = False,
) -> ReadingOrder:
    """Order quadrats into slots with line ends and synthetic Unknowns.
    A cartouche among the quadrats takes one marker slot in its line."""
    if not quadrats:
        return ReadingOrder([], [], [], [])

    read_axis = 0 if layout == 'rows' else 1
    lo, hi = axis_edges(read_axis)
    med_step = _median_step(quadrats, lo, hi)

    # single_line is required for cartouche interiors: narrow signs are not
    # aligned, so line grouping would split them into fake columns and the
    # extent checks would flood the interior with spurious Unknowns.
    if single_line:
        lines = [list(quadrats)]
    else:
        lines = _group_lines(quadrats, 1 - read_axis, line_gap_frac)

    _sort_lines(lines, layout, direction)
    descending = layout == 'rows' and direction == 'rtl'
    unknowns = partial(
        _unknown_slots,
        med_step=med_step,
        missing_gap_frac=missing_gap_frac,
        max_inserts=max_inserts,
    )
    order = ReadingOrder([], [], [], lines)

    # A gap above missing_gap_frac median steps means the detector dropped
    # signs there. With a known extent, leading and trailing gaps count too.
    for line in lines:
        line.sort(key=partial(centroid_on, axis=read_axis), reverse=descending)

        for k, q in enumerate(line):
            if k > 0:
                gap = _gap_between(line[k - 1], q, lo, hi, descending)
                _pad_unknowns(order, gap, unknowns)
            elif extent is not None:
                gap = _gap_before(q, extent, lo, hi, descending)
                _pad_unknowns(order, gap, unknowns)

            for d in q.ordered(direction):
                order.slots.append(_slot_scores(d))
                order.slot_detections.append(d)

        if extent is not None and line:
            gap = _gap_after(line[-1], extent, lo, hi, descending)
            _pad_unknowns(order, gap, unknowns)

        order.boundary_hints.append(len(order.slots))

    return order
