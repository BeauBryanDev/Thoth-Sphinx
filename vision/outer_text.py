# Outer text of a wall: the signs outside the cartouches, in reading order,
# with each cartouche kept as one place-marker slot where it sits in the line.
# The corrector knows signs only, so it reads the line without the markers.

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from vision import sphinx_corrector as SC
from vision.cartouche_matcher import normalize_slots
from vision.lexicon_trie import UNKNOWN_TOKEN, SphinxTrie
from vision.quadrat_clustering import cluster_quadrats
from vision.reading_order import ReadingOrder, assemble_reading_order
from vision.spatial_primitives import CARTOUCHE_CLASS, Detection, Slot

# What load_corrector returns: trie, bigram log-probs, unigrams, sub costs.
Corrector = tuple[SphinxTrie, dict, Counter, Optional[dict]]


@dataclass
class OuterText:
    # Canonical codes per slot, cartouche markers included.
    slots: list[Slot]
    reading_order: ReadingOrder
    correction: SC.CorrectionResult
    # The cartouches in reading order: the nth marker is the nth of these.
    cartouches: list[Detection]


def _is_outer(detection: Detection) -> bool:
    """A cartouche, or a sign outside every cartouche. A weak
    cartouche-class box is neither."""
    if detection.is_cartouche():
        return True

    return not detection.inside_cartouche and detection.cls != CARTOUCHE_CLASS


def _sign_positions(slots: list[Slot]) -> list[int]:
    """Indices of the slots that are signs, not cartouche markers."""
    positions: list[int] = []

    for k, slot in enumerate(slots):
        if slot[0][0] != CARTOUCHE_CLASS:
            positions.append(k)

    return positions


def _correct_signs(
    slots: list[Slot],
    boundary_hints: list[int],
    corrector: Corrector,
) -> SC.CorrectionResult:
    """Run the corrector without the markers, then put them back in place."""
    trie, log_prob, unigrams, sub_cost = corrector
    keep = _sign_positions(slots)

    correction = SC.correct(
        [slots[k] for k in keep],
        trie,
        log_prob,
        unigrams,
        sub_cost_matrix=sub_cost,
        boundary_hints=[bisect_left(keep, b) for b in boundary_hints],
    )

    codes = [CARTOUCHE_CLASS] * len(slots)
    for k, code in zip(keep, correction.flat_corrected_seq):
        codes[k] = code

    correction.flat_corrected_seq = codes

    for resolution in correction.unknowns_resolved:
        resolution.slot = keep[resolution.slot]

    return correction


def _top1_codes(slots: list[Slot]) -> SC.CorrectionResult:
    """Same shape with no trie and no Viterbi: each slot's top-1 code."""
    codes: list[str] = []

    for slot in slots:
        codes.append(slot[0][0] if slot else UNKNOWN_TOKEN)

    return SC.CorrectionResult(
        segmented_words=[],
        unknowns_resolved=[],
        flat_corrected_seq=codes,
        flat_translit='',
        flat_translation='',
        score=0.0,
        had_fallback=False,
    )


def read_outer_text(
    detections: list[Detection],
    layout: str,
    direction: str,
    corrector: Optional[Corrector],
) -> OuterText:
    """Order and correct the outer text. No corrector: YOLO's top-1 codes."""
    outer = [d for d in detections if _is_outer(d)]
    quadrats = cluster_quadrats(outer, layout=layout, split_lines=True)
    reading_order = assemble_reading_order(
        quadrats, layout=layout, direction=direction
    )
    slots = normalize_slots(reading_order.slots)

    if corrector is None:
        correction = _top1_codes(slots)
    else:
        correction = _correct_signs(
            slots, reading_order.boundary_hints, corrector
        )

    cartouches: list[Detection] = []
    for detection in reading_order.slot_detections:
        if detection is not None and detection.is_cartouche():
            cartouches.append(detection)

    return OuterText(
        slots=slots,
        reading_order=reading_order,
        correction=correction,
        cartouches=cartouches,
    )
