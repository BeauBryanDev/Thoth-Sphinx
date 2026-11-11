# Cartouche interiors: read each cartouche's signs as one line, match them
# against the royal names, then apply the panel consensus.

from __future__ import annotations

from typing import Any, Optional

from vision.cartouche_matcher import (
    RoyalMatch,
    apply_panel_consensus,
    match_cartouche,
    normalize_slots,
)
from vision.quadrat_clustering import cluster_quadrats
from vision.reading_order import assemble_reading_order
from vision.spatial_primitives import Detection, Slot


def cartouche_indices(
    detections: list[Detection],
    cartouches: list[Detection],
) -> list[int]:
    """Where each cartouche sits in the detection list, in the given order.
    Interior signs name their cartouche by this index."""
    indices: list[int] = []

    for cartouche in cartouches:
        for i, detection in enumerate(detections):
            if detection is cartouche:
                indices.append(i)

    return indices


def _interior_slots(
    members: list[Detection],
    cartouche: Detection,
    layout: str,
    direction: str,
) -> list[Slot]:
    """The interior signs in reading order, canonical codes."""
    if not members:
        return []

    # single_line is mandatory here: line grouping splits an interior and
    # floods it with synthetic Unknowns.
    quadrats = cluster_quadrats(members, layout=layout)
    reading_order = assemble_reading_order(
        quadrats,
        layout=layout,
        direction=direction,
        single_line=True,
        extent=cartouche.bbox,
    )

    return normalize_slots(reading_order.slots)


def _entry(
    cartouche: Detection,
    n_members: int,
    slots: list[Slot],
    match: Optional[RoyalMatch],
    inferred: bool,
) -> dict[str, Any]:
    """One cartouche of the pipeline result; a refused match leaves Nones."""
    entry: dict[str, Any] = {
        'bbox': cartouche.bbox,
        'n_members': n_members,
        'slots': slots,
        'inferred': inferred,
        'translit': None,
        'english': None,
        'spelling': None,
        'score': None,
        'aligned_codes': None,
        'verified': None,
    }

    if match is not None:
        entry['translit'] = match.translit
        entry['english'] = match.english
        entry['spelling'] = match.spelling
        entry['score'] = match.score
        entry['aligned_codes'] = match.aligned_codes
        entry['verified'] = match.verified

    return entry


def read_cartouches(
    detections: list[Detection],
    cart_idxs: list[int],
    layout: str,
    direction: str,
    royal_names: dict[str, dict],
    sub_cost: Optional[dict],
) -> list[dict[str, Any]]:
    """One result entry per cartouche, in the order of cart_idxs."""
    matches: list[Optional[RoyalMatch]] = []
    slots_per_cart: list[list[Slot]] = []
    member_counts: list[int] = []

    for cart_idx in cart_idxs:
        members: list[Detection] = []
        for detection in detections:
            if not detection.inside_cartouche:
                continue

            if detection.cartouche_id == cart_idx:
                members.append(detection)

        slots = _interior_slots(
            members, detections[cart_idx], layout, direction
        )

        match = None
        if members:
            match = match_cartouche(slots, royal_names, sub_cost=sub_cost)

        matches.append(match)
        slots_per_cart.append(slots)
        member_counts.append(len(members))

    final = apply_panel_consensus(
        matches, slots_per_cart, royal_names, sub_cost=sub_cost
    )

    entries: list[dict[str, Any]] = []
    for k, (match, inferred) in enumerate(final):
        entries.append(_entry(
            detections[cart_idxs[k]],
            member_counts[k],
            slots_per_cart[k],
            match,
            inferred,
        ))

    return entries
