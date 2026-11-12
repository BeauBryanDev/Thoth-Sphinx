# Cartouche reading: match a cartouche's interior signs against the royal
# names in royal_names.json, then let the kings a panel agrees on settle
# its refused cartouches. The alignment itself lives in name_alignment.

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from vision.lexicon_trie import UNKNOWN_TOKEN
from vision.name_alignment import NEG_INF, Slot, align_either_way

# Minimum normalized score to accept a name.
ACCEPT_THRESHOLD = -2.0
# Weight of a king's log frequency, a light prior for famous names.
FREQ_PRIOR_WEIGHT = 0.1

# Panel consensus: cartouches that must agree on a king, how far that king
# may trail a refused cartouche's own best name, and the score below which
# the pixels contradict the name whatever the panel says.
CONSENSUS_MIN_VOTES = 2
CONSENSUS_MARGIN = 1.0
CONSENSUS_FLOOR = -3.0

ROYAL_NAMES_JSON = Path(__file__).parent / 'data' / 'royal_names.json'


def normalize_code(yolo_name: str) -> str:
    """YOLO class name to canonical Gardiner code: n35 to N35, aa15 to Aa15.
    The cartouche class passes through."""
    if yolo_name in ('cartouche', UNKNOWN_TOKEN):
        return yolo_name

    if yolo_name.lower() == 'unknown':
        return UNKNOWN_TOKEN

    if yolo_name[:2].lower() == 'aa':
        return 'Aa' + yolo_name[2:]

    return yolo_name[0].upper() + yolo_name[1:]


def normalize_slots(slots: list[Slot]) -> list[Slot]:
    """The slots with canonical Gardiner codes and plain float scores."""
    normalized: list[Slot] = []

    for slot in slots:
        normalized.append([(normalize_code(c), float(s)) for c, s in slot])

    return normalized


@dataclass
class RoyalMatch:
    # Key in royal_names.json.
    name_key: str
    # The name the matched spelling writes, with its kind where known.
    translit: str
    english: str
    # The spelling that matched, canonical codes.
    spelling: list[str]
    # Normalized alignment score plus the frequency prior.
    score: float
    # Per input slot: the name code, or YOLO's top-1 for a skipped slot.
    aligned_codes: list[str]
    verified: bool


def load_royal_names(path: Optional[Path] = None) -> dict[str, dict]:
    """The names of royal_names.json by key; empty if the file is absent."""
    p = Path(path) if path else ROYAL_NAMES_JSON
    if not p.exists():
        return {}

    with open(p, encoding='utf-8') as f:
        data = json.load(f)

    return data.get('names', {})


def _name_label(key: str, meta: dict, index: int) -> str:
    """Which name this spelling writes, e.g. mn-xpr-ra (throne name).
    Kings with one recorded name keep their plain transliteration."""
    names = meta.get('spelling_names')
    if not names or index >= len(names):
        return meta.get('translit', key)

    name = names[index]

    return f"{name['translit']} ({name['kind']} name)"


def _to_match(
    key: str,
    meta: dict,
    index: int,
    spelling: list[str],
    score: float,
    aligned: list[str],
) -> RoyalMatch:
    return RoyalMatch(
        name_key=key,
        translit=_name_label(key, meta, index),
        english=meta.get('english', ''),
        spelling=list(spelling),
        score=score,
        aligned_codes=aligned,
        verified=bool(meta.get('verified', False)),
    )


def match_cartouche(
    slots: list[Slot],
    royal_names: dict[str, dict],
    sub_cost: Optional[dict] = None,
    threshold: float = ACCEPT_THRESHOLD,
) -> Optional[RoyalMatch]:
    """Best royal name for the interior slots, read in either direction.
    None when nothing clears the threshold: the caller keeps YOLO's codes."""
    slots = [slot for slot in slots if slot]
    if not slots:
        return None

    mirrored = list(reversed(slots))
    best: Optional[RoyalMatch] = None

    for key, meta in royal_names.items():
        prior = FREQ_PRIOR_WEIGHT * math.log(meta.get('freq', 1) + 1)

        for index, spelling in enumerate(meta.get('spellings', [])):
            if not spelling:
                continue

            score, aligned = align_either_way(
                slots, mirrored, spelling, sub_cost
            )
            score += prior

            if best is None or score > best.score:
                best = _to_match(
                    key, meta, index, spelling, score, aligned
                )

    if best is not None and best.score < threshold:
        return None

    return best


def _panel_king(
    results: list[Optional[RoyalMatch]],
    min_votes: int,
) -> Optional[str]:
    """The name most cartouches resolved to, if enough of them agree."""
    votes = Counter(r.name_key for r in results if r is not None)
    if not votes:
        return None

    name, n_votes = votes.most_common(1)[0]
    if n_votes < min_votes:
        return None

    return name


def _adopt_consensus(
    slots: list[Slot],
    king: str,
    royal_names: dict[str, dict],
    sub_cost: Optional[dict],
    margin: float,
    floor: float,
) -> Optional[RoyalMatch]:
    """The panel's king for a refused cartouche, if its signs allow it."""
    own_best = match_cartouche(slots, royal_names, sub_cost, NEG_INF)
    as_king = match_cartouche(
        slots, {king: royal_names[king]}, sub_cost, NEG_INF
    )

    if own_best is None or as_king is None:
        return None

    if as_king.score < floor or as_king.score < own_best.score - margin:
        return None

    return as_king


def apply_panel_consensus(
    results: list[Optional[RoyalMatch]],
    slots_per_cart: list[list[Slot]],
    royal_names: dict[str, dict],
    sub_cost: Optional[dict] = None,
    min_votes: int = CONSENSUS_MIN_VOTES,
    margin: float = CONSENSUS_MARGIN,
    floor: float = CONSENSUS_FLOOR,
) -> list[tuple[Optional[RoyalMatch], bool]]:
    """A panel usually names one king: refused cartouches may adopt it.
    Returns (match, inferred) per cartouche; inferred is context, not pixels."""
    king = _panel_king(results, min_votes)
    if king is None:
        return [(r, False) for r in results]

    out: list[tuple[Optional[RoyalMatch], bool]] = []

    for result, slots in zip(results, slots_per_cart):
        if result is not None or not slots:
            out.append((result, False))
            continue

        adopted = _adopt_consensus(
            slots, king, royal_names, sub_cost, margin, floor
        )
        out.append((adopted, adopted is not None))

    return out
