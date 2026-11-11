# Needleman-Wunsch alignment of a cartouche's detected slots against one
# royal-name spelling. Scores are log-like: 0 is a perfect sign, lower is
# worse. Tuned on the cartouche fixtures; change only with the tests.

from __future__ import annotations

import math
from typing import Optional

from vision.lexicon_trie import UNKNOWN_TOKEN

# One slot is YOLO's candidates for a sign: [(code, confidence)], best first.
Slot = list[tuple[str, float]]

EPS = 1e-6
# A name code the detector missed entirely.
SKIP_NAME = -3.0
# A spurious detection: a double box or bracket noise.
SKIP_SLOT = -2.5
# An Unknown slot standing in for a name code.
GAP_FILL = -1.2
# A substitution when there is no confusion matrix.
SUB_COARSE = -4.0
# Floor for matrix substitutions: one bad sign stays cheaper than a
# missed code plus a spurious slot.
SUB_CAP = -5.0

NEG_INF = float('-inf')

# Backtrack moves: a slot matched to a code, a slot skipped, a code missed.
MATCH = 'diag'
SPURIOUS = 'up'
MISSED = 'left'


def match_score(slot: Slot, 
                code: str, 
                sub_cost: Optional[dict]
                ) -> float:
    """Score for reading one detected slot as one name code."""
    for candidate, conf in slot:
        if candidate == code:
            return math.log(conf + EPS)

    top1 = slot[0][0] if slot else UNKNOWN_TOKEN
    if top1 == UNKNOWN_TOKEN:
        return GAP_FILL

    if sub_cost is not None:
        cost = sub_cost.get((top1.lower(), code.lower()))
        if cost is not None:
            return max(-cost, SUB_CAP)

    return SUB_COARSE


def _score_table(
    slots: list[Slot],
    spelling: list[str],
    sub_cost: Optional[dict],
) -> tuple[list[list[float]], list[list[str]]]:
    """Fill the alignment table; returns the scores and the moves taken."""
    m = len(slots)
    n = len(spelling)
    dp = [[NEG_INF] * (n + 1) for _ in range(m + 1)]
    back = [[''] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0.0

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + SKIP_SLOT
        back[i][0] = SPURIOUS

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + SKIP_NAME
        back[0][j] = MISSED

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            matched = dp[i - 1][j - 1] + match_score(
                slots[i - 1], spelling[j - 1], sub_cost
            )
            spurious = dp[i - 1][j] + SKIP_SLOT
            missed = dp[i][j - 1] + SKIP_NAME

            # Ties prefer a match, then a spurious slot.
            dp[i][j] = matched
            back[i][j] = MATCH

            if spurious > dp[i][j]:
                dp[i][j] = spurious
                back[i][j] = SPURIOUS

            if missed > dp[i][j]:
                dp[i][j] = missed
                back[i][j] = MISSED

    return dp, back


def _backtrack(
    slots: list[Slot],
    spelling: list[str],
    back: list[list[str]],
) -> list[str]:
    """One code per slot: the name code, or YOLO's own for a spurious slot."""
    aligned = [''] * len(slots)
    i = len(slots)
    j = len(spelling)

    while i > 0 or j > 0:
        move = back[i][j]

        if move == MATCH:
            aligned[i - 1] = spelling[j - 1]
            i -= 1
            j -= 1
        elif move == SPURIOUS:
            aligned[i - 1] = slots[i - 1][0][0]
            i -= 1
        else:
            j -= 1

    return aligned


def align(
    slots: list[Slot],
    spelling: list[str],
    sub_cost: Optional[dict],
) -> tuple[float, list[str]]:
    """Score per name code, and the code assigned to each slot."""
    dp, back = _score_table(slots, spelling, sub_cost)
    score = dp[len(slots)][len(spelling)] / max(len(spelling), 1)

    return score, _backtrack(slots, spelling, back)


def align_either_way(
    slots: list[Slot],
    mirrored: list[Slot],
    spelling: list[str],
    sub_cost: Optional[dict],
) -> tuple[float, list[str]]:
    """Best of both reading orders; the codes come back in slot order.
    A cartouche can face against the outer text, and royal_names.json
    records one canonical order, so a mirrored read is not a bad match."""
    score, aligned = align(slots, spelling, sub_cost)
    mirror_score, mirror_aligned = align(mirrored, spelling, sub_cost)

    if mirror_score > score:
        mirror_aligned.reverse()
        return mirror_score, mirror_aligned

    return score, aligned
