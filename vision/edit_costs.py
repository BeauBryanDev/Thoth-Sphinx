# Cost of editing a sign sequence: the coarse penalty table, the per-pair
# penalty from the YOLO confusion matrix, and the confidence gate.

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

# Penalty by Levenshtein distance. Used without a cost matrix, or when a fuzzy
# hit changed the segment length (no per-pair lookup is possible then).
EDIT_PENALTY = {0: 0.0, 1: -1.0, 2: -2.5}


def compute_edit_penalty(
    orig_seg: list[str],
    eff_seg: list[str],
    edit_dist: int,
    sub_cost: Optional[dict],
) -> float:
    """Penalty (<= 0) for an edited segment, per pair when lengths match."""
    if edit_dist == 0:
        return 0.0

    # Without a matrix, or after an insert or delete, use the coarse table.
    # Otherwise sum log P(truth | pred) over the substituted positions:
    # often-confused pairs cost little, rarely-confused pairs cost a lot.
    if sub_cost is None or len(orig_seg) != len(eff_seg):
        return EDIT_PENALTY.get(edit_dist, -5.0)

    total = 0.0
    for o, e in zip(orig_seg, eff_seg):
        if o == e:
            continue

        c = sub_cost.get((o, e))
        if c is None:
            total += EDIT_PENALTY.get(1, -1.0)
        else:
            total += -c

    return total


def should_correct(
    yolo_conf: float,
    edit_dist: int,
    threshold: float = 0.30,
) -> bool:
    """Naive Bayes gate: apply a trie correction or keep the YOLO sign."""
    if edit_dist == 0:
        return True

    # P(correction is right) is roughly (1 - yolo_conf) / (edit_dist + 1).
    p_correction = (1.0 - yolo_conf) / (edit_dist + 1)

    return p_correction >= threshold


def load_sub_cost_matrix(
    confusion_csv: str | Path,
    classes: Optional[list[str]] = None,
    epsilon: float = 1e-4,
) -> dict:
    """Per-pair cost dict (pred, truth) -> -log(P(truth | pred) + epsilon)."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required: pip install pandas")

    # The CSV is row-normalized: rows are the predicted class, columns the
    # truth, each cell is P(truth | pred). Never-seen pairs cost about 9.2.
    df = pd.read_csv(confusion_csv, index_col=0)

    # Drop the Ultralytics background axis: it is never a valid edit.
    for label in ('background', 'Background'):
        if label in df.index:
            df = df.drop(index=label)

        if label in df.columns:
            df = df.drop(columns=label)

    if classes is not None:
        keep_idx = [c for c in classes if c in df.index]
        keep_col = [c for c in classes if c in df.columns]
        df = df.loc[keep_idx, keep_col]

    cost = {}
    rows, cols = list(df.index), list(df.columns)
    mat = df.values.astype(float)
    for i, pred in enumerate(rows):
        for j, truth in enumerate(cols):
            p = float(mat[i, j])
            cost[(pred, truth)] = -math.log(p + epsilon)

    return cost
