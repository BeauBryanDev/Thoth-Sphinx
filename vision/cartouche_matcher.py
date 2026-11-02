#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import Counter

UNKNOWN_TOKEN    = 'Unknown'
EPS              = 1e-6
SKIP_NAME        = -3.0    # name code the detector missed entirely
SKIP_SLOT        = -2.5    # spurious detection (double-box, bracket noise)
GAP_FILL         = -1.2    # synthetic Unknown slot consumed by a name code
SUB_COARSE       = -4.0    # substitution without a confusion matrix
SUB_CAP          = -5.0    # floor for matrix-derived substitution scores
                           # (keeps one bad sub cheaper than del+ins)
ACCEPT_THRESHOLD = -2.0    # min normalized score to accept a name


def normalize_code(yolo_name: str) -> str:
    """YOLO class name -> canonical Gardiner code ('n35'->'N35',
    'aa15'->'Aa15', 'unknown'->'Unknown'). 'cartouche' passes through."""
    if yolo_name in ('cartouche', UNKNOWN_TOKEN):
        
        return yolo_name
    
    if yolo_name.lower() == 'unknown':
        
        return UNKNOWN_TOKEN
    
    if yolo_name[:2].lower() == 'aa':
        
        return 'Aa' + yolo_name[2:]
    
    return yolo_name[0].upper() + yolo_name[1:]


@dataclass
class RoyalMatch:
    name_key      : str               # key in royal_names.json
    translit      : str
    english       : str
    spelling      : list[str]         # the matched spelling (canonical codes)
    score         : float             # normalized alignment score
    aligned_codes : list[str]         # per input slot: assigned name code,
                                      # or original top-1 if slot was skipped
    verified      : bool


def load_royal_names(path: Optional[Path] = None) -> dict:
    """Load royal_names.json -> {key: meta}. Missing file -> {}."""
    p = Path(path) if path else Path(__file__).parent / 'royal_names.json'
    if not p.exists():
        return {}
    with open(p, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('names', {})


def _match_score(
    slot     : list[tuple[str, float]],
    code     : str,
    sub_cost : Optional[dict],
) -> float:
    """Score for aligning one detected slot to one name code."""
    top1 = slot[0][0] if slot else UNKNOWN_TOKEN
    
    for c, conf in slot:
        
        if c == code:
            
            return math.log(conf + EPS)
        
    if top1 == UNKNOWN_TOKEN:
        
        return GAP_FILL
    
    if sub_cost is not None:
        
        c = sub_cost.get((top1.lower(), code.lower()))
        
        if c is not None:
            
            return max(-c, SUB_CAP)
        
    return SUB_COARSE

# Needleman-Wunsch over (slots x spelling).
def _align(
    slots    : list[list[tuple[str, float]]],
    spelling : list[str],
    sub_cost : Optional[dict],
) -> tuple[float, list[str]]:
    """
    Needleman-Wunsch over (slots x spelling). 
    Returns
    (normalized_score, aligned_codes) — aligned_codes has one entry per
    slot: the name code assigned to it, or the slot's own top-1 when the
    slot was skipped as spurious.
    """
    m, n = len(slots), len(spelling)
    
    NEG = float('-inf')
    
    dp   = [[NEG] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    
    dp[0][0] = 0.0
    
    for i in range(1, m + 1):
        
        dp[i][0] = dp[i - 1][0] + SKIP_SLOT
        back[i][0] = 'up'
        
    for j in range(1, n + 1):
        
        dp[0][j] = dp[0][j - 1] + SKIP_NAME
        back[0][j] = 'left'

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = dp[i - 1][j - 1] + _match_score(
                slots[i - 1], spelling[j - 1], sub_cost)
            up   = dp[i - 1][j] + SKIP_SLOT       # spurious slot
            left = dp[i][j - 1] + SKIP_NAME       # missed name code
            best = max(diag, up, left)
            dp[i][j] = best
            back[i][j] = ('diag' if best == diag else
                          'up'   if best == up   else 'left')

    # Backtrack: assign a code (or keep top-1) per slot
    aligned = [''] * m
    i, j = m, n
    
    while i > 0 or j > 0:
        move = back[i][j]
        if move == 'diag':
            aligned[i - 1] = spelling[j - 1]
            i, j = i - 1, j - 1
        elif move == 'up':
            aligned[i - 1] = slots[i - 1][0][0]   # spurious: keep YOLO code
            i -= 1
        else:
            j -= 1                                 # missed sign: no slot
    return dp[m][n] / max(n, 1), aligned


def match_cartouche(
            slots       : list[list[tuple[str, float]]],
            royal_names : dict,
            sub_cost    : Optional[dict] = None,
            threshold   : float = ACCEPT_THRESHOLD,
        ) -> Optional[RoyalMatch]:
    """
    Match a cartouche's interior slots (reading order, top-3 per slot,
    canonical Gardiner codes) against every royal-name spelling. Returns
    the best RoyalMatch, or None if nothing c
    clears `threshold` — in that case the caller keeps the raw YOLO codes.
    """
    slots = [s for s in slots if s]
    
    if not slots:
        return None

    best: Optional[RoyalMatch] = None
    
    for key, meta in royal_names.items():
        
        prior = 0.1 * math.log(meta.get('freq', 1) + 1)
        
        for spelling in meta.get('spellings', []):

            if not spelling:
                continue

            # Cartouche interior reading order depends on which way the
            # wall/figures face, which is independent of (and sometimes
            # opposite to) the `direction` passed for the outer text —
            # royal_names.json spellings are recorded in one canonical
            # order, so a mirrored cartouche must not be penalized as a
            # bad match. Try both orientations, keep the better one.
            score_fwd, aligned_fwd = _align(slots, spelling, sub_cost)
            score_rev, aligned_rev = _align(
                list(reversed(slots)), spelling, sub_cost)

            if score_rev > score_fwd:
                score, aligned = score_rev, list(reversed(aligned_rev))
            else:
                score, aligned = score_fwd, aligned_fwd

            score += prior
            
            if best is None or score > best.score:
                
                best = RoyalMatch(
                    name_key      = key,
                    translit      = meta.get('translit', key),
                    english       = meta.get('english', ''),
                    spelling      = list(spelling),
                    score         = score,
                    aligned_codes = aligned,
                    verified      = bool(meta.get('verified', False)),
                )
                
    if best is not None and best.score < threshold:
        return None
    
    return best


def apply_panel_consensus(
    results        : list[Optional[RoyalMatch]],
    slots_per_cart : list[list[list[tuple[str, float]]]],
    royal_names    : dict,
    sub_cost       : Optional[dict] = None,
    min_votes      : int   = 2,
    margin         : float = 1.0,
    floor          : float = -3.0,
) -> list[tuple[Optional[RoyalMatch], bool]]:
    """
    Panel-level aggregation: a wall/panel usually names ONE king, so
    confidently-resolved cartouches inform the refused ones.

    If >= `min_votes` cartouches resolved to the same name, each refused
    cartouche is re-scored against THAT name only; it adopts the consensus
    iff (a) the consensus score clears `floor` (an absolute guard — pixels
    that actively contradict the name stay refused; matters especially
    when the lexicon is small and the margin test is vacuous) and (b) the
    consensus name is within `margin` of the cartouche's own best name.

    Returns [(match_or_None, inferred)] aligned with the inputs;
    inferred=True marks consensus-adopted results (surface this in the
    API — it is a contextual inference, not a pixel-level read).
    """
    
    votes = Counter(r.name_key for r in results if r is not None)
    if not votes:
        return [(r, False) for r in results]
    top_name, n_votes = votes.most_common(1)[0]
    if n_votes < min_votes:
        return [(r, False) for r in results]

    consensus_lex = {top_name: royal_names[top_name]}
    out: list[tuple[Optional[RoyalMatch], bool]] = []
    for r, slots in zip(results, slots_per_cart):
        if r is not None or not slots:
            out.append((r, False))
            continue
        best_any  = match_cartouche(slots, royal_names, sub_cost,
                                    threshold=float('-inf'))
        best_cons = match_cartouche(slots, consensus_lex, sub_cost,
                                    threshold=float('-inf'))
        if (best_cons is not None and best_any is not None
                and best_cons.score >= floor
                and best_cons.score >= best_any.score - margin):
            out.append((best_cons, True))
        else:
            out.append((None, False))
    return out

