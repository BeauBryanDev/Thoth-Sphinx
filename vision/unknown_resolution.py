# Unknown-slot resolution: propose a real Gardiner code for an Unknown sign
# from the corpus records and the YOLO alternatives. Moved verbatim from
# sphinx_corrector.py.

from __future__ import annotations

from collections import Counter

from vision.beam_decoder import YOLO_TRUST_THRESHOLD
from vision.correction_types import ResolvedUnknown
from vision.lexicon_trie import UNKNOWN_TOKEN, SphinxTrie


def resolve_unknowns(
    path: list[tuple[list[str], list, int]],
    trie: SphinxTrie,
    yolo_topk: list[list[tuple[str, float]]],
    slot_offset: int = 0,
    trust_threshold: float = YOLO_TRUST_THRESHOLD,
) -> list[ResolvedUnknown]:
    """One ResolvedUnknown per Unknown slot of the path that could be read.
    Uses gardiner_orig on the records: the code the Unknown replaced."""
    resolved = []
    current_slot = slot_offset

    for seg_codes, records, _ in path:
        for idx, code in enumerate(seg_codes):
            if code != UNKNOWN_TOKEN or not records:
                continue

            global_idx = current_slot + idx
            slot_candidates = []
            if global_idx < len(yolo_topk):
                slot_candidates = yolo_topk[global_idx]

            yolo_alt_codes = {c for c, _ in slot_candidates
                              if c != UNKNOWN_TOKEN}

            # Tally the ground-truth codes this Unknown stood in for.
            orig_counter: Counter = Counter()
            for rec in records:
                if rec.gardiner_orig:
                    orig_codes = rec.gardiner_orig.split()
                    if idx < len(orig_codes):
                        orig_counter[orig_codes[idx]] += rec.freq

            if not orig_counter:
                continue

            overlap = [c for c in orig_counter if c in yolo_alt_codes]

            if overlap:
                # Best case: the corpus proposal is also in YOLO's top-3.
                best_code = max(overlap, key=orig_counter.get)
                best_freq = orig_counter[best_code]
                reason = (f"corpus proposal '{best_code}' confirmed by a "
                          f"YOLO top-3 candidate at this slot")
            else:
                # No overlap: use a decent YOLO alternative, else stay
                # Unknown. A corpus-only guess is not evidence: YOLO knows
                # 148 classes, and Unknown says so.
                decent_yolo_alt = next(
                    (c for c, conf in slot_candidates
                     if c != UNKNOWN_TOKEN and conf >= 0.2),
                    None,
                )
                if decent_yolo_alt is None:
                    continue

                best_code = decent_yolo_alt
                best_freq = 0
                reason = (f"no corpus/YOLO overlap; falling back to "
                          f"YOLO's own alternative '{decent_yolo_alt}'")

            resolved.append(ResolvedUnknown(
                slot=global_idx,
                proposed=best_code,
                reason=reason,
                freq=best_freq,
            ))

        current_slot += len(seg_codes)

    return resolved
