# Beam search over the top-3 YOLO candidates of every slot, scored with the
# bigram model. Moved verbatim from sphinx_corrector.py.

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from operator import attrgetter

from vision.language_model import transition_log_prob

# Partial paths kept during beam search.
BEAM_WIDTH = 8

# A YOLO top-1 at or above this confidence is never overridden by the beam.
YOLO_TRUST_THRESHOLD = 0.75


@dataclass
class BeamState:
    codes: list[str] = field(default_factory=list)
    confs: list[float] = field(default_factory=list)
    score: float = 0.0

    def extend(self, code: str, conf: float, trans_score: float) -> BeamState:
        return BeamState(
            codes=self.codes + [code],
            confs=self.confs + [conf],
            score=self.score + math.log(conf + 1e-9) + trans_score,
        )


def beam_decode_topk(
    slots: list[list[tuple[str, float]]],
    log_prob: dict,
    unigrams: Counter,
    beam_width: int = BEAM_WIDTH,
    trust_threshold: float = YOLO_TRUST_THRESHOLD,
) -> tuple[list[str], list[bool]]:
    """Best flat code sequence and a per-slot flag for beam overrides.
    Each slot is [(code, conf), ...], best first."""
    if not slots:
        return [], []

    beam: list[BeamState] = [BeamState()]

    for slot_candidates in slots:
        new_beam: list[BeamState] = []

        for state in beam:
            for code, conf in slot_candidates:
                trans = 0.0
                if state.codes:
                    trans = transition_log_prob(
                        state.codes[-1], code, log_prob, unigrams
                    )

                new_beam.append(state.extend(code, conf, trans))

        new_beam.sort(key=attrgetter('score'), reverse=True)
        beam = new_beam[:beam_width]

    best = beam[0] if beam else BeamState()

    # Revert any deviation from a YOLO top-1 that was confident enough.
    overridden = []
    for i, chosen in enumerate(best.codes):
        top1_code, top1_conf = slots[i][0]
        deviated = chosen != top1_code

        if deviated and top1_conf >= trust_threshold:
            best.codes[i] = top1_code
            overridden.append(False)
        else:
            overridden.append(deviated)

    return best.codes, overridden
