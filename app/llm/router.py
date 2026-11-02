from dataclasses import dataclass, field
from typing import Literal

# Reason codes are a fixed vocabulary (spec §11); tests cover one rule per code.
REASONS: frozenset[str] = frozenset({
    'escalation_disabled',
    'haiku_requested_second_opinion',
    'most_lines_low_confidence',
    'unresolved_cartouche',
    'detector_low_confidence',
    'haiku_confident',
})


@dataclass(frozen=True)
class Thresholds:
    """Provisional, hand-set until Phase 3 calibration replaces them."""
    max_low_line_share     : float = 0.5    # escalate when at least this share of lines is LOW
    max_detector_low_share : float = 0.35   # escalate when at least this share of codes is < 0.5 or Unknown


@dataclass(frozen=True)
class Signals:
    """Measured inputs to the escalation decision for one read."""
    n_codes               : int
    n_lines               : int
    detector_low_share    : float          # codes with confidence < 0.5, Unknown included
    unresolved_cartouches : int
    has_image             : bool
    line_confidences      : tuple[str, ...]   # Haiku's HIGH/MEDIUM/LOW per line
    haiku_requested       : bool
    haiku_reason          : str = ''

    @property
    def low_line_share(self) -> float:
        """Share of Haiku's lines rated LOW."""
        if not self.line_confidences:
            return 1.0
        return sum(c == 'LOW' for c in self.line_confidences) / len(self.line_confidences)


@dataclass(frozen=True)
class Decision:
    """The router's verdict and why."""
    action  : Literal['answer', 'escalate']
    reasons : tuple[str, ...] = field(default_factory=tuple)


def detector_low_share(codes: list[str], confidences: list[float]) -> float:
    """Share of codes the detector was unsure of (confidence below 0.5, or Unknown)."""
    if not codes:
        return 1.0
    low = sum(c == 'Unknown' or s < 0.5 for c, s in zip(codes, confidences))
    return low / len(codes)


def decide(signals: Signals, thresholds: Thresholds = Thresholds(), enabled: bool = True) -> Decision:
    """Answer with Haiku or escalate to Sonnet; rules in order, first match wins."""
    if not enabled:
        return Decision('answer', ('escalation_disabled',))
    if signals.haiku_requested:
        return Decision('escalate', ('haiku_requested_second_opinion',))
    if signals.low_line_share >= thresholds.max_low_line_share:
        return Decision('escalate', ('most_lines_low_confidence',))
    # Sonnet can only resolve a cartouche the lexicon refused if it can see it.
    if signals.unresolved_cartouches and signals.has_image:
        return Decision('escalate', ('unresolved_cartouche',))
    if signals.detector_low_share >= thresholds.max_detector_low_share:
        return Decision('escalate', ('detector_low_confidence',))
    return Decision('answer', ('haiku_confident',))
