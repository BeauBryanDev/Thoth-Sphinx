
from dataclasses import dataclass, field
from typing import Literal

# Reason codes are a fixed vocabulary; tests cover one rule per code.
REASONS: frozenset[str] = frozenset({
    'escalation_disabled',
    'haiku_requested_second_opinion',
    'most_lines_low_confidence',
    'no_corpus_support',
    'unresolved_cartouche',
    'detector_low_confidence',
    'haiku_confident',
})


# Verdicts that judge a line; too_short, unrated and error say nothing either way.
RATED_VERDICTS: frozenset[str] = frozenset({'strong', 'possible', 'none'})


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
    detector_low_share    : float  # codes with confidence < 0.5, Unknown included
    unresolved_cartouches : int
    has_image             : bool
    line_confidences      : tuple[str, ...]   # Haiku's HIGH/MEDIUM/LOW per line
    haiku_requested       : bool
    haiku_reason          : str = ''
    line_verdicts         : tuple[str, ...] = ()   # search_parallels verdict per line

    @property
    def no_corpus_support(self) -> bool:
        """Every line the corpus could rate came back 'none' (too_short and unrated cannot judge)."""
        rated = [v for v in self.line_verdicts if v in RATED_VERDICTS]

        return bool(rated) and all(v == 'none' for v in rated)

    @property
    def low_line_share(self) -> float:
        """Share of Haiku's lines rated LOW."""
        if not self.line_confidences:
            return 1.0
        # Calculate the share of lines rated LOW
        return sum(c == 'LOW' for c in self.line_confidences) / len(self.line_confidences)


@dataclass(frozen=True)
class Decision:
    """The router's verdict and why."""
    action  : Literal['answer', 'escalate']
    reasons : tuple[str, ...] = field(default_factory=tuple)


def detector_low_share(codes: list[str], 
                       confidences: list[float]
                       ) -> float:
    """Share of codes the detector was unsure of (confidence below 0.5, or Unknown)."""
    # a 'cartouche' place marker is not a sign, and cartouche boxes score low by nature
    signs = [(c, s) for c, s in zip(codes, confidences) if c != 'cartouche']
    if not signs:
        return 1.0
    low = sum(c == 'Unknown' or s < 0.5 for c, s in signs)
    return low / len(signs)


def decide(signals: Signals, 
           thresholds: Thresholds = Thresholds(), 
           enabled: bool = True
           ) -> Decision:
    """Answer with Haiku or escalate to Sonnet; rules in order, first match wins."""
    # If escalation is disabled, return an answer with the reason code 'escalation_disabled'
    if not enabled:
        return Decision('answer', ('escalation_disabled',))
    # If Haiku requested a second opinion, return an escalation with the reason code 'haiku_requested_second_opinion'
    if signals.haiku_requested:
        return Decision('escalate', ('haiku_requested_second_opinion',))
    # If the share of lines rated LOW is greater than or equal to the maximum low line share, return an escalation with the reason code 'most_lines_low_confidence'
    if signals.low_line_share >= thresholds.max_low_line_share:
        return Decision('escalate', ('most_lines_low_confidence',))
    # If there is no corpus support, return an escalation with the reason code 'no_corpus_support'
    if signals.no_corpus_support:
        return Decision('escalate', ('no_corpus_support',))
    # Sonnet can only resolve a cartouche the lexicon refused if it can see it.
    # If there are unresolved cartouches and an image is present, return an escalation with the reason code 'unresolved_cartouche'
    if signals.unresolved_cartouches and signals.has_image:
        return Decision('escalate', ('unresolved_cartouche',))
    # If the share of codes the detector was unsure of is greater than or equal to the maximum detector low share, return an escalation with the reason code 'detector_low_confidence'
    if signals.detector_low_share >= thresholds.max_detector_low_share:
        return Decision('escalate', ('detector_low_confidence',))
    # If the detector was confident, return an answer with the reason code 'haiku_confident'
    return Decision('answer', ('haiku_confident',))


