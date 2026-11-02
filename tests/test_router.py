import pytest

from app.llm import router
from app.llm.router import Decision, Signals, decide

BASE = dict(n_codes=40, n_lines=3, detector_low_share=0.1, unresolved_cartouches=0,
            has_image=True, line_confidences=('HIGH', 'MEDIUM', 'MEDIUM'), haiku_requested=False)


@pytest.mark.parametrize(('changes', 'expected'), [
    ({}, Decision('answer', ('haiku_confident',))),
    ({'haiku_requested': True}, Decision('escalate', ('haiku_requested_second_opinion',))),
    ({'line_confidences': ('LOW', 'LOW', 'MEDIUM')}, Decision('escalate', ('most_lines_low_confidence',))),
    ({'line_confidences': ('LOW', 'MEDIUM', 'MEDIUM')}, Decision('answer', ('haiku_confident',))),
    ({'line_confidences': ()}, Decision('escalate', ('most_lines_low_confidence',))),
    ({'unresolved_cartouches': 1}, Decision('escalate', ('unresolved_cartouche',))),
    ({'unresolved_cartouches': 1, 'has_image': False}, Decision('answer', ('haiku_confident',))),
    ({'detector_low_share': 0.35}, Decision('escalate', ('detector_low_confidence',))),
    ({'detector_low_share': 0.34}, Decision('answer', ('haiku_confident',))),
    # First match wins: Haiku's own request outranks every signal after it.
    ({'haiku_requested': True, 'detector_low_share': 0.9}, Decision('escalate', ('haiku_requested_second_opinion',))),
])
def test_rules_in_order(changes: dict, expected: Decision) -> None:
    assert decide(Signals(**{**BASE, **changes})) == expected


def test_disabled_never_escalates() -> None:
    signals = Signals(**{**BASE, 'haiku_requested': True})
    assert decide(signals, enabled=False) == Decision('answer', ('escalation_disabled',))


def test_every_reason_is_in_the_vocabulary() -> None:
    seen = {decide(Signals(**{**BASE, **c}), enabled=e).reasons[0] for c, e in [
        ({}, True), ({'haiku_requested': True}, True), ({'line_confidences': ('LOW',)}, True),
        ({'unresolved_cartouches': 1}, True), ({'detector_low_share': 0.5}, True), ({}, False)]}
    assert seen == router.REASONS


def test_detector_low_share_counts_unknown_and_low() -> None:
    assert router.detector_low_share(['G17', 'Unknown', 'N35', 'X1'], [0.9, 0.9, 0.3, 0.6]) == 0.5
    assert router.detector_low_share([], []) == 1.0
