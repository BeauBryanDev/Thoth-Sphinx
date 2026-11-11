# Records returned by the corrector. CorrectionResult.to_dict() is the
# JSON-ready form that reaches the API and the LLM prompt.

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SegmentedWord:
    codes: list[str]
    translit: str
    translation: str
    freq: int
    # 0 is an exact match, 1 or 2 is fuzzy corrected.
    edit_dist: int
    # Mean YOLO confidence over the slot range.
    confidence: float
    # dickson, bbaw or fallback.
    source: str


@dataclass
class ResolvedUnknown:
    # Index in the flat YOLO sequence.
    slot: int
    # The Gardiner code we propose.
    proposed: str
    # Human-readable justification.
    reason: str
    # Corpus frequency of the proposal.
    freq: int


@dataclass
class CorrectionResult:
    segmented_words: list[SegmentedWord]
    unknowns_resolved: list[ResolvedUnknown]
    # The next three go into the prompt.
    flat_corrected_seq: list[str]
    flat_translit: str
    flat_translation: str
    # Total score of the segmentation.
    score: float
    # True if graceful degradation was used.
    had_fallback: bool

    def to_dict(self) -> dict:
        return {
            'segmented_words': [asdict(w) for w in self.segmented_words],
            'unknowns_resolved': [asdict(u) for u in self.unknowns_resolved],
            'flat_corrected_seq': self.flat_corrected_seq,
            'flat_translit': self.flat_translit,
            'flat_translation': self.flat_translation,
            'score': self.score,
            'had_fallback': self.had_fallback,
        }
