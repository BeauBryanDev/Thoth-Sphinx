from __future__ import  annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import threading
 
RIDDLES_PATH = Path(__file__).resolve().parent / "riddles.json"
MAX_ATTEMPTS = 3
HINT_AFTER_ATTEMPT = MAX_ATTEMPTS - 1
TRIAL_SIZE = 3
PASS_THRESHOLD = 2
TURN_MESSAGE = "wait for the seeker's next message before calling the trial again"
ARTICLES = {"a", "an", "the"}
FIELD_TYPES = {
    "riddle_id": int,
    "riddle_question": str,
    "riddle_answer": list,
    "hints": list,
    "sources": str,
    "difficulty_level": int,
    "subject": str,
}
 
 
def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse spaces, drop a leading article."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = re.sub(r"[^\w\s]", "", text).split()
    if len(words) > 1 and words[0] in ARTICLES:
        words = words[1:]

    return " ".join(words)
 
 
@dataclass(frozen=True)
class Riddle:
    riddle_id: int
    riddle_question: str
    riddle_answer: tuple[str, ...]
    hints: tuple[str, ...]
    sources: str
    difficulty_level: int
    subject: str
 
    @classmethod
    def from_dict(cls, data: dict) -> Riddle:
        return cls(
            riddle_id=data["riddle_id"],
            riddle_question=data["riddle_question"],
            riddle_answer=tuple(data["riddle_answer"]),
            hints=tuple(data["hints"]),
            sources=data["sources"],
            difficulty_level=data["difficulty_level"],
            subject=data["subject"],
        )
 
    @cached_property
    def normalized_answers(self) -> frozenset[str]:
        return frozenset(normalize(a) for a in self.riddle_answer)
 
    def is_correct(self, guess: str) -> bool:
        cleaned = normalize(guess)
        return bool(cleaned) and cleaned in self.normalized_answers
 
 
def validate_item(item: object, position: int) -> list[str]:

    if not isinstance(item, dict):
        return [f"item {position}: expected a JSON object"]

    label = f"item {position} (riddle_id={item.get('riddle_id')})"
    errors = []
    missing = sorted(FIELD_TYPES.keys() - item.keys())
    unknown = sorted(item.keys() - FIELD_TYPES.keys())

    if missing:
        errors.append(f"{label}: missing keys {missing}")
    if unknown:
        errors.append(f"{label}: unknown keys {unknown}")

    for key, expected in FIELD_TYPES.items():
        if key in item and not isinstance(item[key], expected):
            errors.append(f"{label}: '{key}' must be of type {expected.__name__}")

    question = item.get("riddle_question")

    if isinstance(question, str) and not question.strip():
        errors.append(f"{label}: 'riddle_question' is empty")

    for key in ("riddle_answer", "hints"):
        values = item.get(key)
        if isinstance(values, list):
            valid = bool(values) and all(isinstance(v, str) and normalize(v) for v in values)
            if not valid:
                errors.append(f"{label}: '{key}' must be a non-empty list of non-empty strings")

    return errors
 
 
def load_riddles(path: Path = RIDDLES_PATH) -> list[Riddle]:
    try:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)

    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc

    if not isinstance(items, list):
        raise ValueError(f"{path}: top level must be a list of riddles")

    errors: list[str] = []
    seen: set[int] = set()

    for position, item in enumerate(items, start=1):
        item_errors = validate_item(item, position)
        errors.extend(item_errors)

        if item_errors:
            continue

        if item["riddle_id"] in seen:
            errors.append(f"item {position}: duplicate riddle_id {item['riddle_id']}")

        seen.add(item["riddle_id"])

    if errors:
        details = "\n".join(f"- {e}" for e in errors)
        raise ValueError(f"{path}: invalid riddles file\n{details}")

    return [Riddle.from_dict(item) for item in items]
 
 
@dataclass
class RiddleTrial:
    riddle_ids: list[int]
    index: int = 0
    attempt: int = 1
    solved: int = 0
 
    @property
    def current_id(self) -> int:
        return self.riddle_ids[self.index]
 
    @property
    def failed(self) -> int:
        return self.index - self.solved
 
    @property
    def passed(self) -> bool:
        return self.solved >= PASS_THRESHOLD
 
    @property
    def finished(self) -> bool:
        return (
            self.passed
            or self.failed > TRIAL_SIZE - PASS_THRESHOLD
            or self.index >= len(self.riddle_ids)
        )
 
 
class RiddleGame:
    """State of one seeker's riddle trial. Call begin_turn() once per seeker message."""
    def __init__(self, path: Path = RIDDLES_PATH, 
                rng: random.Random | None = None):

        self.riddles = {r.riddle_id: r for r in load_riddles(path)}
        self.rng = rng or random.Random()
        self.asked: set[int] = set()
        self.trial: RiddleTrial | None = None
        self._pass_available = False
        self._acted = False 
        self._lock = threading.Lock()
 
    @property
    def trial_passed(self) -> bool:
        return self._pass_available


    def begin_turn(self) -> None:
        with self._lock:
            self._acted = False

 
    def consume_pass(self) -> bool:
        """Spend the pass earned by a won trial. Returns False if there is none."""
        with self._lock:
            if not self._pass_available:
                return False
            self._pass_available = False
            return True
 

    def start_trial(self) -> dict:

        with self._lock:
            if self._acted:
                raise ValueError(TURN_MESSAGE)
            if self._pass_available:
                raise ValueError("a pass is already available; claim it before starting a new trial")

            ids = list(self.riddles)
            self.rng.shuffle(ids)
            ids.sort(key=lambda riddle_id: riddle_id in self.asked)
            chosen: list[int] = []
            used: frozenset[str] = frozenset()

            for riddle_id in ids:
                answers = self.riddles[riddle_id].normalized_answers
                if answers & used:
                    continue
                chosen.append(riddle_id)
                used = used | answers
                if len(chosen) == TRIAL_SIZE:
                    break

            if len(chosen) < TRIAL_SIZE:
                raise ValueError("not enough riddles with distinct answers for a trial")
            self.asked.update(chosen)
            self.trial = RiddleTrial(chosen)
            self._acted = True

            return self.current_question()
 

    def current_question(self) -> dict:
        trial = self._active_trial()
        question = self.get_question(trial.current_id)

        return {**question, 
        "riddle_number": trial.index + 1, 
        "trial_size": TRIAL_SIZE
               }
 

    def submit(self, guess: str) -> dict:

        with self._lock:
            if self._acted:
                raise ValueError(TURN_MESSAGE)
            if not isinstance(guess, str):
                raise ValueError("guess must be a string")

            trial = self._active_trial()
            self._acted = True
            riddle_number = trial.index + 1
            result = self.check_riddle(trial.current_id,
                                        guess, trial.attempt)

            status = result["status"]

            if status in ("correct", "reveal"):
                if status == "correct":
                    trial.solved += 1
                trial.index += 1
                trial.attempt = 1
            else:
                trial.attempt += 1

            result["riddle_number"] = riddle_number
            result["solved"] = trial.solved
            result["finished"] = trial.finished

            if trial.finished:
                result["passed"] = trial.passed
                if trial.passed:
                    self._pass_available = True

            elif status in ("correct", "reveal"):
                result["next_question"] = self.current_question()

            return result
 
    def get_question(self, riddle_id: int) -> dict:
        riddle = self._get(riddle_id)
        return {
            "riddle_id": riddle.riddle_id,
            "difficulty_level": riddle.difficulty_level,
            "subject": riddle.subject,
            "question": riddle.riddle_question,
        }
 
    def check_riddle(self, riddle_id: int, 
    guess: str, attempt: int
    ) -> dict:
        riddle = self._get(riddle_id)
        if not 1 <= attempt <= MAX_ATTEMPTS:
            raise ValueError(f"attempt must be between 1 and {MAX_ATTEMPTS}")
        result = {
            "riddle_id": riddle_id,
            "attempt": attempt,
            "attempts_left": MAX_ATTEMPTS - attempt,
        }
        if riddle.is_correct(guess):
            return {**result, "status": "correct"}

        if attempt >= MAX_ATTEMPTS:
            return {**result, "status": "reveal", "answer": riddle.riddle_answer[0]}

        if attempt == HINT_AFTER_ATTEMPT:
            return {**result, "status": "hint", "hint": riddle.hints[0]}

        return {**result, "status": "wrong"}


    def _question(self, trial: RiddleTrial) -> dict:
        return {
            "riddle_number": trial.index + 1,
            "trial_size": TRIAL_SIZE,
            "question": self.riddles[trial.current_id].riddle_question,
        }

 
    def _active_trial(self) -> RiddleTrial:

        if self.trial is None or self.trial.finished:
            raise ValueError("no active trial; call start_trial first")
        return self.trial
 

    def _get(self, riddle_id: int) -> Riddle:
        try:
            return self.riddles[riddle_id]
        except KeyError:
            raise ValueError(f"unknown riddle_id: {riddle_id}") from None
 
