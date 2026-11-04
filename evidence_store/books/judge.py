import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import anthropic
from anthropic.types import JSONOutputFormatParam
from dotenv import dotenv_values

from evidence_store.paths import ENV_FILE, REPORTS_DIR

# Claude Opus judges whether a retrieved passage answers a golden question, from any book.
# Verdicts are cached by question and passage text, so a new chunking only pays for passages it has not seen.

MODEL = 'claude-opus-5-5'
EFFORT = 'medium'
MAX_TOKENS = 16000
PRICE_PER_MILLION_INPUT = 4.00
PRICE_PER_MILLION_OUTPUT = 20.00
FALLBACK_BETA = 'server-side-fallback-2026-07-01'
CACHE_FILE = REPORTS_DIR / 'book_judgments.jsonl'
VERDICTS = ('yes', 'partial', 'no')
SYSTEM = (
    'You are an expert Egyptologist and historian of the ancient world, with deep knowledge of Egyptian '
    'religion, language, texts, kings and archaeology, and of the history of Egyptology. You grade the '
    'results of a search engine over books on ancient Egypt.\n\n'
    'You get one question, a reference answer taken from one book, and numbered passages from many books. '
    'Judge every passage on its own, only by what the passage itself says:\n'
    '- yes: a reader could answer the question correctly from this passage alone. Another book may give a '
    'different but well-founded answer, or phrase it differently; that still counts as yes.\n'
    '- partial: the passage is on the right subject and gives part of the answer, but a key part is missing.\n'
    '- no: the passage does not answer the question, even if it mentions the same names.\n\n'
    'Use your expertise to recognise variant spellings and names (Thoth/Djehuty, Ra/Re, Amen/Amun, '
    'Usertsen/Senusret), old transliteration conventions and OCR noise. Do not credit a passage with facts '
    'that only you know: the passage must contain them. Give a short reason for each verdict.'
)
OUTPUT_FORMAT: JSONOutputFormatParam = {
    'type': 'json_schema',
    'schema': {
        'type': 'object',
        'properties': {
            'verdicts': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'passage': {'type': 'integer'},
                        'verdict': {'type': 'string', 'enum': list(VERDICTS)},
                        'reason': {'type': 'string'},
                    },
                    'required': ['passage', 'verdict', 'reason'],
                    'additionalProperties': False,
                },
            },
        },
        'required': ['verdicts'],
        'additionalProperties': False,
    },
}


@cache
def client() -> anthropic.Anthropic:
    """Anthropic client with the key from .env."""
    return anthropic.Anthropic(
        api_key=dotenv_values(ENV_FILE)['ANTHROPIC_API_KEY'],
        max_retries=5,
    )


def digest(text: str) -> str:
    """Short stable hash."""
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def cache_key(question: str, body: str) -> str:
    """One verdict per question text and passage text, whatever chunking produced the passage."""
    return f'{MODEL}:{digest(question)}:{digest(body)}'


class Judge:
    """Cached Opus verdicts, with a running cost and a spending guard."""

    def __init__(self, path: Path = CACHE_FILE, max_usd: float = 5.0) -> None:
        self.path = path
        self.max_usd = max_usd
        self.spent = 0.0
        self.calls = 0
        self.verdicts: dict[str, dict[str, str]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.verdicts[row['key']] = row

    def known(self, question: str, body: str) -> dict[str, str] | None:
        """The cached verdict, if any."""
        return self.verdicts.get(cache_key(question, body))

    def judge(self, question: dict[str, Any], passages: list[dict[str, Any]]) -> None:
        """Judge the passages of one question that are not cached yet, in one call."""
        todo = [p for p in passages if self.known(question['question'], p['body']) is None]
        if not todo:
            return
        if self.spent >= self.max_usd:
            raise SystemExit(f'stopped at the ${self.max_usd:.2f} judge guard after ${self.spent:.2f}')
        reference = question.get('answer') or '(none: this question may have no answer in these books)'
        lines = [f'Question: {question["question"]}', f'Reference answer from one book: {reference}', '']
        for n, p in enumerate(todo, start=1):
            lines += [f'Passage {n} ({p["header"]}):', p['body'], '']
        message = client().beta.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks='default',
            output_config={'effort': EFFORT, 'format': OUTPUT_FORMAT},
            system=SYSTEM,
            messages=[{'role': 'user', 'content': '\n'.join(lines)}],
        )
        self.calls += 1
        self.spent += (
            message.usage.input_tokens / 1e6 * PRICE_PER_MILLION_INPUT
            + message.usage.output_tokens / 1e6 * PRICE_PER_MILLION_OUTPUT
        )
        if message.stop_reason != 'end_turn':
            print(f'  {question["id"]}: judge stopped with {message.stop_reason}; passages left unjudged')
            return
        text = next(b.text for b in message.content if b.type == 'text')
        by_number = {v['passage']: v for v in json.loads(text)['verdicts']}
        with self.path.open('a') as f:
            for n, p in enumerate(todo, start=1):
                v = by_number.get(n)
                if v is None:
                    continue
                row = {
                    'key': cache_key(question['question'], p['body']),
                    'question_id': question['id'],
                    'chunk_id': p['id'],
                    'verdict': v['verdict'],
                    'reason': v['reason'],
                }
                self.verdicts[row['key']] = row
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
