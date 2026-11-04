import argparse
import json
import random
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

import anthropic
from anthropic.types import JSONOutputFormatParam, OutputConfigParam
from dotenv import dotenv_values

from evidence_store.books.chunk import MD_DIR, Chunk, chunk_book, read_book
from evidence_store.paths import ENV_FILE, REPORTS_DIR

# Haiku writes one question per sampled chunk; the gold is the book, the pages and a verbatim quote.
# Quotes are checked against the chunk, so the set scores any chunking of the same books.

MODEL = 'claude-haiku-5-5'
MAX_TOKENS = 1500
PRICE_PER_MILLION_INPUT = 0.10
PRICE_PER_MILLION_OUTPUT = 0.50
OUT_FILE = REPORTS_DIR / 'book_golden_questions.jsonl'
QUESTIONS = 50
NEGATIVES = 5
PER_BOOK_MIN = 2
MIN_QUALITY = 0.8                 # skip chunks with less than this share of dictionary words
MIN_TOKENS = 200
SYSTEM = (
    'You write evaluation questions for a search engine over books on ancient Egypt. '
    'You get one passage. Write one question a curious visitor might ask that this passage answers. '
    'The question must stand alone: never mention "the passage", "the text", "the author", a chapter '
    'or a page. Ask about a specific fact, person, god, place, object, ritual or reason, not a yes/no '
    'question. Use different words from the passage where natural, but keep names as they are. '
    'Give a short answer, and copy the evidence exactly from the passage: one contiguous span of '
    '8 to 30 words that supports the answer, character for character. If the passage is too garbled '
    'or has nothing worth asking, set usable to false.'
)
NEGATIVE_SYSTEM = (
    'You write negative-control questions for a search engine that covers only ancient Egypt. '
    'Write questions about other ancient cultures (Maya, Aztec, Inca, China, India, Mesopotamia '
    'before contact with Egypt, Norse, Celtic), each specific and natural, none mentioning Egypt.'
)
OUTPUT_FORMAT: JSONOutputFormatParam = {
    'type': 'json_schema',
    'schema': {
        'type': 'object',
        'properties': {
            'usable': {'type': 'boolean'},
            'question': {'type': 'string'},
            'answer': {'type': 'string'},
            'evidence': {'type': 'string'},
            'kind': {'type': 'string', 'enum': ['fact', 'person', 'deity', 'place', 'ritual', 'reason', 'definition']},
        },
        'required': ['usable', 'question', 'answer', 'evidence', 'kind'],
        'additionalProperties': False,
    },
}
NEGATIVE_FORMAT: JSONOutputFormatParam = {
    'type': 'json_schema',
    'schema': {
        'type': 'object',
        'properties': {'questions': {'type': 'array', 'items': {'type': 'string'}}},
        'required': ['questions'],
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


def ask(system: str, content: str, fmt: JSONOutputFormatParam) -> tuple[dict | None, int, int]:
    """One Haiku call with a JSON schema; (reply or None on refusal or cut-off, input tokens, output tokens)."""
    config: OutputConfigParam = {'effort': 'low', 'format': fmt}
    message = client().messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config=config,
        system=system,
        messages=[{'role': 'user', 'content': content}],
    )
    usage = message.usage.input_tokens, message.usage.output_tokens
    if message.stop_reason != 'end_turn':
        return None, *usage
    text = next(block.text for block in message.content if block.type == 'text')
    return json.loads(text), *usage


def squash(text: str) -> str:
    """Whitespace and quote style folded, for checking that evidence is verbatim."""
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', text).strip().lower()


def quota(chunks: dict[str, list[Chunk]], total: int) -> dict[str, int]:
    """Questions per book: at least PER_BOOK_MIN each, the rest by number of chunks."""
    books = [b for b, cs in chunks.items() if cs]
    share = {b: PER_BOOK_MIN for b in books}
    left = total - sum(share.values())
    size = sum(len(chunks[b]) for b in books)
    exact = {b: left * len(chunks[b]) / size for b in books}
    for b in books:
        share[b] += int(exact[b])
    for b in sorted(books, key=lambda b: exact[b] - int(exact[b]), reverse=True)[:total - sum(share.values())]:
        share[b] += 1
    return share


def candidates(chunks: list[Chunk], rng: random.Random) -> list[Chunk]:
    """Readable chunks of one book in random order."""
    good = [c for c in chunks if c.quality >= MIN_QUALITY and c.tokens >= MIN_TOKENS]
    rng.shuffle(good)
    return good


def craft(chunks: dict[str, list[Chunk]], total: int, seed: int) -> tuple[list[dict], int, int]:
    """Positive questions, each checked: its evidence must appear verbatim in its chunk."""
    rng = random.Random(seed)
    rows: list[dict] = []
    tokens_in = tokens_out = 0
    for doc_id, want in quota(chunks, total).items():
        made = 0
        for c in candidates(chunks[doc_id], rng):
            if made == want:
                break
            reply, i, o = ask(SYSTEM, f'Passage from "{c.header}":\n\n{c.text}', OUTPUT_FORMAT)
            tokens_in += i
            tokens_out += o
            if not reply or not reply['usable'] or squash(reply['evidence']) not in squash(c.text):
                continue
            made += 1
            rows.append({
                'id': f'q{len(rows) + 1:03d}',
                'negative': False,
                'question': reply['question'].strip(),
                'answer': reply['answer'].strip(),
                'evidence': reply['evidence'].strip(),
                'kind': reply['kind'],
                'doc_id': doc_id,
                'section': c.section,
                'page_start': c.page_start,
                'page_end': c.page_end,
            })
        print(f'{doc_id:<48} {made}/{want}', flush=True)
    return rows, tokens_in, tokens_out


def negatives(count: int, start: int) -> tuple[list[dict], int, int]:
    """Questions no book here can answer: retrieval should return nothing convincing."""
    if not count:
        return [], 0, 0
    reply, i, o = ask(NEGATIVE_SYSTEM, f'Write {count} questions.', NEGATIVE_FORMAT)
    questions = (reply or {}).get('questions', [])[:count]
    rows = [
        {
            'id': f'n{start + k:03d}',
            'negative': True,
            'question': q.strip(),
            'answer': None,
            'evidence': None,
            'kind': 'negative',
            'doc_id': None,
            'section': None,
            'page_start': None,
            'page_end': None,
        }
        for k, q in enumerate(questions)
    ]
    return rows, i, o


def main() -> None:
    parser = argparse.ArgumentParser(description='Have Claude Haiku write the golden question set for book retrieval.')
    parser.add_argument('--questions', type=int, default=QUESTIONS, help=f'positive questions (default {QUESTIONS})')
    parser.add_argument('--negatives', type=int, default=NEGATIVES, help=f'off-corpus controls (default {NEGATIVES})')
    parser.add_argument('--md', type=Path, default=MD_DIR, help=f'default {MD_DIR}')
    parser.add_argument('--out', type=Path, default=OUT_FILE, help=f'default {OUT_FILE}')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--force', action='store_true', help='overwrite an existing set')
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f'{args.out} exists; a golden set should not change silently. Use --force to replace it.')
    chunks = {p.stem: chunk_book(read_book(p)) for p in sorted(args.md.glob('*.md'))}
    rows, tokens_in, tokens_out = craft(chunks, args.questions, args.seed)
    extra, i, o = negatives(args.negatives, 1)
    rows += extra
    tokens_in += i
    tokens_out += o
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    by_kind = defaultdict(int)
    for row in rows:
        by_kind[row['kind']] += 1
    cost = tokens_in / 1e6 * PRICE_PER_MILLION_INPUT + tokens_out / 1e6 * PRICE_PER_MILLION_OUTPUT
    positives = sum(not r['negative'] for r in rows)
    print(f'\nwrote {args.out}: {positives} questions + {len(extra)} negatives  {dict(by_kind)}')
    print(f'{tokens_in:,} input + {tokens_out:,} output tokens, about ${cost:.4f}')


if __name__ == '__main__':
    main()
