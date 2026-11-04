import argparse
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from itertools import accumulate
from pathlib import Path

import tiktoken

from evidence_store.books.headings import is_word

# Splits the Markdown books into search chunks: never across a section, whole sentences, 224-640 tokens.
# Dry run: nothing goes to the database (evidence_store.books.ingest loads them).

MD_DIR = Path('assets/md')
MIN_TOKENS = 224
MAX_TOKENS = 640
OVERLAP_TOKENS = 64
MIN_KEEP = 16                     # smaller chunks are title-page stubs ('PART I', an author name): dropped
ENCODING = 'cl100k_base'          # the tokenizer of text-embedding-3
NOISY_SHARE = 0.5                 # below this share of dictionary words a chunk is flagged as noisy OCR
ABBREVIATIONS = {
    'b.c', 'b.c.e', 'c.e', 'a.d', 'dyn', 'cf', 'i.e', 'e.g', 'p', 'pp', 'col', 'cols', 'fig', 'figs',
    'vol', 'vols', 'ch', 'chap', 'mr', 'mrs', 'dr', 'st', 'mt', 'no', 'nos', 'ibid', 'op', 'cit', 'viz',
    'ca', 'c', 'r', 'fl', 'ed', 'eds', 'trans', 'sir', 'prof', 'rev', 'lit', 'var', 'l', 'll', 'etc',
    'vs', 'pl', 'pls', 'n', 'nn', 'sq', 'ff', 'cp', 'esp', 'approx',
}
SENTENCE_BREAK = re.compile(r'(?<=[.!?])(["\')\]]*)\s+(?=["\'(\[]?[A-Z0-9])')
LAST_WORD = re.compile(r'([A-Za-z.]+)[.]["\')\]]*$')
INITIAL = re.compile(r'(^|[\s(])[A-Z]\.["\')\]]*$')    # 'E. A. Wallis Budge': an initial, not a sentence end
PAGE_MARK = re.compile(r'^<!-- (pdf )?p\. (\d+) -->$')
DOC_MARK = re.compile(r'^<!-- doc_id: ([^ |]+)')

encoder = tiktoken.get_encoding(ENCODING)


@dataclass
class Sentence:
    """One sentence, with the page it sits on and whether it opens a paragraph."""
    text   : str
    tokens : int
    page   : str
    opens  : bool


@dataclass
class Section:
    """A '##' section of one book."""
    doc_id    : str
    book      : str
    title     : str
    sentences : list[Sentence] = field(default_factory=list)


@dataclass
class Chunk:
    """A search unit: the text that gets embedded, plus what citations and filters need."""
    doc_id     : str
    section    : str
    ordinal    : int
    header     : str
    text       : str
    tokens     : int
    page_start : str
    page_end   : str
    cut        : str
    overlap    : int
    quality    : float


def count(text: str) -> int:
    """Tokens as text-embedding-3 counts them."""
    return len(encoder.encode(text, disallowed_special=()))


def split_sentences(paragraph: str) -> list[str]:
    """Sentence ends, except after abbreviations ('B.C.E.', 'Dyn.', 'cf.') and initials ('E. A. Budge')."""
    pieces = []
    last = 0
    for m in SENTENCE_BREAK.finditer(paragraph):
        piece = paragraph[last:m.end(1)]
        word = LAST_WORD.search(piece)
        if INITIAL.search(piece) or (word and word.group(1).lower().strip('.') in ABBREVIATIONS):
            continue
        pieces.append(piece.strip())
        last = m.end()
    pieces.append(paragraph[last:].strip())
    return [p for p in pieces if p]


def hard_split(text: str, limit: int = MAX_TOKENS - 8) -> list[str]:
    """A 'sentence' longer than a chunk (OCR run-ons, tables) cut into token windows."""
    ids = encoder.encode(text, disallowed_special=())
    return [encoder.decode(ids[i:i + limit]) for i in range(0, len(ids), limit)]


def read_book(path: Path) -> list[Section]:
    """The sections of one exported Markdown book, each a list of sentences tagged with pages."""
    book, doc_id, page = path.stem, path.stem, '?'
    sections: list[Section] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('# '):
            book = line[2:].strip()
        elif DOC_MARK.match(line):
            doc_id = DOC_MARK.match(line).group(1)
        elif line.startswith('## '):
            sections.append(Section(doc_id=doc_id, book=book, title=line[3:].strip()))
        elif PAGE_MARK.match(line):
            m = PAGE_MARK.match(line)
            page = f'{"pdf p." if m.group(1) else "p."} {m.group(2)}'
        elif sections:
            for i, text in enumerate(split_sentences(line)):
                for j, piece in enumerate(hard_split(text) if count(text) > MAX_TOKENS else [text]):
                    sections[-1].sentences.append(Sentence(
                        text=piece,
                        tokens=count(' ' + piece),
                        page=page,
                        opens=i == 0 and j == 0,
                    ))
    return sections


def quality(text: str) -> float:
    """Share of dictionary words: low for OCR garbage and garbled transliteration."""
    words = re.findall(r"[A-Za-z']{3,}", text)
    return round(sum(is_word(w) for w in words) / len(words), 3) if words else 0.0


def choose_cut(window: list[Sentence], first: int) -> tuple[int, str]:
    """Where to end a full chunk: the last paragraph break, else the last sentence."""
    sizes = list(accumulate(s.tokens for s in window))
    candidates = [k for k in range(first + 1, len(window)) if sizes[k - 1] >= MIN_TOKENS]
    if not candidates:
        return len(window), 'short'
    paragraphs = [k for k in candidates if window[k].opens]
    if paragraphs:
        return paragraphs[-1], 'paragraph'
    return candidates[-1], 'sentence'


def overlap_tail(sentences: list[Sentence]) -> list[Sentence]:
    """The trailing sentences, up to OVERLAP_TOKENS, repeated at the start of the next chunk."""
    tail: list[Sentence] = []
    total = 0
    for s in reversed(sentences[1:]):
        if total + s.tokens > OVERLAP_TOKENS:
            break
        tail.insert(0, s)
        total += s.tokens
    return tail


def pack(section: Section) -> list[tuple[list[Sentence], str, int]]:
    """Sentences of one section packed into chunks: (sentences, cut kind, overlap sentences)."""
    out: list[tuple[list[Sentence], str, int]] = []
    window: list[Sentence] = []
    carried = 0
    for s in section.sentences:
        while window and sum(x.tokens for x in window) + s.tokens > MAX_TOKENS:
            k, kind = choose_cut(window, carried)
            out.append((window[:k], kind, carried))
            tail = overlap_tail(window[:k]) if k < len(window) or kind != 'short' else []
            window = tail + window[k:]
            carried = len(tail)
            if sum(x.tokens for x in window) + s.tokens > MAX_TOKENS and len(window) == carried:
                window, carried = [], 0
        window.append(s)
    if window[carried:]:
        last = window
        size = sum(x.tokens for x in last)
        if out and size < MIN_TOKENS:
            prev, kind, prev_carried = out[-1]
            fresh = last[carried:]
            if sum(x.tokens for x in prev + fresh) <= MAX_TOKENS:
                out[-1] = (prev + fresh, kind, prev_carried)
                return out
        out.append((last, 'section_end', carried))
    return out


def chunk_book(sections: list[Section]) -> list[Chunk]:
    """Every chunk of one book, numbered in reading order."""
    chunks: list[Chunk] = []
    for section in sections:
        for sentences, kind, carried in pack(section):
            text = ' '.join(s.text for s in sentences)
            if count(text) < MIN_KEEP:
                continue
            first, last = sentences[0].page, sentences[-1].page
            pages = first if first == last else f'{first}–{last.split()[-1]}'
            chunks.append(Chunk(
                doc_id=section.doc_id,
                section=section.title,
                ordinal=len(chunks),
                header=f'{section.book} › {section.title} › {pages}',
                text=text,
                tokens=count(text),
                page_start=first,
                page_end=last,
                cut=kind,
                overlap=sum(s.tokens for s in sentences[:carried]),
                quality=quality(text),
            ))
    return chunks


def report(chunks: dict[str, list[Chunk]]) -> None:
    """Per-book and total chunk statistics: sizes, how chunks were cut, short, noisy and duplicate chunks."""
    print(f'\n{"book":<46} {"chunks":>6} {"min":>4} {"med":>4} {"p95":>4} {"max":>4} '
          f'{"short":>5} {"noisy":>5} {"dups":>4}  cuts')
    every = [c for book in chunks.values() for c in book]
    seen = Counter(c.text for c in every)
    for doc_id, book in list(chunks.items()) + [('TOTAL', every)]:
        sizes = sorted(c.tokens for c in book)
        p95 = sizes[int(0.95 * (len(sizes) - 1))]
        short = sum(c.tokens < MIN_TOKENS for c in book)
        noisy = sum(c.quality < NOISY_SHARE for c in book)
        dups = sum(seen[c.text] > 1 for c in book)
        cuts = Counter(c.cut for c in book)
        kinds = ' '.join(f'{k}:{n}' for k, n in cuts.most_common())
        print(f'{doc_id[:46]:<46} {len(book):>6} {sizes[0]:>4} {int(statistics.median(sizes)):>4} '
              f'{p95:>4} {sizes[-1]:>4} {short:>5} {noisy:>5} {dups:>4}  {kinds}')
    total = sum(c.tokens for c in every)
    overlap = sum(c.overlap for c in every)
    print(f'\n{total:,} chunk tokens, {overlap:,} of them overlap ({100 * overlap / max(1, total):.1f}%). '
          f'short = under {MIN_TOKENS} (a section smaller than one chunk); noisy = under '
          f'{NOISY_SHARE:.0%} dictionary words.')


def samples(chunks: dict[str, list[Chunk]], n: int, seed: int) -> None:
    """A few random chunks to read by eye."""
    every = [c for book in chunks.values() for c in book]
    for c in random.Random(seed).sample(every, min(n, len(every))):
        print(f'\n─── {c.header}  [{c.tokens} tokens, cut {c.cut}, overlap {c.overlap}, quality {c.quality}]')
        print(c.text[:900] + (' …' if len(c.text) > 900 else ''))


def main() -> None:
    parser = argparse.ArgumentParser(description='Dry-run chunking of the Markdown books, with statistics.')
    parser.add_argument('doc_ids', nargs='*', help='books to chunk (default: every file in assets/md)')
    parser.add_argument('--md', type=Path, default=MD_DIR, help=f'default {MD_DIR}')
    parser.add_argument('--samples', type=int, default=0, help='print this many random chunks')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, help='write all chunks as JSON lines here')
    args = parser.parse_args()

    paths = sorted(p for p in args.md.glob('*.md') if not args.doc_ids or p.stem in args.doc_ids)
    books = {p.stem: read_book(p) for p in paths}
    chunks = {doc_id: chunk_book(sections) for doc_id, sections in books.items()}
    report(chunks)
    samples(chunks, args.samples, args.seed)
    if args.out:
        with args.out.open('w') as f:
            for book in chunks.values():
                for c in book:
                    f.write(json.dumps(asdict(c), ensure_ascii=False) + '\n')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
