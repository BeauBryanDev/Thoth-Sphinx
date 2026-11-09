import argparse
import json
import re
from pathlib import Path

from corpus.books.book_inspector import MANIFEST, page_texts
from corpus.books.headings import clean_text

# Writes one Markdown file per manifest book: its sections as '##', printed-page markers for citations.
# Text comes from pdftotext, like ingest; strip_lines, char_map and junk filtering are applied per page.

ASSETS = Path('assets')
OUT_DIR = ASSETS / 'md'
PAGE_ONLY = re.compile(r'^[\s\-–—.,:;|•■\[\]()]*[0-9ivxlcIVXLC]{1,5}[\s\-–—.,:;|•■\[\]()]*$')
MIN_LETTER_SHARE = 0.5    # a line with fewer letters than this share is OCR debris
EDGE_LINES = 3            # lines at each page edge checked for running headers
HEADER_WORDS = 8          # a running header is at most this many words
SENTENCE_END = re.compile(r'[.!?:;"\')\]]$')
SHORT_LINE_SHARE = 0.8    # raw text: a sentence-ending line shorter than this share of the median line ends a paragraph


def keep_line(line: str, strip: set[str]) -> bool:
    """False for running headers, bare page numbers and lines that are mostly symbols."""
    text = line.strip()
    if not text or text in strip or PAGE_ONLY.match(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    content = sum(ch.isalnum() for ch in text)   # digits count, or citation lines '152–54; … 2005, pp. 205' are lost
    return letters >= 1 and content >= MIN_LETTER_SHARE * len(text.replace(' ', ''))


def header_vocabulary(book: dict) -> set[str]:
    """Words of the book title, section titles and strip_lines: what running headers are made of."""
    texts = [book['title']] + [s['title'] for s in book['sections']] + book.get('strip_lines', [])
    return {w for t in texts for w in re.findall(r"[a-z']+", t.lower())}


def is_header(line: str, vocabulary: set[str]) -> bool:
    """A short edge line made only of title words (plus OCR'd page numbers): 'io TALES OF THE MAGICIANS'."""
    words = re.findall(r"[a-z']+", line.lower())
    return 0 < len(words) <= HEADER_WORDS and all(w in vocabulary or len(w) <= 2 for w in words)


def strip_edges(lines: list[str], vocabulary: set[str]) -> list[str]:
    """Drop running headers and footers from the top and bottom few lines of a page."""
    body = [ln for ln in lines if ln.strip()]
    top = [ln for ln in body[:EDGE_LINES] if is_header(ln, vocabulary)]
    bottom = [ln for ln in body[-EDGE_LINES:] if is_header(ln, vocabulary)]
    drop = {id(ln) for ln in top + bottom}
    return [ln for ln in lines if id(ln) not in drop]


def paragraph_breaks(text: str) -> str:
    """pdftotext -raw has no blank lines: a short line that ends a sentence closes its paragraph."""
    lines = text.split('\n')
    lengths = sorted(len(ln.strip()) for ln in lines if ln.strip())
    if not lengths:
        return text
    short = SHORT_LINE_SHARE * lengths[len(lengths) // 2]
    out: list[str] = []
    for line in lines:
        out.append(line)
        if SENTENCE_END.search(line.strip()) and len(line.strip()) < short:
            out.append('')
    return '\n'.join(out)


def page_markdown(
    raw: str,
    book: dict,
    vocabulary: set[str],
    pdf_order: bool = False,
) -> str:
    """One page as clean paragraphs: headers dropped, wrapped lines joined, hyphenated breaks mended."""
    for bad, good in book.get('char_map', {}).items():
        raw = raw.replace(bad, good)
    strip = {clean_text(s) for s in book.get('strip_lines', [])}
    patterns = [re.compile(p) for p in book.get('strip_patterns', [])]   # lines that vary: '….indd 29', '6/24/24 2:13 PM'
    text = clean_text(raw)
    if patterns:
        text = '\n'.join(ln for ln in text.split('\n') if not any(p.fullmatch(ln.strip()) for p in patterns))
    if pdf_order:
        text = paragraph_breaks(text)
    paragraphs: list[str] = []
    current: list[str] = []
    for line in strip_edges(text.split('\n'), vocabulary):
        if not line.strip():
            if current and SENTENCE_END.search(current[-1]):
                paragraphs.append(' '.join(current))
                current = []
            continue
        if keep_line(line, strip):
            current.append(line.strip())
    if current:
        paragraphs.append(' '.join(current))
    text = '\n\n'.join(paragraphs)
    return re.sub(r'(\w)- (\w)', r'\1\2', text)


def printed(page: int, book: dict, section: dict) -> str:
    """The page number a citation shows: printed if the offset is known, else the PDF page."""
    offset = section.get('page_offset', book.get('page_offset'))
    return f'p. {page - offset}' if offset is not None and page - offset > 0 else f'pdf p. {page}'


def book_markdown(book: dict, assets: Path) -> str:
    """The whole book: a small header, then every section with its pages."""
    path = assets / book['path']
    lines = [
        f'# {book["title"]}',
        '',
        (
            f'<!-- doc_id: {book["doc_id"]} | author: {book.get("author") or "unknown"} | '
            f'year: {book.get("year") or "unknown"} | source: {book["path"]} -->'
        ),
    ]
    vocabulary = header_vocabulary(book)
    for section in book['sections']:
        mode = section.get('layout', book.get('layout'))
        start, end = section['pdf_start'], section['pdf_end']
        lines += ['', f'## {section["title"]}']
        pages = page_texts(path, start, end, layout=mode == 'layout', raw=mode == 'raw')
        for number, raw in enumerate(pages, start=start):
            body = page_markdown(raw, book, vocabulary, pdf_order=mode == 'raw')
            if body:
                lines += ['', f'<!-- {printed(number, book, section)} -->', '', body]
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Export manifest books as Markdown, one file per book.')
    parser.add_argument('doc_ids', nargs='*', help='books to export (default: every book not marked skip)')
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help=f'default {MANIFEST}')
    parser.add_argument('--out', type=Path, default=OUT_DIR, help=f'default {OUT_DIR}')
    parser.add_argument('--ready-only', action='store_true', help='only books with status ready')
    args = parser.parse_args()

    books = json.loads(args.manifest.read_text())['books']
    args.out.mkdir(parents=True, exist_ok=True)
    for book in books:
        if args.doc_ids and book['doc_id'] not in args.doc_ids:
            continue
        if book.get('status') == 'skip' or (args.ready_only and book.get('status') != 'ready'):
            continue
        target = args.out / f'{book["doc_id"]}.md'
        text = book_markdown(book, ASSETS)
        target.write_text(text)
        print(f'{target}  {len(book["sections"])} sections  {len(text) // 1024} KB')


if __name__ == '__main__':
    main()
