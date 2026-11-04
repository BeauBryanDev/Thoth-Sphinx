import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from evidence_store.books.headings import (
    FINDERS,
    Heading,
    clean_text,
    find_headings,
    merge,
    read_layout,
    skippable,
)

# Inspects PDFs before they go into the books manifest; the human writes the manifest.
# Poppler CLI only (pdfinfo, pdffonts, pdfimages, pdftotext, pdftohtml): the same extractor ingest will use.
# Section titles come from headings.py: PDF outline, then the printed Contents, then font size, then label lines.

EGYPTOLOGY_CHARS = set('ꜣꜢꜥꜤḥḤḫḪẖšŠḳḲṯṮḏḎꞽꞼ')   # Unicode transliteration: the good case
PRIVATE_USE = re.compile('[-]')          # custom Egyptology fonts land here: garbled
JUNK = re.compile(r"[^\w\s.,;:!?()\[\]'\"’‘“”\-–—/&%§*+=<>°]")
# A page number alone, or at either end of a running header ("290 THE ANCIENT EGYPTIAN …").
PAGE_NUMBER = re.compile(r'^(\d{1,4})$|^(\d{1,4})\s+[A-Za-z]|[A-Za-z.]\s+(\d{1,4})$')
MIN_OFFSET_SHARE = 0.2    # below this the offset is noise, reported as unknown
OCR_PRODUCERS = (
    'abbyy', 'finereader', 'tesseract', 'ocrmypdf', 'omnipage',
    'readiris', 'paper capture', 'clearscan', 'google',
)
EDGE_LINES = 3            # lines at the top and bottom of a page checked for headers and numbers
EMPTY_PAGE_CHARS = 40     # less text than this: blank or image-only
REPEAT_SHARE = 0.15       # an edge line on this share of pages is a running header or footer
MANIFEST = Path('assets/sphinx_books_manifest.json')
PROTECTED = {'the-egyptian-book-of-the-dead'}   # sections audited by hand; --write never touches them
SAMPLE_CHARS = 70


@dataclass
class PageStats:
    """Counts for one page."""
    number : int
    chars  : int
    junk   : int
    pua    : int
    egypt  : int
    images : int
    lines  : list[str] = field(repr=False)


@dataclass
class BookReport:
    """Everything the inspector learned about one PDF."""
    path        : Path
    info        : dict[str, str]
    fonts       : list[str]
    pages       : list[PageStats]
    verdict     : str
    offset      : int | None
    offset_seen : float
    repeats     : list[tuple[str, int]]
    headings    : list[Heading]
    source      : str
    alternatives: dict[str, int]

    @property
    def n_pages(self) -> int:
        """Pages in the PDF."""
        return len(self.pages)

    @property
    def empty_pages(self) -> list[int]:
        """Pages with almost no text."""
        return [p.number for p in self.pages if p.chars < EMPTY_PAGE_CHARS]

    @property
    def junk_share(self) -> float:
        """Share of odd characters: OCR noise or a broken font."""
        chars = sum(p.chars for p in self.pages)
        return sum(p.junk for p in self.pages) / chars if chars else 0.0


def run(command: list[str]) -> str:
    """Run a Poppler tool and return its output."""
    done = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors='replace',
        check=False,
    )
    if done.returncode != 0 and not done.stdout:
        raise RuntimeError(f'{command[0]} failed: {done.stderr.strip()[:200]}')
    return done.stdout


def pdf_info(path: Path) -> dict[str, str]:
    """pdfinfo as a dict (Title, Author, Producer, Creator, Pages...)."""
    info: dict[str, str] = {}
    for line in run(['pdfinfo', str(path)]).splitlines():
        key, _, value = line.partition(':')
        if value:
            info[key.strip()] = value.strip()
    return info


def pdf_fonts(path: Path) -> list[str]:
    """Font names used in the PDF."""
    rows = run(['pdffonts', str(path)]).splitlines()[2:]
    return sorted({row.split()[0] for row in rows if row.strip()})


def images_per_page(path: Path) -> Counter[int]:
    """How many images each page holds."""
    counts: Counter[int] = Counter()
    for row in run(['pdfimages', '-list', str(path)]).splitlines()[2:]:
        parts = row.split()
        if parts and parts[0].isdigit():
            counts[int(parts[0])] += 1
    return counts


def page_texts(
    path: Path,
    first: int | None = None,
    last: int | None = None,
    layout: bool = False,
) -> list[str]:
    """pdftotext, one string per page."""
    command = ['pdftotext', '-enc', 'UTF-8']
    if layout:
        command.append('-layout')
    if first:
        command += ['-f', str(first)]
    if last:
        command += ['-l', str(last)]
    pages = run(command + [str(path), '-']).split('\f')
    return pages[:-1] if pages and not pages[-1].strip() else pages


def page_stats(number: int, text: str, images: int) -> PageStats:
    """Count characters that tell text, OCR noise and font trouble apart."""
    body = ''.join(text.split())
    return PageStats(
        number=number,
        chars=len(body),
        junk=len(JUNK.findall(body)),
        pua=len(PRIVATE_USE.findall(body)),
        egypt=sum(ch in EGYPTOLOGY_CHARS for ch in body),
        images=images,
        lines=[ln.strip() for ln in text.splitlines() if ln.strip()],
    )


def verdict(info: dict[str, str], fonts: list[str], pages: list[PageStats]) -> str:
    """text, ocr (scanned with a text layer), scan (needs OCR) or mixed."""
    producer = f"{info.get('Producer', '')} {info.get('Creator', '')}".lower()
    ocr_marked = any(p in producer for p in OCR_PRODUCERS) or any('glyphless' in f.lower() for f in fonts)
    imaged = sum(p.images > 0 for p in pages) / max(1, len(pages))
    empty = sum(p.chars < EMPTY_PAGE_CHARS for p in pages) / max(1, len(pages))
    if empty > 0.5:
        return 'scan'
    if ocr_marked or imaged > 0.8:
        return 'ocr'
    if empty > 0.1:
        return 'mixed'
    return 'text'


def page_offset(pages: list[PageStats]) -> tuple[int | None, float]:
    """printed page = pdf page - offset, from the numbers at the page edges; and how often it holds."""
    votes: Counter[int] = Counter()
    for p in pages:
        for line in p.lines[:2] + p.lines[-2:]:
            match = PAGE_NUMBER.search(line)
            if match:
                printed = int(next(g for g in match.groups() if g))
                votes[p.number - printed] += 1
                break
    if not votes:
        return None, 0.0
    offset, seen = votes.most_common(1)[0]
    share = seen / len(pages)
    return (offset, share) if share >= MIN_OFFSET_SHARE else (None, share)


def repeated_edges(pages: list[PageStats]) -> list[tuple[str, int]]:
    """Edge lines that repeat across pages: running headers and footers, i.e. strip_lines candidates."""
    counts: Counter[str] = Counter()
    example: dict[str, str] = {}
    for p in pages:
        seen = set()
        for line in p.lines[:EDGE_LINES] + p.lines[-EDGE_LINES:]:
            key = re.sub(r'\d+', '#', line.lower())
            if key.strip('# ') and key not in seen:
                seen.add(key)
                counts[key] += 1
                example.setdefault(key, line)
    floor = max(3, int(REPEAT_SHARE * len(pages)))
    # A lone short word ('THE', 'OF') is an OCR header split over lines; stripping it would eat body lines too.
    return [(example[k], n) for k, n in counts.most_common() if n >= floor and len(k.strip('# .')) > 3]


def inspect(
    path: Path,
    with_headings: bool = True,
    source: str | None = None,
    max_headings: int | None = None,
) -> BookReport:
    """Run every check on one PDF; headings need a full pdftohtml pass, so the folder overview skips them."""
    info = pdf_info(path)
    fonts = pdf_fonts(path)
    images = images_per_page(path)
    texts = page_texts(path)
    pages = [page_stats(i, t, images.get(i, 0)) for i, t in enumerate(texts, start=1)]
    offset, seen = page_offset(pages)
    repeats = repeated_edges(pages)
    found: list[Heading] = []
    chosen = 'skipped'
    alternatives: dict[str, int] = {}
    if with_headings:
        layout = read_layout(path)
        chosen, found = find_headings(layout, source)
        alternatives = {name: len(merge(finder(layout))) for name, finder in FINDERS.items()}
    return BookReport(
        path=path,
        info=info,
        fonts=fonts,
        pages=pages,
        verdict=verdict(info, fonts, pages),
        offset=offset,
        offset_seen=seen,
        repeats=repeats,
        headings=found[:max_headings],
        source=chosen,
        alternatives=alternatives,
    )


def translit_guess(report: BookReport) -> str:
    """unicode, garbled (private-use font codes) or none."""
    pua = sum(p.pua for p in report.pages)
    egypt = sum(p.egypt for p in report.pages)
    if pua > 20:
        return 'garbled'
    return 'unicode' if egypt > 20 else 'none'


def sections(report: BookReport) -> list[dict]:
    """Page ranges between headings; front and back matter (cover, contents, index...) left out."""
    starts = report.headings + [Heading(page=report.n_pages + 1, title='', source='end')]
    out = []
    for i, h in enumerate(report.headings):
        if skippable(h.title):
            continue
        out.append({
            'title': h.title,
            'pdf_start': h.page,
            'pdf_end': max(h.page, starts[i + 1].page - 1),
        })
    return out


def book_title(report: BookReport) -> str:
    """The PDF's Title, or a readable form of a file-name-like one."""
    title = report.info.get('Title') or report.path.stem
    if ' ' not in title:
        title = re.sub(r'[_-]+', ' ', title).strip().title()
    return title


def review_notes(report: BookReport) -> str:
    """What the human should check: where titles came from and which ones could not be found on their page."""
    notes = f'Sections from {report.source}. '
    unsure = [h for h in report.headings if not h.verified]
    if unsure:
        pages = ', '.join(f'p.{h.page}' for h in unsure[:10])
        notes += f'Not found on their page, check: {pages}. '
    if report.empty_pages:
        notes += f'{len(report.empty_pages)} near-empty pages (plates or blanks?). '
    return notes.strip()


def skeleton(report: BookReport) -> dict:
    """A draft manifest entry; the human corrects titles, ranges and the TODO fields."""
    return {
        'doc_id': re.sub(r'[^a-z0-9]+', '-', report.path.stem.lower()).strip('-'),
        'status': 'draft',
        'title': book_title(report),
        'author': report.info.get('Author', ''),
        'year': None,
        'edition': None,
        'path': report.path.name,
        'pages': report.n_pages,
        'kind': 'ocr' if report.verdict in ('ocr', 'scan') else 'text',
        'topic': 'TODO: magic | ritual | religion | funerary | history | art | archaeology | reference',
        'language': 'en',
        'page_offset': report.offset,
        'translit': translit_guess(report),
        'char_map': {},
        'footnotes': 'keep',
        'ocr': report.verdict in ('ocr', 'scan'),
        'layout': 'reading',
        'strip_lines': [line for line, _ in report.repeats[:10]],
        'notes': review_notes(report),
        'sections': sections(report) or [{'title': 'TODO', 'pdf_start': 1, 'pdf_end': report.n_pages}],
    }


def write_manifest(report: BookReport, manifest: Path) -> None:
    """Put the drafted sections into the manifest: a new entry, or new sections for a draft entry. Backs up first."""
    text = manifest.read_text()
    spans = entry_spans(text)
    books = [json.loads(text[a:b]) for a, b in spans]
    draft = skeleton(report)
    index = next((i for i, b in enumerate(books) if b['path'] == report.path.name), None)
    entry = books[index] if index is not None else None
    if entry and (entry['doc_id'] in PROTECTED or entry.get('status') != 'draft'):
        raise SystemExit(f'{entry["doc_id"]} is {entry.get("status")} or protected; edit it by hand.')
    if entry is None and draft['doc_id'] in PROTECTED:
        raise SystemExit(f'{draft["doc_id"]} is protected; edit it by hand.')
    if entry is None:
        start, end = spans[-1][1], spans[-1][1]
        block = ',\n    ' + _indented(draft)
        print(f'added {draft["doc_id"]}: {len(draft["sections"])} sections')
    else:
        print(f'{entry["doc_id"]}: {len(entry["sections"])} -> {len(draft["sections"])} sections')
        print(f'  {draft["notes"]}')
        entry['sections'] = draft['sections']
        start, end = spans[index]
        block = _indented(entry)
    manifest.with_suffix('.json.bak').write_text(text)
    manifest.write_text(text[:start] + block + text[end:])
    print(f'wrote {manifest} (previous version in {manifest.with_suffix(".json.bak")})')


def entry_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each object in "books", so a write leaves every other entry byte for byte."""
    decoder = json.JSONDecoder()
    pos = text.index('[', text.index('"books"')) + 1
    spans: list[tuple[int, int]] = []
    while True:
        while text[pos] in ' \t\r\n,':
            pos += 1
        if text[pos] == ']':
            return spans
        _, end = decoder.raw_decode(text, pos)
        spans.append((pos, end))
        pos = end


def _indented(entry: dict) -> str:
    """One manifest entry as JSON, indented to sit inside "books"."""
    return json.dumps(entry, ensure_ascii=False, indent=2).replace('\n', '\n    ')


def overview(paths: list[Path]) -> None:
    """One row per PDF."""
    print(f'{"book":<46} {"pages":>5} {"kind":<6} {"empty":>5} {"junk%":>6} {"PUA":>5} {"ꜣꜥḥ":>5} {"offset":>8}')
    for path in paths:
        try:
            r = inspect(path, with_headings=False)
        except RuntimeError as e:
            print(f'{path.name[:46]:<46} ERROR {e}')
            continue
        offset = f'{r.offset:+d}' if r.offset is not None else '?'
        print(
            f'{path.name[:46]:<46} {r.n_pages:>5} {r.verdict:<6} {len(r.empty_pages):>5} '
            f'{100 * r.junk_share:>5.1f}% {sum(p.pua for p in r.pages):>5} '
            f'{sum(p.egypt for p in r.pages):>5} {offset:>8}'
        )
    print('\nkind: text | ocr (scan with a text layer) | scan (needs OCR) | mixed.  '
          'offset: printed page = pdf page - offset.')


def detail(r: BookReport) -> None:
    """The full report for one PDF."""
    print(f'{r.path.name}: {r.n_pages} pages, kind {r.verdict}, translit {translit_guess(r)}')
    for key in ('Title', 'Author', 'Producer', 'Creator', 'CreationDate'):
        if r.info.get(key):
            print(f'  {key}: {r.info[key]}')
    print(f'  fonts ({len(r.fonts)}): {", ".join(r.fonts[:12])}{" …" if len(r.fonts) > 12 else ""}')
    print(f'  odd characters: {100 * r.junk_share:.1f}%  private-use: {sum(p.pua for p in r.pages)}  '
          f'Egyptology letters: {sum(p.egypt for p in r.pages)}')
    if r.offset is not None:
        print(f'  page offset: {r.offset:+d} (printed numbers found on {100 * r.offset_seen:.0f}% of pages)')
    empty = r.empty_pages
    if empty:
        print(f'  near-empty pages ({len(empty)}): {_ranges(empty)}')
    noisy = sorted(r.pages, key=lambda p: p.junk / max(1, p.chars), reverse=True)[:5]
    print('  noisiest pages: ' + ', '.join(
        f'{p.number} ({100 * p.junk / max(1, p.chars):.0f}%)' for p in noisy if p.chars))
    print('\n  running headers / footers (strip_lines candidates):')
    for line, n in r.repeats[:15]:
        print(f'    {n:>4}×  {line[:SAMPLE_CHARS]}')
    others = ', '.join(f'{name} {n}' for name, n in r.alternatives.items())
    print(f'\n  sections from {r.source} (all sources: {others}; pick one with --source):')
    for h in r.headings:
        mark = '?' if not h.verified else ('-' if skippable(h.title) else ' ')
        print(f'   {mark} p.{h.page:<5} {h.title[:SAMPLE_CHARS + 30]}')
    print('\n  ? = not found on its page, check it.  - = front/back matter, left out of the skeleton.')


def print_pages(path: Path, spec: str, layout: bool, clean: bool = False) -> None:
    """The text of a page range, as pdftotext gives it, or as ingest will see it after clean_text."""
    first, _, last = spec.partition('-')
    start, end = int(first), int(last or first)
    for number, text in enumerate(page_texts(path, start, end, layout), start=start):
        text = clean_text(text) if clean else text.rstrip()
        print(f'\n───── page {number} ─────\n{text}')


def _ranges(numbers: list[int]) -> str:
    """[1, 2, 3, 7] -> '1-3, 7'."""
    out, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(f'{start}-{prev}' if start != prev else str(start))
        if n is not None:
            start = prev = n
    return ', '.join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description='Inspect PDFs before writing their books manifest entry.')
    parser.add_argument('target', type=Path, help='a PDF, or a folder of PDFs (subfolders are skipped)')
    parser.add_argument('--pages', help='print the text of these pages, e.g. 40-42')
    parser.add_argument('--layout', action='store_true', help='with --pages: keep the physical layout')
    parser.add_argument('--clean', action='store_true', help='with --pages: show the text after junk removal')
    parser.add_argument('--skeleton', action='store_true', help='print a draft manifest entry as JSON')
    parser.add_argument('--source', choices=list(FINDERS), help='take section titles from this source only')
    parser.add_argument('--max-headings', type=int, default=None, help='cap on draft sections (default: none)')
    parser.add_argument('--write', action='store_true',
                        help='put the drafted sections into the manifest (draft entries only; backs it up first)')
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help=f'default {MANIFEST}')
    args = parser.parse_args()

    if args.target.is_dir():
        overview(sorted(args.target.glob('*.pdf')))
        return
    if args.pages:
        print_pages(args.target, args.pages, args.layout, args.clean)
        return
    report = inspect(
        args.target,
        source=args.source,
        max_headings=args.max_headings,
    )
    if args.write:
        write_manifest(report, args.manifest)
        return
    if args.skeleton:
        print(json.dumps(skeleton(report), ensure_ascii=False, indent=2))
        return
    detail(report)


if __name__ == '__main__':
    main()
