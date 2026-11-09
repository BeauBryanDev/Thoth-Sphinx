import html
import re
import statistics
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# Section titles from three sources, best first: the PDF outline, the printed Contents page, font-size headings.
# Plain text only (pdftohtml -xml); no OCR, no vision model.

WORDLIST = Path('/usr/share/dict/words')
ALLOWED = re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿꜣꜢꜥꜤḥḤḫḪẖšŠḳḲṯṮḏḎꞽꞼ0-9 .,;:!?'()&/\-–—]")
QUOTES = str.maketrans({
    '’': "'",
    '‘': "'",
    '`': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '­': '',
})
DOT_LEADER = re.compile(r'(\s*[.·•…_■]\s*){2,}')
OCR_DIGIT = str.maketrans({
    'i': '1',
    'l': '1',
    'I': '1',
    'L': '1',
    '|': '1',
    'o': '0',
    'O': '0',
    'g': '9',
    'S': '5',
    'Z': '2',
})
TRAILING_PAGE = re.compile(r'^(.*?\S)[\s.,:;\-–—■]+([0-9][0-9ilIoOgSZ|]{0,3}|[ilIo|][0-9][0-9ilIoOgSZ|]{0,2})$')
TOC_PAGE = re.compile(r'^(.*?\S)[\s.,:;\-–—■]+([0-9ilIoOgSZ|]{1,4})$')   # Contents rows: 'Horus II' is page 11
DITTO = re.compile(r'^["„”\']+\s*')     # 'Chapter I. …' then '„ II. …' in old Contents pages
ROMAN = re.compile(r'^[ivxlcdm]{1,7}\.?$', re.IGNORECASE)
LABEL = re.compile(
    r'^(chapter|chap\.|part|book|section|spell|utterance|lecture|appendix|plate|tale|volume|vol\.)\b',
    re.IGNORECASE,
)
NUMBERED = re.compile(LABEL.pattern + r'\s+([0-9]{1,3}|[IVXLC]{1,6})\b', re.IGNORECASE)   # 'CHAPTER II.'
STANDALONE = {
    'abbreviations',
    'acknowledgments',
    'acknowledgements',
    'addenda',
    'afterword',
    'appendix',
    'bibliography',
    'chronology',
    'conclusion',
    'contents',
    'corrigenda',
    'epilogue',
    'foreword',
    'glossary',
    'index',
    'introduction',
    'notes',
    'preface',
    'prologue',
    'references',
    'remarks',
    'summary',
}
FRONT_MATTER = re.compile(
    r'\b(printed|published|publishers?|copyright|isbn|all rights reserved|library|bought|'
    r'university press|digitized|internet archive|https?|www\.)\b',
    re.IGNORECASE,
)
SMALL_WORDS = {
    'a',
    'an',
    'and',
    'as',
    'at',
    'by',
    'for',
    'from',
    'in',
    'into',
    'of',
    'on',
    'or',
    'the',
    'to',
    'with',
}
OUTLINE_NOISE = re.compile(
    r'^\s*\d+\.\d+\s*pp[\s0-9ivxlc]*?(?=(?-i:[A-Z]))'    # '12.0pp17Introduction', '03.0 pp iv iv Copyright page'
    r'|\s*\(page [0-9ivxlc]+\)\s*$',                 # 'Introduction (page 1)'
    re.IGNORECASE,
)
SKIP = re.compile(
    r'^(cover|title page|half title|copyright( page)?|frontmatter|front matter|dedication|contents|'
    r'table of contents|list of (illustrations|figures|plates|maps)( and \w+)?|figures|maps|index|'
    r'image credits|corrigenda)$',
    re.IGNORECASE,
)
SMALL_CAPS = re.compile(r"(?<![\w'])([BEFGHJKNOPQRSTUWYZ])\s+([A-Z]{2,})\b")   # 'F UNERARY' -> 'FUNERARY'
TOC_HEADER = re.compile(r'^\s*(chap(ter)?\.?\s+)?page\b\.?\s*', re.IGNORECASE)
GENERIC = {'remarks', 'notes', 'summary', 'commentary', 'translation', 'introduction'}
CHAPTER_OCR = re.compile(r'^(chapter|chap\.)\s+(in|hi|ill|iii)\b\.?', re.IGNORECASE)
JUNK_OUTLINE = re.compile(r'^(scan|page|image|img|untitled|bookmark)[\s_-]*\d*$', re.IGNORECASE)
HEADING_RATIO = 1.3       # a heading line is this much taller than the body text
DISPLAY_RATIO = 1.8       # this much taller counts anywhere on the page (titles under a picture)
TOP_SHARE = 0.45          # headings sit in the upper part of the page
EDGE_ROWS = 2             # rows at each page edge checked for running headers
RUNNING_SHARE = 0.08      # an edge line on this share of pages is a running header
VERIFY_SCORE = 0.6        # share of a title's words that must appear near the top of its page
VERIFY_ROWS = 10
OFFSET_RANGE = range(-10, 80)
WINDOW = 3                # pages searched around a Contents entry's expected page
MIN_ENTRIES = 3
PAGES_PER_SECTION = 50    # a source must give at least one section per this many pages
MAX_TITLE = 140


@dataclass
class Row:
    """One visual line of a page: pdftohtml text pieces sharing a baseline."""
    page   : int
    top    : int
    left   : int
    height : int
    bold   : bool
    text   : str


@dataclass
class Heading:
    """A candidate section start."""
    page     : int
    title    : str
    source   : str
    verified : bool = True


@dataclass
class Layout:
    """The text rows of every page, the outline and the page size."""
    pages   : list[list[Row]]
    outline : list[tuple[int, int, str]]
    height  : int
    body    : float


@cache
def dictionary() -> frozenset[str]:
    """Lowercase English words, for repairing OCR splits and judging titles."""
    if not WORDLIST.exists():
        return frozenset()
    words = WORDLIST.read_text(errors='ignore').split()
    return frozenset(w.lower() for w in words if w.isalpha() and (len(w) > 1 or w in 'aAI'))


def is_word(token: str) -> bool:
    """A dictionary word, allowing a possessive or plural ending."""
    words = dictionary()
    t = token.lower().strip(".,;:!?'\"()")
    return t in words or t.removesuffix("'s") in words or (t.endswith('s') and t[:-1] in words)


def read_layout(path: Path) -> Layout:
    """pdftohtml -xml: rows with position and height per page, plus the outline."""
    done = subprocess.run(
        ['pdftohtml', '-xml', '-i', '-hidden', '-stdout', str(path)],
        capture_output=True,
        text=True,
        errors='replace',
        check=False,
    )
    xml = done.stdout
    fonts: dict[str, tuple[int, bool]] = {}
    for m in re.finditer(r'<fontspec id="(\d+)" size="(-?\d+)" family="([^"]*)"', xml):
        fonts[m.group(1)] = (int(m.group(2)), 'bold' in m.group(3).lower())
    pages: list[list[Row]] = []
    height = 0
    chunks = re.split(r'<page number="\d+"[^>]*height="(\d+)"[^>]*>', xml)
    for i in range(1, len(chunks), 2):
        height = max(height, int(chunks[i]))
        pages.append(_rows(len(pages) + 1, chunks[i + 1], fonts))
    sizes = [r.height for page in pages for r in page for _ in range(len(r.text))]
    body = statistics.median(sizes) if sizes else 12.0

    return Layout(
        pages=pages,
        outline=_outline(xml),
        height=height or 1000,
        body=body,
    )


def _rows(number: int, chunk: str, fonts: dict[str, tuple[int, bool]]) -> list[Row]:
    """Text pieces of one page grouped into visual lines, top to bottom."""
    pieces = []
    pattern = r'<text top="(-?\d+)" left="(-?\d+)" width="\d+" height="(\d+)" font="(\d+)">(.*?)</text>'
    for m in re.finditer(pattern, chunk, re.DOTALL):
        top, left, height = int(m.group(1)), int(m.group(2)), int(m.group(3))
        raw = m.group(5)
        text = html.unescape(re.sub(r'<[^>]+>', '', raw)).strip()
        if left < 0 or not text:
            continue
        bold = fonts.get(m.group(4), (0, False))[1] or '<b>' in raw
        pieces.append((top, left, height, bold, text))
    pieces.sort()
    groups: list[list[tuple[int, int, int, bool, str]]] = []
    for piece in pieces:
        last = groups[-1][0] if groups else None
        if last and abs(piece[0] - last[0]) <= max(3, 0.5 * min(piece[2], last[2])):
            groups[-1].append(piece)
        else:
            groups.append([piece])
    rows: list[Row] = []
    for group in groups:
        group.sort(key=lambda p: p[1])
        rows.append(Row(
            page=number,
            top=group[0][0],
            left=group[0][1],
            height=max(p[2] for p in group),
            bold=all(p[3] for p in group),
            text=re.sub(r'\s+', ' ', ' '.join(p[4] for p in group)),
        ))
    return rows


def _outline(xml: str) -> list[tuple[int, int, str]]:
    """Bookmarks as (depth, page, title)."""
    start = xml.find('<outline>')
    if start < 0:
        return []
    items: list[tuple[int, int, str]] = []
    depth = 0
    for m in re.finditer(r'<outline>|</outline>|<item page="(\d+)">(.*?)</item>', xml[start:], re.DOTALL):
        token = m.group(0)
        if token == '<outline>':
            depth += 1
        elif token == '</outline>':
            depth -= 1
        else:
            title = html.unescape(re.sub(r'<[^>]+>', '', m.group(2)))
            items.append((depth, int(m.group(1)), title))
    return items


def clean_text(text: str) -> str:
    """Plain readable text: no glyph codes, private-use characters, control bytes or soft hyphens."""
    text = unicodedata.normalize('NFKC', text).translate(QUOTES)
    text = ''.join(ch for ch in text if ch in '\n\t' or unicodedata.category(ch)[0] not in 'CS')
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def clean_title(text: str) -> str:
    """A heading as plain English: junk stripped, letter spacing and OCR splits repaired, no page number."""
    text = clean_text(text).replace('\n', ' ')
    text = ALLOWED.sub(' ', text)
    text = DOT_LEADER.sub(' ', text)
    text = _strip_page(text)
    text = re.sub(r'\(\s*\)', ' ', text)
    text = re.sub(r'\b([a-z]+)- ([a-z]+)\b', r'\1\2', text)
    text = SMALL_CAPS.sub(r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip(" .,;:-–—/'\"")
    text = re.sub(r'\s+([,;:.!?)])', r'\1', text)
    text = _join_spaced(text)
    text = _unglue(text)
    text = _join_splits(text)
    text = CHAPTER_OCR.sub(lambda m: f'{m.group(1)} III.', text)
    if sum(ch.isupper() for ch in text) > 0.6 * max(1, sum(ch.isalpha() for ch in text)):
        text = title_case(text)
    return text[:MAX_TITLE].strip()


def _strip_page(text: str) -> str:
    """Drop a trailing page number, OCR-mangled or not ('Thoth in magic 124', 'Epithets i36')."""
    m = TRAILING_PAGE.match(text.strip())
    if m and any(ch.isdigit() for ch in m.group(2)):
        return m.group(1)
    return re.sub(r'\s+\d{1,4}$', '', text)


def _join_spaced(text: str) -> str:
    """'C O N T E N T S' -> 'CONTENTS'; 'L I S T O F F I GU R E S' -> 'LIST OF FIGURES'."""
    tokens = text.split()
    short = sum(len(t) <= 2 for t in tokens)
    if len(tokens) < 4 or short < 0.6 * len(tokens):
        return text
    return ' '.join(segment(''.join(tokens)))


def segment(glued: str) -> list[str]:
    """Split run-together letters into the fewest dictionary words; the input itself when it can't."""
    low = glued.lower()
    best: list[list[int] | None] = [[]] + [None] * len(low)
    for end in range(1, len(low) + 1):
        for start in range(max(0, end - 20), end):
            prev = best[start]
            piece = low[start:end]
            if prev is None or not (piece in dictionary() and (len(piece) > 1 or piece in 'ai')):
                continue
            if best[end] is None or len(prev) + 1 < len(best[end]):
                best[end] = prev + [end]
    cuts = best[-1]
    if not cuts:
        return [glued]
    words, start = [], 0
    for end in cuts:
        words.append(glued[start:end])
        start = end
    return words


def _unglue(text: str) -> str:
    """'theMS' -> 'the MS', 'X.TheMythsofEa' -> 'X. The Myths of Ea': words OCR ran together."""
    text = re.sub(r'\b([IVXLC]+\.)(?=[A-Z])', r'\1 ', text)
    out: list[str] = []
    for token in text.split():
        parts = re.split(r'(?<=[a-z])(?=[A-Z])', token)
        if len(parts) > 1 and is_word(parts[0]):
            token = ' '.join(parts)
        pieces = []
        for part in token.split():
            core = part.strip(".,;:()'")
            if len(core) >= 6 and core.isalpha() and not is_word(core):
                split = segment(core)
                glue = any(w.lower() in SMALL_WORDS for w in split[1:])
                if 1 < len(split) <= 3 and glue and all(len(w) > 1 for w in split):
                    part = part.replace(core, ' '.join(split))
            pieces.append(part)
        out.append(' '.join(pieces))
    return ' '.join(out)


def _join_splits(text: str) -> str:
    """'TEXTS TRANS LA TED' -> 'TEXTS TRANSLATED': rejoin pieces that only make a word together."""
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        for span in (3, 2):
            parts = tokens[i:i + span]
            cores = [t.strip("(),.;:") for t in parts]
            if len(parts) < span or not all(c.isalpha() for c in cores):
                continue
            joined = ''.join(cores)
            fragment = any(len(c) <= 2 or not is_word(c) for c in cores)
            if joined.lower() in dictionary() and len(joined) >= 5 and fragment:
                lead = parts[0][:len(parts[0]) - len(parts[0].lstrip('('))]
                tail = parts[-1][len(parts[-1].rstrip('),.;:')):]
                out.append(f'{lead}{joined}{tail}')
                i += span
                break
        else:
            out.append(tokens[i])
            i += 1
    return ' '.join(out)


def title_case(text: str) -> str:
    """ALL CAPS -> Title Case, small words lowercase, roman numerals kept."""
    words = text.lower().split()
    out = []
    for i, word in enumerate(words):
        if ROMAN.match(word) and word.rstrip('.') not in {'i', 'mild', 'mix', 'dim', 'civil', 'did', 'lid', 'mid'}:
            out.append(word.upper())
        elif i and word in SMALL_WORDS:
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
    return ' '.join(out)


def plausible(title: str, display: bool = False) -> bool:
    """A real heading: mostly letters, mostly real words, not a sentence fragment or an imprint line."""
    letters = sum(ch.isalpha() for ch in title)
    if letters < 3 or letters < 0.6 * len(title.replace(' ', '')):
        return False
    if FRONT_MATTER.search(title) or not (title[0].isupper() or title[0].isdigit()):
        return False
    words = [w for w in re.findall(r"[A-Za-z']+", title) if len(w) > 2]
    if not words:
        return bool(LABEL.match(title))
    known = sum(is_word(w) for w in words)
    if len(words) == 1:
        word = words[0].lower()
        return word in STANDALONE or (known == 1 and len(word) > 3) or bool(LABEL.match(title)) or display
    return known >= 0.5 * len(words) or (display and known > 0)


def key(title: str) -> str:
    """Letters only, lowercase: compares titles across OCR and numbering noise."""
    return re.sub(r'[^a-z]', '', title.lower())


def running_keys(layout: Layout) -> set[str]:
    """Edge lines repeated across the book: running headers and footers."""
    counts: Counter[str] = Counter()
    for page in layout.pages:
        seen = {key(clean_title(r.text)) for r in page[:EDGE_ROWS] + page[-EDGE_ROWS:]}
        counts.update(k for k in seen if k)
    floor = max(3, int(RUNNING_SHARE * len(layout.pages)))
    return {k for k, n in counts.items() if n >= floor}


def from_outline(layout: Layout) -> list[Heading]:
    """Bookmark titles, children prefixed with their chapter; empty when the outline is junk."""
    items = layout.outline
    if len(items) < MIN_ENTRIES or sum(bool(JUNK_OUTLINE.match(t.strip())) for _, _, t in items) > len(items) / 3:
        return []
    found: list[Heading] = []
    parents: dict[int, str] = {}
    for depth, page, raw in items:
        title = clean_title(OUTLINE_NOISE.sub('', raw.replace('_', ' ')))
        if len(title) == 1 and title.isalpha():
            title = f'Entries {title}'
        if not title or page < 1 or page > len(layout.pages):
            continue
        parents[depth] = title
        for d in [d for d in parents if d > depth]:
            del parents[d]
        if depth > 1 and parents.get(depth - 1) and len(title) < 60:
            title = f'{parents[depth - 1]} — {title}'
        found.append(Heading(page=page, title=title, source='outline'))
    return found
    


def toc_pages(layout: Layout) -> list[int]:
    """Pages of the printed Contents: a 'Contents' heading, then pages full of 'title ... page' rows."""
    limit = max(12, len(layout.pages) // 6)
    pages: list[int] = []
    for i, page in enumerate(layout.pages[:limit]):
        numbered = sum(bool(TRAILING_PAGE.match(r.text)) for r in page)
        titled = any(key(r.text) in {'contents', 'tableofcontents'} for r in page[:4])
        if (titled and numbered >= 2) or (pages and pages[-1] == i and numbered >= max(4, len(page) // 3)):
            pages.append(i + 1)
    return pages


def toc_entries(layout: Layout) -> list[tuple[str, int]]:
    """(title, printed page) from the Contents rows: ditto marks resolved, wrapped titles joined, groups prefixed."""
    entries: list[tuple[str, int]] = []
    pending: list[str] = []
    label = ''
    group: tuple[str, int] | None = None
    main = ''
    for number in toc_pages(layout):
        for row in layout.pages[number - 1]:
            text = row.text.strip()
            if key(text) in {'contents', 'tableofcontents', 'page', 'chapter', 'pages'}:
                continue
            if DITTO.match(text) and label:
                text = DITTO.sub(f'{label} ', text)
            m = LABEL.match(text)
            if m:
                label = m.group(1)
            text = TOC_HEADER.sub('', text)
            first, inner, rest = _split_row(text)
            if inner is not None:
                entries.append((clean_title(' '.join(pending + [first])), inner))
                pending = []
                text = rest
            body, page = split_page(text)
            if page is None:
                if re.search(r'[—:-]\s*$', text):
                    group = (clean_title(text), row.left)
                    pending = []
                else:
                    pending.append(text)
                continue
            if group and row.left <= group[1] + 5:
                group = None
            title = clean_title(' '.join(pending + [body]))
            pending = []
            if title and group:
                title = f'{group[0]} — {title}'
            elif key(title) in GENERIC and main:
                title = f'{main} — {title}'
            elif title:
                main = title
            if title and (not entries or page >= entries[-1][1]):
                entries.append((title, page))
    return entries


def _split_row(text: str) -> tuple[str, int | None, str]:
    """'IV. Previous work 7 V. The Glosses 9' -> ('IV. Previous work', 7, 'V. The Glosses 9')."""
    m = re.match(r'^(.*?\S)\s+(\d{1,4})\s+([IVXLC]+\.\s.*)$', text)
    if not m:
        return text, None, text
    return m.group(1), int(m.group(2)), m.group(3)


def split_page(text: str) -> tuple[str, int | None]:
    """A Contents row as (title, printed page), OCR letters in the number read as digits ('i36' -> 136)."""
    m = TOC_PAGE.match(text)
    if not m:
        return text, None
    digits = m.group(2).translate(OCR_DIGIT)
    return (m.group(1), int(digits)) if digits.isdigit() and int(digits) > 0 else (text, None)


def match_score(title: str, page: list[Row]) -> float:
    """Share of the title's words found in the top rows of a page."""
    words = {w for w in re.findall(r'[a-z]{3,}', title.lower()) if w not in SMALL_WORDS}
    if not words:
        return 0.0
    top = ' '.join(r.text for r in page[:VERIFY_ROWS]).lower()
    found = set(re.findall(r'[a-z]{3,}', top))
    return len(words & found) / len(words)


def from_toc(layout: Layout) -> list[Heading]:
    """Contents entries placed on their PDF page: one offset voted by all entries, then each checked nearby."""
    entries = toc_entries(layout)
    if len(entries) < MIN_ENTRIES:
        return []
    n = len(layout.pages)
    contents = set(toc_pages(layout))
    votes: Counter[int] = Counter()
    for title, printed in entries:
        for offset in OFFSET_RANGE:
            pdf = printed + offset
            if 1 <= pdf <= n and pdf not in contents and match_score(title, layout.pages[pdf - 1]) >= VERIFY_SCORE:
                votes[offset] += 1
    if not votes:
        return []
    offset = votes.most_common(1)[0][0]
    found: list[Heading] = []
    for title, printed in entries:
        guess = printed + offset
        window = [p for p in range(guess - WINDOW, guess + WINDOW + 1) if 1 <= p <= n and p not in contents]
        scored = [(match_score(title, layout.pages[p - 1]), -abs(p - guess), p) for p in window]
        score, _, page = max(scored) if scored else (0.0, 0, guess)
        if 1 <= guess <= n:
            found.append(Heading(
                page=page if score >= VERIFY_SCORE else guess,
                title=title,
                source='contents',
                verified=score >= VERIFY_SCORE,
            ))
    found.sort(key=lambda h: h.page)
    return found


def from_fonts(layout: Layout) -> list[Heading]:
    """Lines clearly taller than the body near the top of a page, wrapped lines joined; running headers dropped."""
    running = running_keys(layout)
    found: list[Heading] = []
    for page in layout.pages:
        block: list[Row] = []
        for row in page:
            tall = row.height >= HEADING_RATIO * layout.body
            near_top = row.top <= TOP_SHARE * layout.height or row.height >= DISPLAY_RATIO * layout.body
            if tall and near_top and key(clean_title(row.text)) not in running:
                if block and row.top - block[-1].top > 2.5 * row.height:
                    break
                block.append(row)
            elif block:
                break
        title = clean_title(' '.join(r.text for r in block))
        display = bool(block) and max(r.height for r in block) >= DISPLAY_RATIO * layout.body
        if block and plausible(title, display):
            found.append(Heading(page=page[0].page, title=title, source='font'))
    return found


def from_lines(layout: Layout) -> list[Heading]:
    """Last resort: 'CHAPTER II.' plus the title line under it; all-caps lines only when a book has no labels."""
    running = running_keys(layout)
    labelled: list[Heading] = []
    capped: list[Heading] = []
    for page in layout.pages:
        caps: Heading | None = None
        for i, row in enumerate(page[:5]):
            title = _bare(clean_title(row.text))
            if not title or key(title) in running or not plausible(title):
                continue
            if NUMBERED.match(title):
                title = clean_title(' '.join([title + '.'] + _title_below(page, i)))
                labelled.append(Heading(page=row.page, title=title, source='lines'))
                caps = None
                break
            if key(title) in STANDALONE:
                labelled.append(Heading(page=row.page, title=title, source='lines'))
                caps = None
                break
            if caps is None and row.text.isupper() and 4 <= len(row.text) <= 60:
                caps = Heading(page=row.page, title=title, source='lines')
        if caps:
            capped.append(caps)
    chapters = [h for h in labelled if NUMBERED.match(h.title)]
    return labelled if len(chapters) >= MIN_ENTRIES else sorted(labelled + capped, key=lambda h: h.page)


def _title_below(page: list[Row], i: int, most: int = 3) -> list[str]:
    """The all-caps lines right under a 'CHAPTER II.' label: the chapter title, wrapped or not."""
    lines: list[str] = []
    last = page[i]
    for row in page[i + 1:i + 1 + most]:
        if not row.text.isupper() or row.top - last.top > 4 * last.height:
            break
        lines.append(row.text)
        last = row
    return lines


def _bare(title: str) -> str:
    """A running-header title without the page number or roman folio stuck to it ('Preface. IX' -> 'Preface')."""
    if LABEL.match(title):
        return title
    words = title.split()
    while words and (words[0].isdigit() or ROMAN.match(words[0]) and words[0].isupper()):
        words = words[1:]
    while words and (words[-1].isdigit() or ROMAN.match(words[-1]) and words[-1].isupper()):
        words = words[:-1]
    return ' '.join(words).strip(' .,;:')


def skippable(title: str) -> bool:
    """Front and back matter that carries no text worth retrieving (cover, contents, index...)."""
    return bool(SKIP.match(title.strip(' .')))


def merge(found: list[Heading]) -> list[Heading]:
    """One heading per page; a title repeated with nothing between (a running chapter header) kept once."""
    out: list[Heading] = []
    for h in sorted(found, key=lambda h: h.page):
        last = out[-1] if out else None
        if last and last.page == h.page:
            if key(h.title).startswith(key(last.title)):
                last.title = h.title
            elif key(h.title) not in key(last.title):
                last.title = _combine(last.title, h.title)
            continue
        if last and key(last.title) == key(h.title):
            continue
        out.append(h)
    return out


def _combine(first: str, second: str) -> str:
    """Two headings on one page; a shared 'Parent — ' prefix written once."""
    head, sep, _ = first.rpartition(' — ')
    if sep and second.startswith(head + sep):
        second = second[len(head + sep):]
    return f'{first} / {second}'


FINDERS = {
    'outline': from_outline,
    'contents': from_toc,
    'font': from_fonts,
    'lines': from_lines,
}


def find_headings(layout: Layout, source: str | None = None) -> tuple[str, list[Heading]]:
    """Headings from `source`, or from the first source (outline, contents, font, lines) that yields enough."""
    if source:
        return source, merge(FINDERS[source](layout))
    for name, finder in FINDERS.items():
        found = merge(finder(layout))
        good = [h for h in found if h.verified]
        floor = max(MIN_ENTRIES, len(layout.pages) / PAGES_PER_SECTION)
        if len(good) >= floor and len(good) >= 0.5 * len(found):
            return name, found
    return 'none', []
