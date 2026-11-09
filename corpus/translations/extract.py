import argparse
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from corpus.books.book_inspector import page_texts
from corpus.books.chunk import count
from corpus.books.headings import is_word
from corpus.paths import CODE_ROOT

# Cuts the translation books and the Unas website text into one passage per spell or plate, with king, wall and number.

TRANSLATE_DIR = CODE_ROOT / 'assets' / 'translate'
ALLEN_PDF = TRANSLATE_DIR / 'the-ancient-egyptian-pyramid-texts-james-p-allen.pdf'
FAULKNER_PDF = TRANSLATE_DIR / 'the-ancient-egyptian-coffin-texts-vol-i-spells-1-354-0856687545-9780856687549_compress.pdf'
ALLEN_TITLE = 'The Ancient Egyptian Pyramid Texts'
FAULKNER_TITLE = 'The Ancient Egyptian Coffin Texts, Vol. I'
UNAS_MD = CODE_ROOT / 'assets' / 'pyramid_texts_clean.md'      # made by clean_unas.py
UNAS_TITLE = 'Pyramid Texts of Unas'
UNAS_AUTHOR = 'Piankoff, modernised by Vincent Brown'
MAX_TOKENS = 420                 # a long spell is cut into parts of at most this many tokens
MIN_TOKENS = 12                   # shorter passages are headings or OCR debris

ALLEN_EVEN = re.compile(r'^\s*\d*\s*THE ANCIENT EGYPTIAN PYRAMID TEXTS\s*\d*\s*$')
ALLEN_ODD = re.compile(r'^\s*([A-Z][A-Z .]+?):\s+(.+?)\s+(\d+)\s*$')
ALLEN_SKIP = ('NOTES', 'VARIANTS', 'INTRODUCTION', 'CONCORDANCE')
CAPS = re.compile(r'^[A-Z0-9 ,.’\'“”()–—:;/&-]+$')
SPELL_START = re.compile(r'^(\d{1,3})\s*(?=[A-Z(“‘])')
NUMBER_ONLY = re.compile(r'^\d{1,3}$')
NOTE_MARK = re.compile(r'(?<=[A-Za-z’”.,:;!?)\]])\d{1,3}(?=[\s,.;:—)”’]|$)')
# Safer for the newer sources: a mark glued to a word, never the decimals of '25.0 x 8.3 cm'.
WORD_NOTE_MARK = re.compile(r'(?<=[A-Za-z’”)\]])\d{1,3}(?=[\s,.;:—)”’]|$)|(?<=[A-Za-z][.,:;!?”’])\d{1,3}(?=\s|$)')

CT_SPELL = re.compile(r'^\s*Spell[s]?\s*(\d*)\s*$')
CT_NOTE_START = re.compile(r'^\s+[1Il][.·]\s')
CT_NOTE = re.compile(r'^\s*\d{1,2}[.·]\s')
CT_MARGIN = re.compile(r'^\s*(?:[IVX]+\s*[,.]\s*[0-9IlS]+\s+|[Il]?[0-9][0-9Il]?\s+)')
CT_LAST_SPELL = 354              # volume I ends at Spell 354; a larger number is an OCR misread
CT_TRAIL = re.compile(r'\s+[0-9Il]?[0-9]{1,2}\s*$')

TEMPLE_DIR = CODE_ROOT / 'assets' / 'translations'
HYPOSTYLE_PDF = TEMPLE_DIR / 'the_great_xviii-dynasty.pdf'
HYPOSTYLE_TITLE = 'The Great Hypostyle Hall in the Temple of Amun at Karnak, Vol. 1.2'
HYPOSTYLE_AUTHOR = 'Brand, Feleg and Murnane'
HYPOSTYLE_PAGES = (45, 374)      # chapter 4, Translation and Commentary (printed pp. 29-358)
HYPOSTYLE_OFFSET = 16            # printed page = pdf page - offset
HYPOSTYLE_FURNITURE = {
    'oi.uchicago.edu',
    'The Great Hypostyle Hall in the Temple of Amun at Karnak',
    'Translation and Commentary',
}
PLATE = re.compile(r'^Plates? (\d+)\s*(.*)$')
WALL = re.compile(r'^(?:(?:West|South|North|East|Eastern) (?:Wall|Vestibule)|Vestibule of)[^.]*$')
FIELD = re.compile(r'^(Bibliography|Location|Subject|Type of Relief|Paint|Observations|Translations?|Notes?)\s*(?::\s*(.*))?$')
SPEAKER = re.compile(r'^[A-Z][A-Za-z’\'\- ]{1,40}:(?:\s|$)')   # 'By Amun:', 'Behind the King:', 'Label:', 'Left:'
FOOTNOTE = re.compile(r'^(\d{1,3}) \S')
SUPERSCRIPTS = re.compile('[⁰¹²³⁴⁵⁶⁷⁸⁹]')
NOTE_MARKS = '[a-p]'               # lettered note marks of the translations
KEEP_WORDS = {                   # real words that look like a word plus a note letter ('nomen' = 'nome' + 'n')
    'nomen', 'prenomen', 'atef', 'mose', 'idem', 'alia', 'seti', 'sethe', 'buto', 'aten', 'heri', 'hisn', 'sanaa',
}
KINGS = (               # the king column, and how a Subject line names him; the first named is the owner
    ('RAMESSES II', re.compile(r'\bRamesses II\b')),
    ('RAMESSES I', re.compile(r'\bRamesses I\b')),
    ('SETY I', re.compile(r'\bSety I\b')),
)
PRENOMENS = {'USERMAꜤATRE': 'RAMESSES II', 'MENMAꜤATRE': 'SETY I', 'MENPEHTYRE': 'RAMESSES I'}
HEAD_SHARE = 0.10                # page header band, ignored when finding the column gutter
FOOT_SHARE = 0.05
MIN_GUTTER = 8                   # points of empty strip that make a page two columns
SMALL_SHARE = 0.98               # a line under this share of the body type size is footnote type
TINY_SHARE = 0.7                 # smaller still: note letters and numbers, ignored
MAX_CROSSING = 2                 # words allowed across the gutter (a chapter title, a page number)
LOCATION_PREFIX = re.compile(r'^[A-Z] \d+[a-z]?(?:[–-]\d+[a-z]?)?:\s*')    # 'B 112:' or 'B 270–71:'

MEDINET_HABU_PDF = TEMPLE_DIR / 'XVII_Dynanisy_temple.pdf'
MEDINET_HABU_TITLE = 'Medinet Habu X: The Eighteenth Dynasty Temple, Part II'
MEDINET_HABU_AUTHOR = 'The Epigraphic Survey'
MEDINET_HABU_PAGES = (29, 93)    # Translations of the Texts and Commentary
MEDINET_HABU_OFFSET = 22
MEDINET_HABU_FURNITURE = {
    'isac.uchicago.edu',
    'TRANSLATIONS OF THE TEXTS AND COMMENTARY',
    'THE EIGHTEENTH DYNASTY TEMPLE, PART II: THE FAÇADE, PILLARS, AND ARCHITRAVE INSCRIPTIONS OF THE THUTMOSID PERIPTEROS',
}
MEDINET_HABU_JUNK = re.compile(        # the print slug and running headers, also as cut by the column crop
    r'^(?:MH X Comm|\d{1,2}/\d{1,2}/\d{2}\b|isac\.uchic|TRANSLATIONS OF THE|THE EIGHTEENTH DYNASTY TEMPLE'
    r'|PILLARS, AND ARCHITRAVE|ARCHITRAVE INSCRIPTIONS)'
)
MH_PLATE = re.compile(r'^PLATES? (\d+)([A-C]?)(?:[–-]\d+)?\. (.+)$')
MH_SPEAKER = re.compile(
    r'^(?:(?:Above|Before|Behind|Below|Beside)(?: and (?:Above|Before|Behind|Below))? (?:the )?[A-Z“][^.,:;]{0,30}'
    r'|(?:Right|Left)(?:most)? (?:Column|Half)|(?:South|North|East|West) (?:Text|Jamb)[a-p]?'
    r'|Inscription of [A-Z][\w ]{2,30})$'
)
MH_START = re.compile(r'^(?:x\+)?1(?=[A-Z(“‘\[<])')   # a translation opens with column number 1
MH_END = re.compile(r'^(?:Notes? to Translation|Epigraphic Comments)$')
MH_KINGS = (
    ('THUTMOSE III', re.compile(r'\bTHUTMOSE III\b|\bMenkheperre\b')),
    ('PTOLEMY VIII', re.compile(r'\bPTOLEMY VIII\b')),
    ('PAINEDJEM', re.compile(r'\bPAINEDJEM\b')),
    ('HATSHEPSUT', re.compile(r'\bHATSHEPSUT\b|\bMaatkare\b')),
)

BIRDS_PDF = CODE_ROOT / 'assets' / 'bird_in_ancient_egypt.pdf'
BIRDS_TITLE = 'Between Heaven and Earth: Birds in Ancient Egypt'
BIRDS_AUTHOR = 'Bailleul-LeSuer (ed.)'
BIRDS_PAGES = (131, 213)          # the Catalog, Nos. 1-40 (printed page = pdf page)
BIRDS_FURNITURE = {'http://oi.uchicago.edu', 'between heaven and earth: birds in ancient egypt'}
BIRDS_SECTIONS = {                # lowercase section titles that open a catalog section, before its introduction
    'birds in creation myths',
    'pharaoh the living horus and his avian subjects',
    'birds as protection in life',
    'fowling in the marshes and aviculture',
    'nina de garis davies’s facsimiles from the painted tomb-chapel of nebamun',
    'bird motifs in ancient egyptian arts and crafts',
    'birds in the writing system',
    'birds in the religious life of ancient egyptians',
    'falcon cults',
    'ibis cults',
    'birds in death and the afterlife',
}
CATALOG_CAPTION = re.compile(r'^Catalog No\. \d+$')
INITIALS_END = re.compile(r'[.”’)] [a-z]{2,4}$')     # '… wary lapwings. et': an essay's author initials
ENTRY_END = ('published', 'note', 'notes', 'references')
INSCRIPTION = re.compile(r'“[^”]*\b\w+\b[^”]*\b\w+\b[^”]*\b\w+\b[^”]*”|\binscri|\breads?\b|\btext\b')
CATALOG_KINGS = (                # kings a catalog inscription may name, as the table spells them
    ('UNIS', re.compile(r'\bUn[ia]s\b')),
    ('TETI', re.compile(r'\bTeti\b')),
    ('PEPI I', re.compile(r'\bPepi I\b')),
    ('PEPI II', re.compile(r'\bPepi II\b')),
    ('HATSHEPSUT', re.compile(r'\bHatshepsut\b')),
    ('THUTMOSE III', re.compile(r'\bThutmose III\b')),
    ('SETY I', re.compile(r'\bSe[tw][iy] I\b')),
    ('RAMESSES II', re.compile(r'\bRamesses II\b')),
)
TITLE_CONTINUES = re.compile(r'(?:\b(?:of|a|an|the|and|in|from|with|for|to|Showing|Shape|Decorated)|,)\s*$')


def title_case(text: str) -> str:
    """str.title() that keeps roman numerals: 'RAMESSES II' -> 'Ramesses II', not 'Ramesses Ii'."""
    return re.sub(r'\b[IVX][ivx]+\b', lambda m: m.group(0).upper(), text.title())


@dataclass
class Passage:
    source: str                   # 'pyramid_texts' | 'coffin_texts'
    work: str
    author: str
    king: str | None
    location: str | None
    section: str | None
    spell: int | None
    title: str | None
    part: int
    page: str
    body: str
    tokens: int
    unit: str = 'Spell'           # 'Utterance' for the Unas website (Sethe's numbering)

    @property
    def id(self) -> str:
        """Stable id: source, king, spell and part."""
        return f'{self.source}:{(self.king or "-").lower()}:{self.spell}:{self.part}'

    @property
    def citation(self) -> str:
        """'Allen, Pyramid Texts › Unis, Antechamber, West Gable › Spell 158 › p. 41'."""
        where = ', '.join(title_case(x) for x in (self.king, self.location) if x)
        name = f'{self.unit} {self.spell}' + (f' ({self.title.capitalize()})' if self.title else '')
        page = f'p. {self.page}' if self.page else ''
        return ' › '.join(x for x in (f'{self.author}, {self.work}', where, name, page) if x)


def pages(pdf: Path) -> list[str]:
    """The PDF's text, one string per page, with the physical layout kept."""
    text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
    return text.split('\f')


def join_lines(lines: list[str], marks: re.Pattern[str] = NOTE_MARK) -> str:
    """One paragraph: soft hyphens rejoined, footnote markers dropped, spaces collapsed."""
    text = '\n'.join(lines)
    text = re.sub(r'(\w)-\s*\n\s*([a-z])', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return marks.sub('', text)


def split_long(body: str, limit: int = MAX_TOKENS) -> list[str]:
    """Cut at sentence ends so each part stays under the token limit."""
    if count(body) <= limit:
        return [body]
    parts: list[str] = []
    current = ''
    for sentence in re.split(r'(?<=[.!?”])\s+', body):
        if current and count(current) + count(sentence) > limit:
            parts.append(current)
            current = sentence
        else:
            current = f'{current} {sentence}'.strip()
    return parts + ([current] if current else [])


def allen(pdf: Path = ALLEN_PDF) -> list[Passage]:
    """Allen's Pyramid Texts: a passage per titled spell, located by the page's running header."""
    king: str | None = None
    location: str | None = None
    page = ''
    section: str | None = None
    spells: list[dict[str, object]] = []
    caps_run: list[str] = []
    for raw in pages(pdf):
        lines = [line for line in raw.split('\n') if line.strip()]
        if not lines:
            continue
        head = lines[0]
        odd = ALLEN_ODD.match(head)
        if odd:
            king, location, page = odd.group(1).strip(), odd.group(2).strip(), odd.group(3)
            if any(location.startswith(s) for s in ALLEN_SKIP) or king in ALLEN_SKIP:
                king = None
            lines = lines[1:]
        elif ALLEN_EVEN.match(head):
            page = re.match(r'^\s*(\d*)', head).group(1) or page
            lines = lines[1:]
        elif head.strip() in ('INTRODUCTION', 'VARIANTS', 'I', 'II', 'III', 'IV', 'V', 'VI'):
            king = None
        if king is None:
            continue
        for line in lines:
            text = line.strip()
            if CAPS.match(text) and re.search(r'[A-Z]{3}', text) and not SPELL_START.match(text):
                caps_run.append(text)
                continue
            if caps_run and caps_run[-1] == 'NOTES':
                king = None
                caps_run = []
                break
            if caps_run:
                *above, title = caps_run
                title = NOTE_MARK.sub('', title)
                if above:
                    joined = ' '.join(above)
                    where = re.search(r'\(([^)]*(?:CHAMBER|CORRIDOR|GABLE|WALL|PASSAGE|VESTIBULE)[^)]*)\)', joined)
                    if where:
                        # '(ANTECHAMBER, WEST GABLE)' moves the wall before the page header does.
                        location = where.group(1).strip()
                    section = re.sub(r'\s*\([^)]*\)', '', joined).strip().title()
                spells.append({'king': king, 'location': location, 'section': section, 'title': title,
                               'page': page, 'spell': None, 'lines': []})
                caps_run = []
            if not spells:
                continue
            current = spells[-1]
            alone = NUMBER_ONLY.match(text)
            start = alone or SPELL_START.match(text)
            number = int(start.group(1) if not alone else text) if start else None
            # Spell numbers rise within a pyramid; any other number at a line start is not a spell.
            last = max((s['spell'] for s in spells if s['king'] == king and s['spell']), default=0)
            if number is not None and last < number <= last + 80:
                if current['spell'] is not None:
                    # A second spell under the same heading. A margin number alone follows its first line.
                    carried = current['lines'][-1:] if alone else []
                    current['lines'] = current['lines'][:-1] if alone else current['lines']
                    current = {**current, 'spell': number, 'lines': carried, 'page': page}
                    spells.append(current)
                current['spell'] = number
                if alone:
                    continue
                text = text[start.end():]
            elif alone:
                continue
            current['lines'].append(text)
    out = [
        Passage('pyramid_texts', ALLEN_TITLE, 'Allen', s['king'], s['location'], s['section'], s['spell'],
                s['title'], part, s['page'], body, count(body))
        for s in spells
        if s['spell'] is not None
        for part, body in enumerate(split_long(join_lines(s['lines'])))
        if count(body) >= MIN_TOKENS and not body.startswith('(See ')
    ]
    return unique_parts(out)


def unique_parts(items: list[Passage]) -> list[Passage]:
    """Renumber parts so a spell number repeated under one king never collides."""
    seen: dict[tuple[str | None, int | None], int] = {}
    for p in items:
        key = (p.king, p.spell)
        p.part = seen.get(key, 0)
        seen[key] = p.part + 1
    return items


def faulkner(pdf: Path = FAULKNER_PDF) -> list[Passage]:
    """Faulkner's Coffin Texts I (OCR): a passage per spell, its footnotes and margin references dropped."""
    spells: list[dict[str, object]] = []
    for number, raw in enumerate(pages(pdf), start=1):
        for line in raw.split('\n'):
            header = CT_SPELL.match(line)
            if header:
                previous = spells[-1]['spell'] if spells else 0
                spell = int(header.group(1)) if header.group(1) else previous + 1
                if not previous < spell <= CT_LAST_SPELL:
                    spell = previous + 1
                spells.append({'spell': spell, 'page': str(number), 'lines': [], 'notes': False})
                continue
            if not spells or not line.strip():
                continue
            current = spells[-1]
            if CT_NOTE_START.match(line) or CT_NOTE.match(line):
                current['notes'] = True
            if current['notes']:
                continue
            text = CT_TRAIL.sub('', CT_MARGIN.sub('', line)).strip()
            if text and not NUMBER_ONLY.match(text):
                current['lines'].append(text)
    seen: set[int] = set()
    out: list[Passage] = []
    for s in spells:
        if s['spell'] in seen:
            continue
        seen.add(s['spell'])
        for part, body in enumerate(split_long(join_lines(s['lines']))):
            if count(body) >= MIN_TOKENS:
                out.append(Passage('coffin_texts', FAULKNER_TITLE, 'Faulkner', None, None, None, s['spell'],
                                   None, part, s['page'], body, count(body)))
    return out


def unas_online(md: Path = UNAS_MD) -> list[Passage]:
    """The cleaned Unas website text (clean_unas.py): a passage per utterance, cut at sentence ends when long."""
    out: list[Passage] = []
    wall = ''
    for block in re.split(r'^(?=## |### )', md.read_text(), flags=re.M):
        head, _, rest = block.partition('\n')
        if head.startswith('## '):
            wall = head[3:].strip()
            continue
        utterance = re.match(r'^### Utterance (\d+)\s*(.*)$', head)
        if not utterance:
            continue
        meta = re.search(r'<!-- walls: (.*?) \|', rest)
        walls = meta.group(1).split('; ') if meta else [wall]
        text = re.sub(r'<!--.*?-->', '', rest).strip()
        note = utterance.group(2).strip('() ') or None
        also = f'also on: {"; ".join(walls[1:])}' if len(walls) > 1 else None
        for part, body in enumerate(split_long(join_lines(text.split('\n')))):
            if count(body) >= MIN_TOKENS:
                out.append(Passage('unas_online', UNAS_TITLE, UNAS_AUTHOR, 'UNIS', wall, also,
                                   int(utterance.group(1)), note, part, '', body, count(body), 'Utterance'))
    return unique_parts(out)


def gutter(words: list[tuple[float, float, float]], width: float, height: float) -> float | None:
    """The x of the widest word-free strip in the middle third of a page's body, or None for one column."""
    crossing = [0] * (int(width) + 1)
    for x_min, x_max, y in words:
        if HEAD_SHARE * height < y < (1 - FOOT_SHARE) * height:
            for x in range(int(x_min), min(int(x_max) + 1, len(crossing))):
                crossing[x] += 1
    covered = [n > MAX_CROSSING for n in crossing]    # a chapter title or page number may cross the gutter
    best: tuple[int, int] = (0, 0)
    start = None
    for x in range(int(width / 3), int(2 * width / 3) + 1):
        if not covered[x] and start is None:
            start = x
        if (covered[x] or x == int(2 * width / 3)) and start is not None:
            best = max(best, (x - start, start))
            start = None
    return best[1] + best[0] / 2 if best[0] >= MIN_GUTTER else None


def footnote_top(words: list[tuple[float, float, float, float]], height: float) -> float:
    """The y where a column's footnotes begin: the run of smaller-type lines at its foot (or the page bottom)."""
    lines: dict[int, list[float]] = {}
    for _, _, y_min, y_max in words:
        if y_min < (1 - FOOT_SHARE) * height:
            lines.setdefault(round(y_min), []).append(y_max - y_min)
    if not lines:
        return height
    sizes = {y: statistics.median(h) for y, h in lines.items()}
    body = statistics.median(sizes.values())
    top = height
    footnote = False
    for y in sorted(sizes, reverse=True):
        size = sizes[y]
        if size < TINY_SHARE * body:
            continue                                 # a line of note letters or a footnote number
        if size >= SMALL_SHARE * body:
            break                                    # body text: the footnotes (if any) start below
        footnote = True
        top = y
    return top - 1 if footnote else height


def column_texts(pdf: Path, first: int, last: int) -> list[tuple[int, str]]:
    """(pdf page, text) per column of each page, left column first: two-column pages cropped at their gutter."""
    html = subprocess.run(['pdftotext', '-bbox', '-f', str(first), '-l', str(last), str(pdf), '-'],
                          capture_output=True, text=True, check=True).stdout
    out: list[tuple[int, str]] = []
    for number, page in enumerate(re.findall(r'<page (.*?)</page>', html, flags=re.S), start=first):
        width, height = (float(v) for v in re.search(r'width="([\d.]+)" height="([\d.]+)"', page).groups())
        boxes = [(float(a), float(c), float(b), float(d)) for a, b, c, d in re.findall(
            r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)"', page)]
        cut = gutter([(a, c, b) for a, c, b, _ in boxes], width, height)
        crops = [(0, width)] if cut is None else [(0, cut), (cut, width - cut)]
        for x, w in crops:
            bottom = footnote_top([b for b in boxes if x <= b[0] < x + w], height)
            text = subprocess.run(
                ['pdftotext', '-f', str(number), '-l', str(number), '-x', str(int(x)), '-y', '0',
                 '-W', str(int(w) + 1), '-H', str(int(bottom)), str(pdf), '-'],
                capture_output=True, text=True, check=True,
            ).stdout
            out.append((number, text))
    return out


def body_lines(
    pdf: Path,
    first: int,
    last: int,
    furniture: set[str],
    junk: re.Pattern[str] | None = None,
    columns: bool = False,
    footnotes: bool = True,
) -> list[tuple[int, str]]:
    """(pdf page, line) of the body text: running headers, page numbers and footnotes (per page or column) dropped."""
    out: list[tuple[int, str]] = []
    previous: set[str] = set()
    last_note = 0
    segments = (column_texts(pdf, first, last) if columns
                else list(enumerate(page_texts(pdf, first, last), start=first)))
    for number, raw_text in segments:
        raw_text = '\n'.join(ln for ln in raw_text.split('\n') if not (junk and junk.match(ln.strip())))
        anchors = set(NOTE_MARK.findall(raw_text))
        for line in raw_text.split('\n'):
            text = line.strip()
            note = FOOTNOTE.match(text)
            # A footnote: its mark is in the body, and footnote numbers only rise through a book.
            # Column crops already end above their footnotes (footnote_top), so no cut there.
            if footnotes and not columns and note and note.group(1) in anchors | previous and int(note.group(1)) > last_note:
                last_note = int(note.group(1))
                break                                # footnotes close the page
            if columns and len(text) >= 4 and any(text in f for f in furniture):
                continue                             # a running header cut by the column crop
            if text and text not in furniture and not NUMBER_ONLY.match(text):
                out.append((number, text))
        previous = anchors
    return out


def strip_note_letters(text: str) -> str:
    """Remove lettered note marks a-h: 'body.a' -> 'body.', 'threeb times' -> 'three times' (dictionary-checked)."""
    text = re.sub(rf'(?<=[.,;:!?”’)\]>]){NOTE_MARKS}(?=[\s.,;:”’)]|$)', '', text)

    def fix(m: re.Match[str]) -> str:
        word = m.group(0)
        keep = word.lower() in KEEP_WORDS or is_word(word)
        return word[:-1] if not keep and is_word(word[:-1]) else word

    # Whole words only: '[he]alth' and 'pre-eminent' are fragments, never marked.
    return re.sub(rf'(?<![\[\]<>\w])[A-Za-z]{{3,}}{NOTE_MARKS}(?![\w\[\]<>])', fix, text)


def first_sentence(text: str) -> str:
    """The opening sentence of a Subject line: the scene in brief."""
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text, maxsplit=1)[0]


def speakers(lines: list[str]) -> list[list[str]]:
    """The Translations lines grouped by speaker label, one group per 'By Amun:' or 'Label:' block."""
    groups: list[list[str]] = []
    for line in lines:
        if SPEAKER.match(line) or not groups:
            groups.append([])
        groups[-1].append(line)
    return groups


def split_clauses(sentence: str, limit: int = MAX_TOKENS) -> list[str]:
    """A sentence over the limit (a litany) cut after commas and semicolons."""
    if count(sentence) <= limit:
        return [sentence]
    parts: list[str] = []
    current = ''
    for clause in re.split(r'(?<=[,;])\s+', sentence):
        if current and count(current) + count(clause) > limit:
            parts.append(current)
            current = clause
        else:
            current = f'{current} {clause}'.strip()
    return parts + ([current] if current else [])


def pack(blocks: list[str], limit: int = MAX_TOKENS) -> list[str]:
    """Speaker blocks joined one per line into parts under the token limit; a long block is cut at sentences."""
    parts: list[str] = []
    current = ''
    pieces = (c for b in blocks for s in split_long(b, limit) for c in split_clauses(s, limit))
    for piece in pieces:
        if current and count(current) + count(piece) > limit:
            parts.append(current)
            current = piece
        else:
            current = f'{current}\n{piece}' if current else piece
    return parts + ([current] if current else [])


def plate_king(subject: str, body: str) -> str | None:
    """The king the Subject names first, else the one whose prenomen the texts give."""
    named = [(m.start(), king) for king, pattern in KINGS for m in [pattern.search(subject)] if m]
    if named:
        return min(named)[1]
    return next((king for name, king in PRENOMENS.items() if name in body), None)


def hypostyle(pdf: Path = HYPOSTYLE_PDF) -> list[Passage]:
    """The Great Hypostyle Hall: a passage per plate, its Translations block only, with location and subject."""
    plates: list[dict[str, object]] = []
    wall: str | None = None
    state = 'other'
    for page, text in body_lines(pdf, *HYPOSTYLE_PAGES, HYPOSTYLE_FURNITURE):
        plate = PLATE.match(text)
        if plate:
            plates.append({'number': int(plate.group(1)), 'label': plate.group(2).strip() or None, 'wall': wall,
                           'page': str(page - HYPOSTYLE_OFFSET), 'location': [], 'subject': [], 'lines': []})
            state = 'other'
            continue
        if WALL.match(text) and len(text) < 90:
            wall = re.sub(r'\s*\(.*\)$', '', text)
            state = 'other'
            continue
        if not plates:
            continue
        current = plates[-1]
        field = FIELD.match(text)
        if field:
            name, text = field.group(1), field.group(2) or ''
            state = {'Location': 'location', 'Subject': 'subject', 'Translation': 'lines', 'Translations': 'lines',
                     'Note': 'notes', 'Notes': 'notes'}.get(name, 'other')
        if state in ('location', 'subject', 'lines') and text:
            current[state].append(text)
    out: list[Passage] = []
    for p in plates:
        blocks = [strip_note_letters(SUPERSCRIPTS.sub('', join_lines(b, WORD_NOTE_MARK))) for b in speakers(p['lines'])]
        subject = first_sentence(join_lines(p['subject'], WORD_NOTE_MARK))
        location = LOCATION_PREFIX.sub('', join_lines(p['location'], WORD_NOTE_MARK)).rstrip('.') or None
        king = plate_king(subject, '\n'.join(blocks))
        for part, chunk in enumerate(pack(blocks)):
            text = f'Scene: {subject}\n{chunk}' if subject else chunk
            if count(chunk) >= MIN_TOKENS:
                out.append(Passage('hypostyle', HYPOSTYLE_TITLE, HYPOSTYLE_AUTHOR, king, location, p['wall'],
                                   p['number'], p['label'], part, p['page'], text, count(text), 'Plate'))
    return unique_parts(out)


def catalog_lines(pdf: Path) -> list[tuple[int, str]]:
    """(page, line) of the birds catalog in PDF order, with each section's introduction cut out."""
    lines = [(n, line.strip())
             for n, raw in enumerate(page_texts(pdf, *BIRDS_PAGES, raw=True), start=BIRDS_PAGES[0])
             for line in raw.split('\n') if line.strip()]
    out: list[tuple[int, str]] = []
    skip_to = -1
    for i, (page, text) in enumerate(lines):
        if i <= skip_to:
            continue
        if text.lower() in BIRDS_SECTIONS:
            # A section title with a 'Catalog No.' caption under it opens an introduction: cut to its author's initials.
            caption = i + 1 < len(lines) and CATALOG_CAPTION.match(lines[i + 1][1])
            end = next((j for j in range(i + 2, min(i + 60, len(lines)))
                        if lines[j][0] == page and INITIALS_END.search(lines[j][1])), None)
            skip_to = end if caption and end is not None else i
            continue
        if text in BIRDS_FURNITURE or NUMBER_ONLY.match(text) or CATALOG_CAPTION.match(text):
            continue
        out.append((page, text))
    return out


def quoted_king(body: str) -> str | None:
    """The one king named inside the entry's quoted inscriptions, else None ('…Son of Re Unis…' -> UNIS)."""
    quoted = ' '.join(re.findall(r'“[^”]*”', body))
    named = {king for king, pattern in CATALOG_KINGS if pattern.search(quoted)}
    return named.pop() if len(named) == 1 else None


def bird_catalog(pdf: Path = BIRDS_PDF) -> list[Passage]:
    """The birds catalog: a passage per numbered object whose entry quotes or discusses an inscription."""
    entries: list[dict[str, object]] = []
    collecting = False
    previous = ''
    for page, text in catalog_lines(pdf):
        last = entries[-1]['number'] if entries else 0
        # The next number, or the one after it: No. 33 has no heading in the text layer.
        heading = re.match(rf'^({last + 1}|{last + 2})\.\s+(\S.*)$', text)
        if heading and not re.search(r'\bNos?\.$', previous):
            entries.append({'number': int(heading.group(1)), 'title': [heading.group(2)], 'page': str(page),
                            'lines': []})
            collecting = True
        elif entries and collecting:
            current = entries[-1]
            if len(current['lines']) == 0 and TITLE_CONTINUES.search(current['title'][-1]):
                current['title'].append(text)
            elif text.lower() in ENTRY_END:
                collecting = False
            else:
                current['lines'].append(text)
        previous = text
    out: list[Passage] = []
    for e in entries:
        body = join_lines(e['lines'], WORD_NOTE_MARK)
        body = re.sub(r'\s[a-z]{2,4}$', '', body)            # the author's initials
        if not INSCRIPTION.search(body):
            continue
        title = ' '.join(e['title'])
        for part, chunk in enumerate(split_long(body)):
            if count(chunk) >= MIN_TOKENS:
                out.append(Passage('birds_catalog', BIRDS_TITLE, BIRDS_AUTHOR, quoted_king(body), None, None, e['number'],
                                   title, part, e['page'], chunk, count(chunk), 'Catalog No.'))
    return out


def heading_case(title: str) -> str:
    """'THUTMOSE III RECEIVED BY AMUN-RE' -> 'Thutmose III Received by Amun-Re'."""
    small = {'a', 'an', 'and', 'at', 'by', 'for', 'from', 'in', 'of', 'on', 'the', 'to', 'with', 'before'}
    words = [w if re.fullmatch(r'[IVX]+[,.;:]?', w) else '-'.join(x.capitalize() for x in w.split('-')) for w in title.split()]
    return ' '.join(w.lower() if i and w.lower() in small else w for i, w in enumerate(words))


def mh_clean(text: str) -> str:
    """A Medinet Habu speaker block: note letters, 'Re' + letter, and split hyphenated epithets mended."""
    text = strip_note_letters(text)
    text = re.sub(rf'\bRe{NOTE_MARKS}\b', 'Re', text)
    text = re.sub(rf'(?<![\w\[]){NOTE_MARKS}(?=[A-Z][a-z])', '', text)     # 'aTake for yourself'
    return re.sub(r'(?<=\w) -(?=[a-z])', '-', text)


def medinet_habu(pdf: Path = MEDINET_HABU_PDF) -> list[Passage]:
    """Medinet Habu X: a passage per plate, its translation only (notes and epigraphic comments dropped)."""
    plates: list[dict[str, object]] = []
    collecting = False
    lines = body_lines(pdf, *MEDINET_HABU_PAGES, MEDINET_HABU_FURNITURE, MEDINET_HABU_JUNK, columns=True)
    for page, text in lines:
        plate = MH_PLATE.match(text)
        if plate:
            plates.append({'number': int(plate.group(1)), 'label': plate.group(2) or None, 'title': [plate.group(3)],
                           'page': str(page - MEDINET_HABU_OFFSET), 'lines': [], 'started': False})
            collecting = True
            continue
        if not plates or not collecting:
            continue
        current = plates[-1]
        if MH_END.match(text):
            collecting = False
        elif not current['started'] and text.isupper():
            current['title'].append(re.sub(r'\d+$', '', text))     # the heading wraps; a trailing note number goes
        elif text == 'Translation':
            current['started'] = True
        elif MH_SPEAKER.match(text):
            current['lines'].append(f'{text}:')
            current['started'] = True
        elif current['started'] or MH_START.match(text):    # commentary prose never opens with column 1
            current['started'] = True
            # Column numbers: glued to a word ('2as he gives', '1(Long) live') or alone at the line end.
            text = re.sub(r'(?<![\w,.])\d{1,2}(?=[A-Za-z(“‘\[<])', '', text)
            current['lines'].append(re.sub(r'\s\d{1,2}$', '', text))
    out: list[Passage] = []
    for p in plates:
        blocks = [mh_clean(join_lines(b, WORD_NOTE_MARK)) for b in speakers(p['lines'])]
        title = ' '.join(p['title']).rstrip('.')
        body = '\n'.join(blocks)
        king = next((k for k, pattern in MH_KINGS if pattern.search(title) or pattern.search(body)), None)
        for part, chunk in enumerate(pack(blocks)):
            if count(chunk) >= MIN_TOKENS:
                text = f'Scene: {heading_case(title)}\n{chunk}'
                out.append(Passage('medinet_habu', MEDINET_HABU_TITLE, MEDINET_HABU_AUTHOR, king, None, None,
                                   p['number'], p['label'], part, p['page'], text, count(text), 'Plate'))
    return unique_parts(out)


def passages() -> list[Passage]:
    """Every passage of the translation books, the Unas website text, the temple volumes and the birds catalog."""
    return allen() + faulkner() + unas_online() + hypostyle() + medinet_habu() + bird_catalog()


def main() -> None:
    parser = argparse.ArgumentParser(description='Cut the translation books into spell passages and print a report.')
    parser.add_argument('--show', type=int, default=3, help='sample passages to print per book')
    args = parser.parse_args()

    for name, items in (
        ('allen', allen()),
        ('faulkner', faulkner()),
        ('unas_online', unas_online()),
        ('hypostyle', hypostyle()),
    ):
        tokens = [p.tokens for p in items]
        kings = sorted({p.king for p in items if p.king})
        print(f'{name}: {len(items):,} passages, {sum(tokens):,} tokens, max {max(tokens)}, '
              f'spells {len({(p.king, p.spell) for p in items}):,}, kings {kings}')
        step = max(1, len(items) // max(1, args.show))
        for p in items[::step][:args.show]:
            print(f'  [{p.id}] {p.citation}\n    {p.body[:300]}')


if __name__ == '__main__':
    main()
