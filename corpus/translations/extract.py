import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from corpus.books.chunk import count
from corpus.paths import CODE_ROOT

# Cuts the two translation books and the Unas website text into one passage per spell, with king, wall and number.

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

CT_SPELL = re.compile(r'^\s*Spell[s]?\s*(\d*)\s*$')
CT_NOTE_START = re.compile(r'^\s+[1Il][.·]\s')
CT_NOTE = re.compile(r'^\s*\d{1,2}[.·]\s')
CT_MARGIN = re.compile(r'^\s*(?:[IVX]+\s*[,.]\s*[0-9IlS]+\s+|[Il]?[0-9][0-9Il]?\s+)')
CT_LAST_SPELL = 354              # volume I ends at Spell 354; a larger number is an OCR misread
CT_TRAIL = re.compile(r'\s+[0-9Il]?[0-9]{1,2}\s*$')


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
        where = ', '.join(x.title() for x in (self.king, self.location) if x)
        name = f'{self.unit} {self.spell}' + (f' ({self.title.capitalize()})' if self.title else '')
        page = f'p. {self.page}' if self.page else ''
        return ' › '.join(x for x in (f'{self.author}, {self.work}', where, name, page) if x)


def pages(pdf: Path) -> list[str]:
    """The PDF's text, one string per page, with the physical layout kept."""
    text = subprocess.run(['pdftotext', '-layout', str(pdf), '-'], capture_output=True, text=True, check=True).stdout
    return text.split('\f')


def join_lines(lines: list[str]) -> str:
    """One paragraph: soft hyphens rejoined, footnote markers dropped, spaces collapsed."""
    text = '\n'.join(lines)
    text = re.sub(r'(\w)-\s*\n\s*([a-z])', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return NOTE_MARK.sub('', text)


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


def passages() -> list[Passage]:
    """Every passage of the two books and the Unas website text."""
    return allen() + faulkner() + unas_online()


def main() -> None:
    parser = argparse.ArgumentParser(description='Cut the translation books into spell passages and print a report.')
    parser.add_argument('--show', type=int, default=3, help='sample passages to print per book')
    args = parser.parse_args()

    for name, items in (('allen', allen()), ('faulkner', faulkner()), ('unas_online', unas_online())):
        tokens = [p.tokens for p in items]
        kings = sorted({p.king for p in items if p.king})
        print(f'{name}: {len(items):,} passages, {sum(tokens):,} tokens, max {max(tokens)}, '
              f'spells {len({(p.king, p.spell) for p in items}):,}, kings {kings}')
        step = max(1, len(items) // max(1, args.show))
        for p in items[::step][:args.show]:
            print(f'  [{p.id}] {p.citation}\n    {p.body[:300]}')


if __name__ == '__main__':
    main()
