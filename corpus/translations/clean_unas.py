import argparse
import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

from corpus.paths import CODE_ROOT

# Cleans the scraped Unas translation: one copy of each utterance per distinct text, with every wall it is on.

RAW = CODE_ROOT / 'assets' / 'pyramid_texts.md'
CLEAN = CODE_ROOT / 'assets' / 'pyramid_texts_clean.md'
SAME_TEXT = 0.90                  # two versions of one utterance at least this similar are one text
SOURCE = re.compile(r'^<!-- source: (\S+) -->$', re.M)
HEADING = re.compile(r'^(#{1,3}) (.*)$')
UTTERANCE = re.compile(r'^Utterance (\d+)(.*)$')
WALL_WORDS = ('Wall', 'Gable', 'Register')
HALL_ORDER = ('Sarcophagus Chamber', 'Passage', 'Antechamber', 'Corridor')


@dataclass
class Version:
    number: int
    note: str                     # e.g. '(continued from the West Wall)'
    wall: str
    url: str
    text: str
    walls: list[str] = field(default_factory=list)


def wall_name(heading: str, previous: str | None) -> str | None:
    """'Pyramid Texts — Antechamber West Wall Gable Translation' -> 'Antechamber West Gable'; None if not a wall."""
    name = re.sub(r'^Pyramid Texts —\s*', '', heading).replace(' Translation', '').strip()
    if not any(w in name for w in WALL_WORDS):
        return None
    name = name.replace('Wall Gable', 'Gable')
    if name.startswith('First Register') or name.startswith('North Wall -'):
        register = name.split('-')[-1].strip()
        return f'Sarcophagus Chamber North Wall, {register}'
    return name


def versions(raw: str) -> list[Version]:
    """Every utterance on every scraped page, with the wall it was listed under."""
    parts = SOURCE.split(raw)
    out: list[Version] = []
    for url, page in zip(parts[1::2], parts[2::2]):
        wall: str | None = None
        current: Version | None = None
        for line in page.split('\n'):
            heading = HEADING.match(line)
            if heading:
                level, title = heading.groups()
                utterance = UTTERANCE.match(title)
                if level == '###' and utterance:
                    current = Version(int(utterance.group(1)), utterance.group(2).strip(), wall or '', url, '')
                    out.append(current)
                    continue
                named = wall_name(title, wall)
                if named:
                    wall, current = named, None
                    continue
                if title.startswith('Utterances') or title in HALL_ORDER or level == '#':
                    current = None
                    continue
                line = title       # a line of spell text set as a heading, e.g. '## Bennu says'
            if current is not None and line.strip() and line.strip() != '---':
                current.text += line.strip() + '\n'
    return [v for v in out if v.text.strip()]


def normal(text: str) -> str:
    """Lowercase letters and digits only, for comparing two versions."""
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def merge(items: list[Version]) -> list[Version]:
    """One version per distinct text of each utterance, longest kept, every wall remembered."""
    kept: list[Version] = []
    for v in sorted(items, key=lambda v: (v.number, -len(v.text))):
        for k in kept:
            if k.number == v.number and difflib.SequenceMatcher(None, normal(k.text), normal(v.text)).ratio() >= SAME_TEXT:
                if v.wall and v.wall not in k.walls:
                    k.walls.append(v.wall)
                break
        else:
            v.walls = [v.wall] if v.wall else []
            kept.append(v)
    return kept


def hall_rank(wall: str) -> int:
    """Walls sorted by hall, burial chamber first."""
    return next((i for i, h in enumerate(HALL_ORDER) if wall.startswith(h)), len(HALL_ORDER))


def render(kept: list[Version]) -> str:
    """The clean Markdown: # title, ## wall, ### utterance, with the other walls in a comment."""
    out = ['# Pyramid Texts of Unas',
           '',
           "Alexandre Piankoff's translation, modernised by Vincent Brown (pyramidtextsonline.com).",
           'Cleaned by corpus/translations/clean_unas.py: one copy per distinct text.',
           '']
    walls = sorted({v.walls[0] for v in kept if v.walls}, key=lambda w: (hall_rank(w), w))
    for wall in walls + ['']:
        here = [v for v in kept if (v.walls[0] if v.walls else '') == wall]
        if not here:
            continue
        out += [f'## {wall or "Unplaced"}', '']
        for v in sorted(here, key=lambda v: v.number):
            out.append(f'### Utterance {v.number}' + (f' {v.note}' if v.note else ''))
            out.append(f'<!-- walls: {"; ".join(v.walls)} | source: {v.url} -->')
            out += [v.text.strip(), '']
    return '\n'.join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description='Deduplicate the scraped Unas Pyramid Texts into one clean Markdown file.')
    parser.add_argument('--raw', type=Path, default=RAW, help=f'default {RAW}')
    parser.add_argument('--out', type=Path, default=CLEAN, help=f'default {CLEAN}')
    args = parser.parse_args()

    found = versions(args.raw.read_text())
    kept = merge(found)
    args.out.write_text(render(kept))
    multi = [v for v in kept if len(v.walls) > 1]
    print(f'{len(found):,} utterance copies -> {len(kept):,} kept '
          f'({len({v.number for v in kept})} utterance numbers, {len(multi)} on more than one wall)')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
