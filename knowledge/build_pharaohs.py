import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
DEFAULT_HTML = HERE / 'List of pharaohs .html'
DEFAULT_OUTPUT = HERE / 'master_pharaohs_v2.json'
PAGE_URL = 'https://en.wikipedia.org/wiki/List_of_pharaohs'
LICENSE = 'CC BY-SA 4.0, Wikipedia contributors'

NAME_COLUMNS = {'personal name': 'personal_name', 'horus name': 'horus_name',
                'egyptian name': 'personal_name', 'name': 'personal_name', 'priest': 'personal_name'}
OTHER_COLUMNS = {'throne name': 'throne_name', 'greek name': 'greek_name', 'notes': 'notes',
                 'reign': 'reign_period', '#': 'number'}
EMPTY_VALUES = {'', '–', '—', '-', '?', 'n/a'}

UNITS = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7,
         'eighth': 8, 'ninth': 9}
TEENS = {'tenth': 10, 'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14,
         'fifteenth': 15, 'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19,
         'twentieth': 20, 'thirtieth': 30}
TENS = {'twenty': 20, 'thirty': 30}

YEAR_RANGE = re.compile(
    r'(?P<circa>c\.\s*|ca\.\s*|~\s*)?(?P<start>\d{1,4})\s*(?P<start_era>BCE?|AD|CE)?\s*[–—−-]\s*'
    r'(?:c\.\s*)?(?P<end>\d{1,4})(?!\s*(?:⁄|\+|years?|months?|days?|\d))\s*(?P<era>BCE?|AD|CE)?',
    re.IGNORECASE)
SINGLE_YEAR = re.compile(r'(?P<circa>c\.\s*|ca\.\s*|~\s*)?(?P<year>\d{3,4})\s*(?P<era>BCE?|AD|CE)\b',
                         re.IGNORECASE)
LENGTH = re.compile(
    r'(?:c\.\s*|~\s*|over\s+)?\d+(?:\s*[–-]\s*\d+)?(?:\s*\+\s*\d+\s*⁄\s*\d+)?\s*'
    r'(?:years?|months?|days?)(?:,?\s*(?:\d+(?:\s*[–-]\s*\d+)?\s*(?:months?|days?)))*',
    re.IGNORECASE)
MONTHS = ('January|February|March|April|May|June|July|August|September|October|November|December'
          '|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec')
CALENDAR_DAY = re.compile(rf'\b(?:\d{{1,2}}\s+)?(?:{MONTHS})\b\.?\s*', re.IGNORECASE)
SPLIT_YEAR = re.compile(r'(\d+)/\d+')
PERIOD_RANGE = re.compile(r'\((?P<dates>[^()]*\d[^()]*)\)\s*$')
HORUS_PREFIX = re.compile(r'^known by his Horus name\s+', re.IGNORECASE)
LOST_NAME = re.compile(r'^(?:\d+\s+)?(?:names?\s+)?(?:lost|destroyed|unknown)$|^names?\s+(?:lost|destroyed)',
                       re.IGNORECASE)


def clean(text):
    return re.sub(r'\s+', ' ', (text or '').replace('\xa0', ' ')).strip()


def plain_text(cell):
    """Return a cell's visible text, without footnote markers or inline styles."""
    copy = BeautifulSoup(str(cell), 'html.parser')
    for hidden in copy.select('sup, style, .mw-editsection'):
        hidden.decompose()
    for line_break in copy.find_all('br'):
        line_break.replace_with(' ')
    text = clean(copy.get_text(' ', strip=True))
    return re.sub(r'\s+([,.;:)\]?])', r'\1', re.sub(r'([(\[])\s+', r'\1', text))


def primary_name(cell):
    """Prefer the bold name, since the rest of the cell holds variants and epithets.

    A damaged name keeps the whole cell, because the bold part is only a fragment of it.
    """
    full = HORUS_PREFIX.sub('', plain_text(cell))
    bold = cell.find('b')
    if bold is None or '[...]' in full or '[…]' in full:
        return full
    return plain_text(bold) or full


def wikipedia_title(cell):
    for link in cell.find_all('a', href=True):
        match = re.search(r'/wiki/([^#?]+)', link['href'])
        if match and ':' not in match.group(1):
            return unquote(match.group(1)).replace('_', ' ')
    return None


def header_key(text):
    return clean(re.sub(r'\[.*', '', text)).lower()


def table_columns(table):
    """Map a ruler table's columns to fields, or return None for any other table."""
    first_row = table.find('tr')
    if first_row is None:
        return None
    columns, position = {}, 0
    for cell in first_row.find_all(['th', 'td']):
        columns[header_key(plain_text(cell))] = position
        position += int(cell.get('colspan', 1))
    if set(columns) == {'name'}:
        return columns, position
    if not set(columns) & set(NAME_COLUMNS) or not {'#', 'image'} & set(columns):
        return None
    return columns, position


def expand_rows(table, width):
    """Expand merged cells into a grid of rows, keeping each cell object in every slot."""
    grid, carried = [], {}
    for row in table.find_all('tr')[1:]:
        cells, slots, position = iter(row.find_all(['th', 'td'], recursive=False)), [], 0
        cell = next(cells, None)
        while position < width:
            if position in carried:
                source, remaining = carried[position]
                slots.append(source)
                if remaining > 1:
                    carried[position] = (source, remaining - 1)
                else:
                    del carried[position]
                position += 1
            elif cell is not None:
                span = int(cell.get('colspan', 1))
                down = int(cell.get('rowspan', 1))
                for offset in range(span):
                    slots.append(cell)
                    if down > 1:
                        carried[position + offset] = (cell, down - 1)
                position += span
                cell = next(cells, None)
            else:
                slots.append(None)
                position += 1
        grid.append(slots[:width])
    return grid


def field_cell(slots, index):
    """Return the cell in a column, unless it only arrived there by spanning sideways."""
    if index is None or index >= len(slots) or slots[index] is None:
        return None
    cell = slots[index]
    return None if any(slots[i] is cell for i in range(index)) else cell


def era_sign(era):
    return 1 if era and era.upper() in {'AD', 'CE'} else -1


def parse_reign(text):
    """Return dates from a reign cell. Reign lengths such as '18 years' are never dates."""
    result = {'start_year': None, 'end_year': None, 'circa': False, 'era_assumed': False,
              'alternate_dates': False}
    text = SPLIT_YEAR.sub(r'\1', CALENDAR_DAY.sub('', text or ''))
    ranges = list(YEAR_RANGE.finditer(text))
    for match in ranges:
        era = match.group('era') or match.group('start_era')
        start, end = match.group('start'), match.group('end')
        if len(end) < len(start) and len(end) < 3 and not era:
            continue
        if not era and len(start) < 3 and len(end) < 3:
            continue
        first = era_sign(match.group('start_era') or era) * int(start)
        last = era_sign(match.group('era') or era) * int(end)
        if last < first:
            continue
        result.update(start_year=first, end_year=last, circa=bool(match.group('circa')),
                      era_assumed=not era)
        result['alternate_dates'] = len(ranges) > 1
        return result
    single = SINGLE_YEAR.search(text)
    if single:
        year = era_sign(single.group('era')) * int(single.group('year'))
        result.update(start_year=year, end_year=year, circa=bool(single.group('circa')))
    return result


def reign_length(text):
    match = LENGTH.search(text or '')
    return clean(match.group(0)) if match else None


def ordinal(word):
    word = word.lower()
    if word in TEENS:
        return TEENS[word]
    if word in UNITS:
        return UNITS[word]
    tens, _, unit = word.partition('-')
    return TENS[tens] + UNITS[unit] if tens in TENS and unit in UNITS else None


def dynasty_numbers(heading):
    words = re.findall(r'[A-Za-z]+(?:-[a-z]+)?', heading or '')
    return [n for n in map(ordinal, words) if n]


def dynasty_heading(headings):
    for heading in reversed(headings):
        if re.search(r'\bdynast(?:y|ies)\b', heading, re.IGNORECASE):
            return re.sub(r'\s+cont\.?$', '', heading)
    return None


def role_for(columns, headings):
    if 'priest' in columns:
        return 'high_priest'
    if 'rebel' in ' '.join(headings).lower():
        return 'rebel'
    return 'pharaoh' if dynasty_heading(headings) else 'predynastic_ruler'


def period_range(heading):
    """Return the years in a heading such as 'New Kingdom (c. 1550–1069 BC)'."""
    match = PERIOD_RANGE.search(heading or '')
    dates = parse_reign(match.group('dates')) if match else {}
    if dates.get('start_year') is None:
        return None
    return {'start': dates['start_year'], 'end': dates['end_year']}


def name_status(name):
    if LOST_NAME.search(name):
        return 'lost'
    return 'partial' if '[...]' in name or '[…]' in name else 'named'


def banner_text(slots):
    """Return a full-width banner row's text, or None for an ordinary row."""
    cells = [cell for cell in slots if cell is not None]
    if len(slots) < 2 or not cells or any(cell is not cells[0] for cell in cells):
        return None
    return plain_text(cells[0])


def next_group(text, current):
    if not text or text.isdigit() or text.startswith('+'):
        return current
    return None if text.lower().startswith('end of') else text


def slug(text):
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def named_cell(slots, name_column, throne_column):
    """Return the cell naming a ruler. Some rulers are known only by their throne name."""
    for column in (name_column, throne_column):
        cell = field_cell(slots, column)
        name = primary_name(cell) if cell is not None else ''
        if name.lower() not in EMPTY_VALUES and not name.isdigit():
            return cell
    return None


def rows_in_table(table, columns, width, headings):
    """Yield one record per ruler row."""
    name_field = next(f for c, f in NAME_COLUMNS.items() if c in columns and f == 'horus_name') \
        if 'horus name' in columns and not {'personal name', 'name'} & set(columns) else 'personal_name'
    name_column = next(columns[c] for c in NAME_COLUMNS if c in columns)
    dynasty = dynasty_heading(headings)
    group = None
    for slots in expand_rows(table, width):
        banner = banner_text(slots)
        if banner is not None:
            group = next_group(banner, group)
            continue
        cell = named_cell(slots, name_column, columns.get('throne name'))
        if cell is None:
            continue
        horus_cell = None
        if name_field == 'horus_name' and wikipedia_title(cell) is None:
            throne_cell = field_cell(slots, columns.get('throne name'))
            if throne_cell is not None and wikipedia_title(throne_cell):
                horus_cell, cell = cell, throne_cell  # known as Khufu, not by Horus name Medjedu
        name = primary_name(cell)
        record = {'king_name': name, 'name_details': None, 'name_status': name_status(name),
                  'role': role_for(columns, headings), 'number': None, 'horus_name': None,
                  'personal_name': None, 'throne_name': None, 'greek_name': None}
        if field_cell(slots, name_column) is cell:
            record[name_field] = name
        elif horus_cell is not None:
            record['horus_name'] = primary_name(horus_cell)
        full = plain_text(cell)
        record['name_details'] = full if full != name else None
        for column, field in OTHER_COLUMNS.items():
            other = field_cell(slots, columns.get(column))
            value = plain_text(other) if other is not None else ''
            record[field] = None if value.lower() in EMPTY_VALUES else value
        if 'horus name' in columns and name_field != 'horus_name':
            other = field_cell(slots, columns['horus name'])
            record['horus_name'] = plain_text(other) if other is not None else None
        dates = parse_reign(record['reign_period'])
        interval = None
        if dates['start_year'] is not None:
            interval = {'start': dates['start_year'], 'end': dates['end_year']}
        record.update(reign_length=reign_length(record['reign_period']), dynasty=dynasty,
                      dynasty_numbers=dynasty_numbers(dynasty), period=headings[0] if headings else None,
                      period_years=period_range(headings[0]) if headings else None,
                      section=headings[-1] if headings else None, group_note=group,
                      time_interval=interval, **dates, wikipedia_title=wikipedia_title(cell))
        title = record['wikipedia_title']
        record['wikipedia_url'] = ('https://en.wikipedia.org/wiki/' + title.replace(' ', '_')
                                   if title else None)
        yield record


def extract_records(soup):
    root = soup.select_one('.mw-parser-output') or soup
    active, records = {}, []
    for node in root.find_all(['h2', 'h3', 'h4', 'h5', 'table']):
        if node.name != 'table':
            level = int(node.name[1])
            active = {old: title for old, title in active.items() if old < level}
            active[level] = re.sub(r'\s*\[\s*edit\s*\]\s*$', '', plain_text(node))
            continue
        if 'wikitable' not in node.get('class', []) or node.find_parent('table') is not None:
            continue
        found = table_columns(node)
        if found:
            headings = [active[level] for level in sorted(active)]
            records.extend(rows_in_table(node, *found, headings))
    return assign_ids(records)


def assign_ids(records):
    seen = Counter()
    for record in records:
        numbers = record['dynasty_numbers']
        base = slug(record['wikipedia_title'] or record['king_name']) or 'ruler'
        key = f"{base}-d{numbers[0]}" if numbers else base
        seen[key] += 1
        record['id'] = key if seen[key] == 1 else f'{key}-{seen[key]}'
    return records


def source_info(text, path):
    revision = re.search(r'wgRevisionId"?:\s*(\d+)', text)
    edited = re.search(r'This page was last edited on ([^<]*?),? at ', text)
    return {'url': PAGE_URL, 'file': Path(path).name,
            'revision_id': int(revision.group(1)) if revision else None,
            'last_edited': clean(edited.group(1)) if edited else None, 'license': LICENSE,
            'years': 'negative is BC, positive is AD, with no year zero'}


def summary(records):
    return {'rulers': len(records),
            'by_role': dict(Counter(r['role'] for r in records)),
            'by_name_status': dict(Counter(r['name_status'] for r in records)),
            'with_dates': sum(r['time_interval'] is not None for r in records),
            'with_period_years': sum(r['period_years'] is not None for r in records),
            'with_reign_length': sum(r['reign_length'] is not None for r in records),
            'with_wikipedia_link': sum(r['wikipedia_title'] is not None for r in records),
            'without_dynasty': sum(r['dynasty'] is None for r in records)}


def main():
    parser = argparse.ArgumentParser(description="Build a clean ruler list from a saved copy of Wikipedia's List of pharaohs.")
    parser.add_argument('--html', type=Path, default=DEFAULT_HTML)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    text = args.html.read_text(encoding='utf-8', errors='replace')
    records = extract_records(BeautifulSoup(text, 'lxml'))
    if not records:
        raise SystemExit('No ruler rows were found. Check that the saved page is complete.')
    payload = {'source': source_info(text, args.html), 'summary': summary(records),
               'rulers': records}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload['summary'], indent=2))
    print(f'Wrote {len(records)} rulers to {args.out}')


if __name__ == '__main__':
    main()
