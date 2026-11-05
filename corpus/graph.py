import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection
from typing import Any

from psycopg.types.json import Jsonb

from corpus import retrieve
from corpus.paths import BUILD_DIR, KNOWLEDGE_DIR, ROYAL_NAMES_FILE

GODS_FILE = KNOWLEDGE_DIR / 'Ancient_Gods.json'
RULERS_FILE = KNOWLEDGE_DIR / 'master_pharaohs_v2.json'
TEMPLES_FILE = KNOWLEDGE_DIR / 'Temples.json'
PYRAMIDS_FILE = KNOWLEDGE_DIR / 'Pyramids.json'
GODS_SOURCE = 'Ancient_Gods.json'
RULERS_SOURCE = 'Wikipedia, List of pharaohs, revision 1383160761'
ROYAL_SOURCE = 'Sphinx royal_names.json'
TEMPLES_SOURCE = 'Temples.json'
PYRAMIDS_SOURCE = 'Pyramids.json'
REIGN_MARGIN = 60  # years a temple's dates may sit outside a named builder's reign
AAEW_SOURCE = 'AAEW lemma list'
SYMMETRIC = {'sibling_of', 'consort_of', 'adversary_of', 'merged_with', 'associated_with'}

# Reviewed by hand: the file's wording doesn't say which side it describes, so 'Father' on both
# Shu -> Geb and Geb -> Shu means Shu is the father. Each pair lists every link it supports.
FAMILY = {
    'parent_of': [('Ra', 'Shu'), ('Ra', 'Tefnut'), ('Ra', 'Hathor'), ('Ra', 'Bastet'),
                  ('Ra', 'Sekhmet'), ('Ra', 'Maat'), ('Atum', 'Shu'), ('Atum', 'Tefnut'),
                  ('Shu', 'Geb'), ('Shu', 'Nut'), ('Tefnut', 'Geb'), ('Tefnut', 'Nut'),
                  ('Geb', 'Osiris'), ('Geb', 'Isis'), ('Geb', 'Set'), ('Geb', 'Nephthys'),
                  ('Nut', 'Osiris'), ('Nut', 'Isis'), ('Nut', 'Set'), ('Nut', 'Nephthys'),
                  ('Osiris', 'Horus'), ('Isis', 'Horus'), ('Osiris', 'Anubis'),
                  ('Nephthys', 'Anubis'), ('Amun', 'Khonsu'), ('Mut', 'Khonsu'),
                  ('Ptah', 'Nefertem'), ('Sekhmet', 'Nefertem'), ('Neith', 'Sobek'),
                  ('Khnum', 'Anuket')],
    'sibling_of': [('Osiris', 'Isis'), ('Osiris', 'Set'), ('Osiris', 'Nephthys'),
                   ('Isis', 'Set'), ('Isis', 'Nephthys'), ('Set', 'Nephthys'), ('Geb', 'Nut'),
                   ('Shu', 'Tefnut')],
    'consort_of': [('Osiris', 'Isis'), ('Set', 'Nephthys'), ('Geb', 'Nut'), ('Shu', 'Tefnut'),
                   ('Amun', 'Mut'), ('Ptah', 'Sekhmet'), ('Khnum', 'Satet'), ('Hathor', 'Horus'),
                   ('Thoth', 'Maat')],
    'uncle_of': [('Set', 'Horus')],
    'adversary_of': [('Osiris', 'Set'), ('Horus', 'Set')],
    'merged_with': [('Atum', 'Ra'), ('Sobek', 'Ra')],
    'associated_with': [('Thoth', 'Ra'), ('Thoth', 'Maat'), ('Bastet', 'Sekhmet'), ('Neith', 'Ra')],
}

# AAEW gods_name lemmas, matched by hand and checked against the lemma list when building.
GOD_LEMMAS = {'Ra': ['400015'], 'Amun': ['26060'], 'Osiris': ['49460'], 'Isis': ['271'],
              'Horus': ['107500'], 'Set': ['148520', '600256'], 'Anubis': ['27360'],
              'Thoth': ['185290'], 'Hathor': ['99960'], 'Bastet': ['53640'], 'Sekhmet': ['142250'],
              'Maat': ['66630'], 'Ptah': ['62980'], 'Nut': ['80940'], 'Geb': ['167010'],
              'Nephthys': ['82260'], 'Khnum': ['123590'], 'Sobek': ['132180'], 'Neith': ['79010'],
              'Shu': ['152710'], 'Tefnut': ['171880'], 'Khonsu': ['118720'],
              'Mut': ['69030', '69020'], 'Atum': ['33040', '171970']}

# Gods named inside royal names, read from Sphinx's underscore-separated keys.
NAME_GODS = {'ra': 'Ra', 'jmn': 'Amun', 'dhwtj': 'Thoth', 'ptH': 'Ptah', 'stj': 'Set', 'hr': 'Horus'}
NAME_GOD_WHOLE = {'xafra': 'Ra', 'mnkawra': 'Ra'}

# German names in the AAEW, and spellings in Sphinx, mapped to the Wikipedia list's spelling.
NAME_SPELLINGS = {'amenemhet': 'amenemhat', 'sesostris': 'senusret', 'amenophis': 'amenhotep',
                  'thutmosis': 'thutmose', 'ramses': 'ramesses', 'ptolemaios': 'ptolemy',
                  'kleopatra': 'cleopatra', 'chefren': 'khafre', 'cheops': 'khufu',
                  'mykerinos': 'menkaure', 'snofru': 'sneferu', 'merenptah': 'merneptah',
                  'sethos': 'seti', 'antef': 'intef', 'echnaton': 'akhenaten',
                  'tutanchamun': 'tutankhamun', 'haremhab': 'horemheb', 'psammetich': 'psamtik',
                  'nektanebos': 'nectanebo', 'scheschonq': 'shoshenq', 'schabaka': 'shabaka',
                  'sobeknofru': 'sobekneferu', 'hatschepsut': 'hatshepsut', 'pije': 'piye',
                  'sebekhotep': 'sobekhotep', 'tanutamon': 'tantamani',
                  'tamphthis': 'thamphthis', 'schepseskaf': 'shepseskaf', 'asosi': 'isesi',
                  'sanacht': 'sanakht', 'sechemchet': 'sekhemkhet', 'niuserre': 'nyuserre'}

# Bare names that many rulers carry, so one lemma can't be pinned to one of them.
SHARED_KING_NAMES = {'cleopatra', 'intef', 'inyotef', 'psammetichus', 'ptolemy'}

# Builders by pyramid, matched by hand. Meidum's first builder is only an attribution.
PYRAMID_BUILDERS = {
    'Step Pyramid of Djoser': ['djoser-d3'], 'Meidum Pyramid': ['qahedjet-d3', 'sneferu-d4'],
    'Bent Pyramid': ['sneferu-d4'], 'Red Pyramid (North Pyramid)': ['sneferu-d4'],
    'Great Pyramid of Giza (Pyramid of Khufu)': ['khufu-d4'],
    'Pyramid of Djedefre': ['djedefre-d4'], 'Pyramid of Khafre': ['khafre-d4'],
    'Pyramid of Menkaure': ['menkaure-d4'], 'Pyramid of Userkaf': ['userkaf-d5'],
    'Pyramid of Sahure': ['sahure-d5'], 'Pyramid of Neferirkare Kakai': ['neferirkare-kakai-d5'],
    'Pyramid of Nyuserre Ini': ['nyuserre-ini-d5'], 'Pyramid of Unas': ['unas-d5'],
    'Pyramid of Teti': ['teti-d6'], 'Pyramid of Pepi II': ['pepi-ii-neferkare-d6'],
    'Pyramid of Senusret I': ['senusret-i-d12'], 'Pyramid of Senusret II': ['senusret-ii-d12'],
    'Black Pyramid (Pyramid of Amenemhat III)': ['amenemhat-iii-d12'],
    'Pyramid of Hawara': ['amenemhat-iii-d12'], 'Pyramid of Ahmose I': ['ahmose-i-d18']}

# Links the pyramid file states, by pyramid name.
PYRAMID_TEMPLES = {
    'Pyramid of Khafre': ('connected_to', 'temple-valley-temple-of-khafre'),
    'Pyramid of Nyuserre Ini': ('near', 'temple-sun-temple-of-nyuserre-shesepibre')}

# Corpus find spots by site word in the pyramid's location. These cover whole sites.
PYRAMID_SITES = {'Saqqara': 'Saqqara', 'Giza Plateau': 'Gisa', 'Abusir': 'Abusir',
                 'Abydos': 'Abydos', 'El-Lahun': 'el-Lahun (Illahun)', 'El-Lisht': 'Lischt',
                 'Dahshur': 'Dahschur'}

# Documents carved inside a pyramid, unlike find spots, are that pyramid's own texts.
# These brackets are U+2329 and U+232A, as in documents.path, not the look-alike U+3008 and U+3009.
PYRAMID_TEXTS = {'Pyramid of Unas': '〈Pyramidentexte〉 > Unas-Pyramide',
                 'Pyramid of Teti': '〈Pyramidentexte〉 > Pyramide des Teti',
                 'Pyramid of Pepi II': '〈Pyramidentexte〉 > Pyramide Pepis II.'}

ROMAN = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9,
         'X': 10, 'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15, 'XVI': 16, 'XVII': 17,
         'XVIII': 18, 'XIX': 19, 'XX': 20, 'XXI': 21, 'XXII': 22, 'XXIII': 23, 'XXIV': 24,
         'XXV': 25, 'XXVI': 26, 'XXVII': 27, 'XXVIII': 28, 'XXIX': 29, 'XXX': 30}


def fold(text: str | None) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', (text or '').lower()).strip()


def english_form(text: str | None) -> str:
    """Translate a folded German royal name into the Wikipedia list's spelling."""
    return ' '.join(NAME_SPELLINGS.get(word, word) for word in fold(text).split())


def slug(text: str) -> str:
    return fold(text).replace(' ', '-')


def link(source_id: str,
         relation: str,
         target_id: str,
         detail: str | None,
         source: str,
         reliability: str) -> dict[str, Any]:
    if relation in SYMMETRIC and target_id < source_id:
        source_id, target_id = target_id, source_id
    return {'source_id': source_id, 'relation': relation, 'target_id': target_id,
            'detail': detail, 'source': source, 'reliability': reliability}


def entity(id: str,
           type: str,
           name: str,
           source: str,
           reliability: str,
           **fields: Any) -> dict[str, Any]:
    return {'id': id, 'type': type, 'name': name, 'aliases': fields.get('aliases', []),
            'start_year': fields.get('start_year'), 'end_year': fields.get('end_year'),
            'lemma_ids': fields.get('lemma_ids', []), 'date_terms': fields.get('date_terms', []),
            'attributes': fields.get('attributes', {}), 'source': source,
            'reliability': reliability}


def family_index() -> dict[frozenset[str], list[tuple[str, str, str]]]:
    """Return {unordered pair: [(relation, first, second)]} from the reviewed list."""
    index: defaultdict[frozenset[str], list[tuple[str, str, str]]] = defaultdict(list)
    for relation, pairs in FAMILY.items():
        for first, second in pairs:
            index[frozenset((first, second))].append((relation, first, second))
    return index


def god_id(name: str) -> str:
    return 'god-' + slug(name)


def build_gods(deities: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],
                                                       list[dict[str, Any]]]:
    """Return god entities and links. An unreviewed relationship stops the build."""
    index = family_index()
    known = {deity['name'] for deity in deities}
    wordings: defaultdict[frozenset[str], list[str]] = defaultdict(list)
    for deity in deities:
        for item in deity['relationships']:
            pair = frozenset((deity['name'], item['deity']))
            if pair not in index:
                raise ValueError(f"unreviewed relationship: {deity['name']} - {item['deity']}")
            wordings[pair].append(f"{deity['name']} to {item['deity']}: {item['relationship']}")
    entities = [entity(god_id(d['name']), 'god', d['name'], GODS_SOURCE, 'curated',
                       lemma_ids=GOD_LEMMAS.get(d['name'], []),
                       attributes={key: d[key] for key in ('description', 'notes', 'powers', 'duty',
                                                           'strength', 'symbols')})
                for d in deities]
    stubs = sorted({name for pair in wordings for name in pair} - known)
    entities += [entity(god_id(name), 'god', name, GODS_SOURCE, 'curated',
                        attributes={'stub': True}) for name in stubs]
    links: list[dict[str, Any]] = []
    for pair, wording in wordings.items():
        detail = '; '.join(wording)
        links += [link(god_id(first), relation, god_id(second), detail, GODS_SOURCE, 'curated')
                  for relation, first, second in index[pair]]
    return entities, links


def dynasty_id(ruler: dict[str, Any]) -> str:
    numbers = ruler['dynasty_numbers']
    return 'dynasty-' + ('-'.join(map(str, numbers)) if numbers else slug(ruler['dynasty']))


def build_rulers(rulers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]],
                                                        list[dict[str, Any]]]:
    """Return ruler and dynasty entities, skipping rows whose name is lost."""
    entities: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    members: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for ruler in rulers:
        if ruler['name_status'] == 'lost':
            continue
        names = (ruler[key] for key in ('personal_name', 'throne_name', 'horus_name', 'greek_name',
                                        'wikipedia_title'))
        aliases = sorted({name for name in names if name and name != ruler['king_name']})
        attributes = {key: ruler[key] for key in (
            'role', 'name_status', 'name_details', 'throne_name', 'notes', 'reign_period',
            'reign_length', 'circa', 'alternate_dates', 'group_note', 'period', 'period_years',
            'wikipedia_url')}
        entities.append(entity(ruler['id'], 'pharaoh', ruler['king_name'], RULERS_SOURCE, 'reference',
                               aliases=aliases, start_year=ruler['start_year'],
                               end_year=ruler['end_year'], attributes=attributes))
        if ruler['dynasty']:
            members[dynasty_id(ruler)].append(ruler)
            links.append(link(ruler['id'], 'ruled_in', dynasty_id(ruler), ruler['dynasty'],
                              RULERS_SOURCE, 'reference'))
    for key, group in members.items():
        first = group[0]
        starts = [r['start_year'] for r in group if r['start_year'] is not None]
        ends = [r['end_year'] for r in group if r['end_year'] is not None]
        entities.append(entity(
            key, 'dynasty', first['dynasty'], RULERS_SOURCE, 'reference',
            start_year=min(starts, default=None), end_year=max(ends, default=None),
            date_terms=[f'{number}. Dyn.' for number in first['dynasty_numbers']],
            attributes={'numbers': first['dynasty_numbers'], 'period': first['period'],
                        'period_years': first['period_years'], 'rulers': len(group)}))
    return entities, links


def ruler_index(rulers: list[dict[str, Any]],
                keys: tuple[str, ...] = ('king_name', 'wikipedia_title', 'greek_name'),
                ) -> dict[str, set[str]]:
    """Return {folded name: set of ruler ids} over the given name fields."""
    index: defaultdict[str, set[str]] = defaultdict(set)
    for ruler in rulers:
        if ruler['name_status'] == 'lost':
            continue
        for key in keys:
            if ruler[key]:
                index[fold(ruler[key])].add(ruler['id'])
    return index


def rulers_named(names: Collection[str],
                 main: dict[str, set[str]],
                 throne: dict[str, set[str]]) -> set[str]:
    """A ruler's main name beats another ruler's throne name, so Teti means the Sixth Dynasty king."""
    found: set[str] = set().union(*(main.get(name, set()) for name in names))
    return found or set().union(*(throne.get(name, set()) for name in names))


def match_royal_name(english: str, rulers: list[dict[str, Any]]) -> str | None:
    """Return the one ruler an English royal name means, or None if unclear."""
    dynasty = re.search(r'Dynasty ([IVX]+)', english)
    number = ROMAN.get(dynasty.group(1)) if dynasty else None
    for name in re.sub(r'\(.*', '', english).split('/'):
        pattern = rf'(?:^| ){re.escape(english_form(name))}(?: |$)'
        found = {r['id'] for r in rulers
                 if r['name_status'] != 'lost' and r['role'] == 'pharaoh'
                 and (number is None or number in r['dynasty_numbers'])
                 and any(re.search(pattern, fold(r[key]))
                         for key in ('king_name', 'wikipedia_title', 'greek_name') if r[key])}
        if len(found) == 1:
            return found.pop()
    return None


def royal_reliability(item: dict[str, Any]) -> str:
    """Trust 'verified' only when the source doesn't call the entry generated or unchecked."""
    doubtful = re.search(r'claude-generated|validate', item['source'], re.IGNORECASE)
    return 'verified' if item['verified'] and not doubtful else 'unverified'


def gods_in_name(key: str) -> list[str]:
    if key in NAME_GOD_WHOLE:
        return [NAME_GOD_WHOLE[key]]
    return [NAME_GODS[part] for part in key.split('_') if part in NAME_GODS]


def build_royal_names(names: dict[str, dict[str, Any]],
                      rulers: list[dict[str, Any]],
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entities: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for key, item in names.items():
        rating = royal_reliability(item)
        node = f'royal-{slug(key)}'
        entities.append(entity(
            node, 'royal_name', item['english'], ROYAL_SOURCE, rating,
            attributes={'translit': item['translit'], 'spellings': item['spellings'],
                        'freq': item['freq'], 'file_says_verified': item['verified'],
                        'provenance': item['source']}))
        ruler = match_royal_name(item['english'], rulers)
        if ruler:
            links.append(link(node, 'names', ruler, item['english'], ROYAL_SOURCE, rating))
        else:
            unmatched.append(item['english'])
        links += [link(node, 'invokes', god_id(god), f"{god} is part of {key}", ROYAL_SOURCE,
                       'derived') for god in gods_in_name(key)]
    return entities, links, unmatched


def lemma_rows() -> list[dict[str, str]]:
    csv.field_size_limit(10 ** 7)
    with open(BUILD_DIR / 'lemmas.csv', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def attach_god_lemmas(entities: list[dict[str, Any]], lemmas: list[dict[str, str]]) -> None:
    """Stop the build if a hand-matched lemma isn't a god name in the list."""
    god_ids = {row['id'] for row in lemmas if row['subtype'] == 'gods_name'}
    for item in entities:
        missing = set(item['lemma_ids']) - god_ids if item['type'] == 'god' else set()
        if missing:
            raise ValueError(f"{item['name']}: not gods_name lemmas: {sorted(missing)}")


def king_lemma_names(row: dict[str, str]) -> set[str]:
    """Return folded names a kings_name lemma may mean, or none if shared. A named owner wins."""
    english, german = row['gloss_en'] or '', row['gloss_de'] or ''
    if ';' in german or 'several' in english:
        return set()
    owner = re.search(r'(?:Thronname|Horusname|Eigenname)\s+([^\](;]+)', german)
    if owner and ',' in owner.group(1):
        return set()
    if owner:
        words = fold(owner.group(1)).split()
        genitive = [words[0].removesuffix('s'), *words[1:]] if words else []
        return {english_form(' '.join(words)), english_form(' '.join(genitive))} - {''}
    names = {fold(re.split(r'[(;,]', english)[0]), english_form(re.split(r'[(;,\]]', german)[0])}
    return names - SHARED_KING_NAMES - {''}


def attach_king_lemmas(entities: list[dict[str, Any]],
                       lemmas: list[dict[str, str]],
                       rulers: list[dict[str, Any]]) -> int:
    """Link a king's lemma only when its name points at exactly one ruler."""
    main, throne = ruler_index(rulers), ruler_index(rulers, ('throne_name',))
    by_id = {item['id']: item for item in entities if item['type'] == 'pharaoh'}
    linked = 0
    for row in lemmas:
        if row['subtype'] != 'kings_name':
            continue
        found = rulers_named(king_lemma_names(row), main, throne)
        if len(found) == 1 and found <= by_id.keys():
            by_id[found.pop()]['lemma_ids'].append(row['id'])
            linked += 1
    return linked


def attach_date_terms(entities: list[dict[str, Any]],
                      rulers: list[dict[str, Any]]) -> tuple[int, list[str]]:
    """Link corpus date terms such as 'Pepi II.' to the one ruler they name."""
    main, throne = ruler_index(rulers), ruler_index(rulers, ('throne_name',))
    by_id = {item['id']: item for item in entities if item['type'] == 'pharaoh'}
    periods = [row['period'] for row in retrieve.query('SELECT period FROM date_periods()')]
    linked = 0
    unmatched: list[str] = []
    for term in periods:
        found: set[str] = set()
        for part in term.split('/'):
            words = english_form(re.sub(r'\(.*', '', part)).split()
            found |= (rulers_named([' '.join(words)], main, throne)
                      or rulers_named([' '.join(words[:2])], main, throne))
        if len(found) == 1 and found <= by_id.keys():
            by_id[found.pop()]['date_terms'].append(term)
            linked += 1
        elif not re.search(r'Dyn|Jh|Reich|Zeit|Viertel|Hälfte|Drittel', term):
            unmatched.append(term)
    return linked, unmatched


def build_temples(temples: list[dict[str, Any]],
                  rulers_by_id: dict[str, dict[str, Any]],
                  dynasties: set[str],
                  god_names: set[str],
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return temple entities, new god stubs and links. Audited ids must already exist."""
    entities: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    stubs: set[str] = set()
    mismatches: list[str] = []
    for temple in temples:
        node = 'temple-' + slug(temple['Name'])
        rating = 'reference' if temple['Added_by_audit'] else 'curated'
        entities.append(entity(
            node, 'temple', temple['Name'], TEMPLES_SOURCE, rating, aliases=temple['Aliases'],
            start_year=temple['Start_year'], end_year=temple['End_year'],
            attributes={key.lower(): temple[key] for key in (
                'Location', 'Build_year', 'Build_by', 'Other_builders', 'Description', 'age',
                'Offerings', 'Offerings_status', 'Corpus_find_spots',
                'Find_spot_covers_whole_site', 'Added_by_audit', 'Audit_notes', 'Sources')}))
        for builder in temple['Builders']:
            if builder not in rulers_by_id and builder not in dynasties:
                raise ValueError(f"{temple['Name']}: unknown builder {builder}")
            links.append(link(node, 'made_by', builder, temple['Build_by'], TEMPLES_SOURCE, rating))
            ruler = rulers_by_id.get(builder)
            if ruler and not reign_overlaps(temple, ruler):
                mismatches.append(f"{temple['Name']}: {ruler['name']}")
        for god in temple['Gods']:
            if god not in god_names:
                stubs.add(god)
            links.append(link(node, 'dedicated_to', god_id(god), temple['Description'][:200],
                              TEMPLES_SOURCE, rating))
    entities += [entity(god_id(name), 'god', name, TEMPLES_SOURCE, 'curated',
                        attributes={'stub': True}) for name in sorted(stubs)]
    return entities, links, mismatches


def reign_overlaps(temple: dict[str, Any], ruler: dict[str, Any]) -> bool:
    """Compare a temple's dates with a builder's reign, when both are known."""
    if None in (temple['Start_year'], temple['End_year'], ruler['start_year'], ruler['end_year']):
        return True
    return (temple['Start_year'] - REIGN_MARGIN <= ruler['end_year']
            and ruler['start_year'] <= temple['End_year'] + REIGN_MARGIN)


def pyramid_years(text: str) -> tuple[int | None, int | None]:
    """Return (start, end) from a year field such as 'c. 2670 BCE – 2650 BCE'."""
    years = [-int(year) for year in re.findall(r'(\d{3,4}) BCE', text)]
    return (years[0], years[-1]) if years else (None, None)


def pyramid_aliases(name: str) -> list[str]:
    outer, _, inner = name.partition(' (')
    return [outer, inner.rstrip(')')] if inner else []


def build_pyramids(pyramids: list[dict[str, Any]],
                   rulers_by_id: dict[str, dict[str, Any]],
                   temple_ids: set[str],
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return pyramid entities and links. Every pyramid needs reviewed builders."""
    entities: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for pyramid in pyramids:
        name = pyramid['Name']
        if name not in PYRAMID_BUILDERS:
            raise ValueError(f'unreviewed pyramid: {name}')
        node = 'pyramid-' + slug(name)
        start, end = pyramid_years(pyramid['Year'])
        spots = [spot for word, spot in PYRAMID_SITES.items() if word in pyramid['Location']]
        entities.append(entity(
            node, 'pyramid', name, PYRAMIDS_SOURCE, 'curated', aliases=pyramid_aliases(name),
            start_year=start, end_year=end,
            attributes={'location': pyramid['Location'], 'year': pyramid['Year'],
                        'pharaoh': pyramid['Pharaoh'], 'dynasty': pyramid['Dynasty'],
                        'description': pyramid['Description'], 'corpus_find_spots': spots,
                        'find_spot_covers_whole_site': True,
                        'corpus_path': PYRAMID_TEXTS.get(name)}))
        for builder in PYRAMID_BUILDERS[name]:
            if builder not in rulers_by_id:
                raise ValueError(f'{name}: unknown builder {builder}')
            links.append(link(node, 'made_by', builder, pyramid['Pharaoh'], PYRAMIDS_SOURCE,
                              'curated'))
            span = {'Start_year': start, 'End_year': end}
            if not reign_overlaps(span, rulers_by_id[builder]):
                mismatches.append(f"{name}: {rulers_by_id[builder]['name']}")
        if name in PYRAMID_TEMPLES:
            relation, temple = PYRAMID_TEMPLES[name]
            if temple not in temple_ids:
                raise ValueError(f'{name}: unknown temple {temple}')
            links.append(link(node, relation, temple, pyramid['Description'][:200],
                              PYRAMIDS_SOURCE, 'curated'))
    return entities, links, mismatches


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gods = json.loads(GODS_FILE.read_text(encoding='utf-8'))['deities']
    rulers = json.loads(RULERS_FILE.read_text(encoding='utf-8'))['rulers']
    names = json.loads(ROYAL_NAMES_FILE.read_text(encoding='utf-8'))['names']
    entities, links = build_gods(gods)
    ruler_entities, ruler_links = build_rulers(rulers)
    royal_entities, royal_links, royal_unmatched = build_royal_names(names, rulers)
    entities += ruler_entities + royal_entities
    links += ruler_links + royal_links
    temples = json.loads(TEMPLES_FILE.read_text(encoding='utf-8'))
    temple_entities, temple_links, reign_mismatches = build_temples(
        temples, {e['id']: e for e in ruler_entities},
        {e['id'] for e in ruler_entities if e['type'] == 'dynasty'},
        {e['name'] for e in entities if e['type'] == 'god'})
    entities += temple_entities
    links += temple_links
    pyramids = json.loads(PYRAMIDS_FILE.read_text(encoding='utf-8'))
    pyramid_entities, pyramid_links, pyramid_mismatches = build_pyramids(
        pyramids, {e['id']: e for e in ruler_entities}, {e['id'] for e in temple_entities})
    entities += pyramid_entities
    links += pyramid_links
    lemmas = lemma_rows()
    attach_god_lemmas(entities, lemmas)
    king_lemmas = attach_king_lemmas(entities, lemmas, rulers)
    date_links, date_unmatched = attach_date_terms(entities, rulers)
    report = {'entities': dict(Counter(e['type'] for e in entities)),
              'links': dict(Counter(item['relation'] for item in links)),
              'reliability': dict(Counter(e['reliability'] for e in entities)),
              'royal_names_unmatched': royal_unmatched, 'king_lemmas_linked': king_lemmas,
              'temple_reign_mismatches': reign_mismatches,
              'pyramid_reign_mismatches': pyramid_mismatches,
              'date_terms_linked': date_links, 'date_terms_unmatched': date_unmatched}
    return entities, links, report


def store(entities: list[dict[str, Any]], links: list[dict[str, Any]]) -> None:
    columns = ('id', 'type', 'name', 'aliases', 'start_year', 'end_year', 'lemma_ids', 'date_terms',
               'attributes', 'source', 'reliability')
    link_columns = ('source_id', 'relation', 'target_id', 'detail', 'source', 'reliability')
    connection = retrieve.connection()
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute('TRUNCATE entity_links, entities')
        cursor.executemany(
            f"INSERT INTO entities ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})",
            [tuple(Jsonb(e[c]) if c == 'attributes' else e[c] for c in columns) for e in entities])
        cursor.executemany(
            f"INSERT INTO entity_links ({', '.join(link_columns)}) "
            f"VALUES ({', '.join(['%s'] * len(link_columns))})",
            [tuple(item[c] for c in link_columns) for item in links])


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the entities and entity_links tables from the knowledge files.')
    parser.add_argument('--report-only', action='store_true', help="don't write to the database")
    args = parser.parse_args()
    entities, links, report = build()
    if not args.report_only:
        store(entities, links)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
