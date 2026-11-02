"""Flatten the raw AAEW JSON export into CSV tables for the evidence store.

Usage:
    python -m evidence_store.extract                         # all corpora
    python -m evidence_store.extract --corpus smaek tb       # a subset
"""
import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from evidence_store.paths import BUILD_DIR, DATA_DIR

# Same pattern as Sphinx's clean_gardiner_seq, so sign sequences stay comparable.
SIGN_RE = re.compile(r'[A-Za-z]{1,3}[0-9]{1,3}[a-z]?')
SUPERFLUOUS_RE = re.compile(r'\{[^}]*\}')
EDITORIAL_RE = re.compile(r'[\[\]()⸢⸣〈〉?⸮!]')
LACUNA_MARKERS = {'destruction_marker', 'DestroyedVersMarker'}

# Passport paths whose values are thesaurus ids.
TERM_FIELDS = {
    'date': ('date', 'date', 'date'),
    'find_spot': ('find_spot', 'find_spot', 'place', 'place'),
    'script': ('text', 'textual_metadata', 'script'),
    'text_type': ('text', 'textual_metadata', 'texttype'),
    'object_type': ('object', 'description_of_object', 'type'),
    'material': ('object', 'technical_details', 'material'),
    'present_location': ('present_location', 'location', 'location'),
}
TEXT_FIELDS = {
    'egy_text_name': ('text', 'textual_metadata', 'egyTextName'),
    'bibliography': ('bibliography', 'bibliographical_text_field'),
}
PATH_TO_FIELD = {path: f for f, path in {**TERM_FIELDS, **TEXT_FIELDS}.items()}

COLUMNS = {
    'terms': ['id', 'type', 'name', 'parent_id'],
    'lemmas': ['id', 'name', 'pos', 'subtype', 'gloss_de', 'gloss_en', 'sign_seq',
               'revision_state'],
    'lemma_relations': ['lemma_id', 'relation', 'target_id'],
    'documents': ['id', 'corpus', 'kind', 'type', 'name', 'parent_ids', 'path', 'owner',
                  'revision_state', 'date_ids', *TERM_FIELDS, *TEXT_FIELDS],
    'sentences': ['id', 'text_id', 'position', 'transliteration', 'translit_norm', 'sign_seq',
                  'word_count', 'has_lacuna', 'translation_de', 'translation_en'],
    'tokens': ['sentence_id', 'position', 'alternate', 'word_id', 'w_char', 'w_norm', 'lemma_id',
               'flex_code', 'gloss_de', 'sign_seq'],
}


def eclass(record):
    return (record.get('eClass') or '').rsplit('//', 1)[-1]


def translations_by_lang(translations):
    by_lang = {}
    for t in (translations or {}).get('translations') or []:
        if t.get('value'):
            by_lang.setdefault(t.get('lang'), []).append(t['value'])
    return {lang: '; '.join(values) for lang, values in by_lang.items()}


def part_of(record):
    return [r['objectId'] for r in record.get('relations') or []
            if r.get('type') == 'partOf' and r.get('objectId')]


def normalize_translit(w_char):
    """Drop editorial marks and scribal surplus but keep restored text and suffix pronouns."""
    text = SUPERFLUOUS_RE.sub('', w_char or '')
    text = EDITORIAL_RE.sub('', text)
    return unicodedata.normalize('NFC', text).lower().strip()


def graphics_signs(graphics):
    graphics = graphics or []
    if all(g.get('innerSentenceOrder') is not None for g in graphics):
        graphics = sorted(graphics, key=lambda g: g['innerSentenceOrder'])
    return SIGN_RE.findall(' '.join(g.get('code') or '' for g in graphics))


def passport_values(passport):
    """Collect the wanted passport leaves as {field: [values]}."""
    found = {}

    def walk(node, path):
        for child in node.get('children') or []:
            child_path = (*path, child.get('type'))
            field = PATH_TO_FIELD.get(child_path)
            if field and child.get('value'):
                found.setdefault(field, []).append(child['value'])
            walk(child, child_path)

    walk(passport or {}, ())
    return found


def sentence_words(sentence):
    """Yield (word, alternate) in order; alternate is the case index inside an ambivalence."""
    for item in sentence.get('sentenceItems') or []:
        kind = eclass(item)
        if kind == 'BTSWord':
            yield item, None
        elif kind == 'BTSAmbivalence':
            for index, case in enumerate(item.get('cases') or []):
                for word in case.get('scenario') or []:
                    if eclass(word) == 'BTSWord':
                        yield word, index


def sentence_rows(sentence, text_id, position):
    """Return one sentence row plus its token rows. The main reading uses the first alternate."""
    tokens, translit, norm, signs = [], [], [], []
    for word, alternate in sentence_words(sentence):
        word_signs = graphics_signs(word.get('graphics'))
        w_char = word.get('wChar') or ''
        w_norm = normalize_translit(w_char)
        tokens.append({
            'sentence_id': sentence['_id'],
            'position': len(tokens),
            'alternate': alternate,
            'word_id': word.get('_id'),
            'w_char': w_char,
            'w_norm': w_norm,
            'lemma_id': word.get('lKey'),
            'flex_code': word.get('flexCode'),
            'gloss_de': translations_by_lang(word.get('translation')).get('de'),
            'sign_seq': ' '.join(word_signs),
        })
        if not alternate:
            translit.append(w_char)
            if w_norm:
                norm.append(w_norm)
            signs += word_signs
    translations = translations_by_lang(sentence.get('translation'))
    row = {
        'id': sentence['_id'],
        'text_id': text_id,
        'position': position,
        'transliteration': ' '.join(translit),
        'translit_norm': ' '.join(norm),
        'sign_seq': ' '.join(signs),
        'word_count': len(translit),
        'has_lacuna': any(i.get('type') in LACUNA_MARKERS
                          for i in sentence.get('sentenceItems') or []),
        'translation_de': translations.get('de'),
        'translation_en': translations.get('en'),
    }
    return row, tokens


def inherited(doc_id, field, docs):
    """Return a field's values from the document or its nearest ancestor that has them."""
    queue, seen = [doc_id], set()
    while queue:
        current = queue.pop(0)
        if current in seen or current not in docs:
            continue
        seen.add(current)
        values = docs[current]['meta'].get(field)
        if values:
            return values
        queue += docs[current]['parent_ids']
    return []


def ancestry_path(doc_id, docs):
    """Join names from the root down, following the first parent."""
    names, seen, current = [], set(), doc_id
    while current in docs and current not in seen:
        seen.add(current)
        if docs[current]['name']:
            names.append(docs[current]['name'])
        parents = docs[current]['parent_ids']
        current = parents[0] if parents else None
    return ' > '.join(reversed(names))


def document_row(doc_id, docs, terms):
    doc = docs[doc_id]
    row = {key: doc[key] for key in ('id', 'corpus', 'kind', 'type', 'name', 'owner',
                                     'revision_state')}
    row['parent_ids'] = json.dumps(doc['parent_ids'])
    row['path'] = ancestry_path(doc_id, docs)
    for field in TERM_FIELDS:
        labels = [terms[v]['name'] for v in inherited(doc_id, field, docs) if v in terms]
        row[field] = '; '.join(dict.fromkeys(labels))
    row['date_ids'] = json.dumps([v for v in inherited(doc_id, 'date', docs) if v in terms])
    for field in TEXT_FIELDS:
        row[field] = '; '.join(doc['meta'].get(field) or [])
    return row


def lemma_rows(entry):
    glosses = translations_by_lang(entry.get('translations'))
    words = entry.get('words') or [{}]
    row = {
        'id': entry['_id'],
        'name': entry.get('name'),
        'pos': entry.get('type'),
        'subtype': entry.get('subtype'),
        'gloss_de': glosses.get('de'),
        'gloss_en': glosses.get('en'),
        'sign_seq': ' '.join(graphics_signs(words[0].get('graphics'))),
        'revision_state': entry.get('revisionState'),
    }
    relations = [{'lemma_id': entry['_id'], 'relation': r['type'], 'target_id': r['objectId']}
                 for r in entry.get('relations') or [] if r.get('type') and r.get('objectId')]
    return row, relations


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def term_row(entry):
    parents = part_of(entry)
    return {'id': entry['_id'], 'type': entry.get('type'), 'name': entry.get('name'),
            'parent_id': parents[0] if parents else None}


class Writers:
    def __init__(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.handles, self.writers = {}, {}
        for table, columns in COLUMNS.items():
            self.handles[table] = open(out_dir / f'{table}.csv', 'w', encoding='utf-8', newline='')
            self.writers[table] = csv.DictWriter(self.handles[table], columns)
            self.writers[table].writeheader()
        self.counts = dict.fromkeys(COLUMNS, 0)

    def write(self, table, rows):
        self.writers[table].writerows(rows)
        self.counts[table] += len(rows)

    def close(self):
        for handle in self.handles.values():
            handle.close()


def build(corpora, out_dir, data_dir=DATA_DIR):
    writers = Writers(out_dir)
    term_entries = load_json(data_dir / 'vocabulary' / 'aaew_ths.json')
    terms = {e['_id']: e for e in term_entries}
    writers.write('terms', [term_row(e) for e in term_entries])

    for entry in load_json(data_dir / 'vocabulary' / 'aaew_wlist.json'):
        row, relations = lemma_rows(entry)
        writers.write('lemmas', [row])
        writers.write('lemma_relations', relations)

    docs = {}
    for corpus in corpora:
        for record in load_json(data_dir / 'corpus' / f'aaew_corpus_{corpus}.json'):
            kind = {'BTSText': 'text', 'BTSTCObject': 'object'}.get(eclass(record))
            if kind is None:
                continue
            docs[record['_id']] = {
                'id': record['_id'], 'corpus': corpus, 'kind': kind, 'type': record.get('type'),
                'name': record.get('name'), 'owner': record.get('owner'),
                'revision_state': record.get('revisionState'), 'parent_ids': part_of(record),
                'meta': passport_values(record.get('passport')),
            }
            items = (record.get('textContent') or {}).get('textItems') or []
            for position, sentence in enumerate(items):
                row, tokens = sentence_rows(sentence, record['_id'], position)
                writers.write('sentences', [row])
                writers.write('tokens', tokens)

    writers.write('documents', [document_row(doc_id, docs, terms) for doc_id in docs])
    writers.close()
    return writers.counts


def all_corpora(data_dir=DATA_DIR):
    return sorted(p.stem.removeprefix('aaew_corpus_') for p in (data_dir / 'corpus').glob('*.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--corpus', nargs='*', help='corpus names; defaults to all of them')
    parser.add_argument('--out', type=Path, default=BUILD_DIR)
    args = parser.parse_args()
    counts = build(args.corpus or all_corpora(), args.out)
    for table, count in counts.items():
        print(f'{table:16s} {count:>9,}')


if __name__ == '__main__':
    main()
