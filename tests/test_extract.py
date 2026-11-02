from evidence_store import extract as ex

WORD = 'http://btsCorpusModel/1.0#//BTSWord'
MARKER = 'http://btsCorpusModel/1.0#//BTSMarker'
AMBIVALENCE = 'http://btsCorpusModel/1.0#//BTSAmbivalence'


def word(w_char, lemma='1', codes=None):
    record = {'eClass': WORD, '_id': f'w-{w_char}', 'wChar': w_char, 'lKey': lemma, 'flexCode': '3',
              'translation': {'translations': [{'lang': 'de', 'value': 'Wort'}]}}
    if codes:
        record['graphics'] = [{'code': c, 'innerSentenceOrder': i} for i, c in codes]
    return record


def sentence(*items):
    return {'_id': 's1', 'sentenceItems': list(items),
            'translation': {'translations': [{'lang': 'de', 'value': 'Der Satz'}]}}


def test_normalize_keeps_suffix_pronouns_and_restorations():
    assert ex.normalize_translit('=sn') == '=sn'
    assert ex.normalize_translit('=f') == '=f'
    assert ex.normalize_translit('[tp,j].pl') == 'tp,j.pl'
    assert ex.normalize_translit('swꜣ.t(j)') == 'swꜣ.tj'
    assert ex.normalize_translit('⸢nḫb,y⸣') == 'nḫb,y'
    assert ex.normalize_translit('〈r〉') == 'r'


def test_normalize_drops_surplus_and_doubt_marks():
    assert ex.normalize_translit('nfr{w}') == 'nfr'
    assert ex.normalize_translit('nfr?') == 'nfr'
    assert ex.normalize_translit('Stj,t') == 'stj,t'


def test_graphics_extracts_codes_in_order():
    graphics = [{'code': '-T21', 'innerSentenceOrder': 2}, {'code': '<-N5', 'innerSentenceOrder': 1}]
    assert ex.graphics_signs(graphics) == ['N5', 'T21']


def test_graphics_keeps_list_order_when_order_is_missing():
    assert ex.graphics_signs([{'code': 'G1'}, {'code': '-Aa1'}]) == ['G1', 'Aa1']
    assert ex.graphics_signs(None) == []


def test_sentence_skips_markers_and_flags_lacunae():
    row, tokens = ex.sentence_rows(
        sentence(word('nfr', codes=[(1, 'F35')]),
                 {'eClass': MARKER, '_id': 'm', 'type': 'destruction_marker'},
                 word('=f', lemma='2')),
        'text1', 4)
    assert row['transliteration'] == 'nfr =f'
    assert row['translit_norm'] == 'nfr =f'
    assert row['sign_seq'] == 'F35'
    assert row['word_count'] == 2
    assert row['position'] == 4
    assert row['has_lacuna'] is True
    assert row['translation_de'] == 'Der Satz'
    assert [t['lemma_id'] for t in tokens] == ['1', '2']


def test_ambivalence_keeps_all_tokens_but_reads_the_first_case():
    ambivalence = {'eClass': AMBIVALENCE, 'cases': [
        {'scenario': [word('a', codes=[(1, 'G1')])]},
        {'scenario': [word('b', codes=[(1, 'D58')]), {'eClass': MARKER}]},
    ]}
    row, tokens = ex.sentence_rows(sentence(word('x'), ambivalence), 't', 0)
    assert row['transliteration'] == 'x a'
    assert row['sign_seq'] == 'G1'
    assert [(t['w_char'], t['alternate']) for t in tokens] == [('x', None), ('a', 0), ('b', 1)]
    assert [t['position'] for t in tokens] == [0, 1, 2]


def test_empty_normalized_words_leave_no_gaps():
    row, _ = ex.sentence_rows(sentence(word('nfr'), word('?'), word('pn')), 't', 0)
    assert row['translit_norm'] == 'nfr pn'


def docs_fixture():
    return {
        'root': {'meta': {'date': ['d1']}, 'parent_ids': [], 'name': 'Tomb'},
        'mid': {'meta': {}, 'parent_ids': ['root'], 'name': 'Wall'},
        'leaf': {'meta': {}, 'parent_ids': ['mid', 'missing'], 'name': 'Text'},
        'own': {'meta': {'date': ['d2']}, 'parent_ids': ['root'], 'name': 'Other'},
        'a': {'meta': {}, 'parent_ids': ['b'], 'name': 'A'},
        'b': {'meta': {}, 'parent_ids': ['a'], 'name': 'B'},
    }


def test_inherited_uses_nearest_ancestor():
    docs = docs_fixture()
    assert ex.inherited('leaf', 'date', docs) == ['d1']
    assert ex.inherited('own', 'date', docs) == ['d2']


def test_inherited_survives_cycles_and_missing_parents():
    assert ex.inherited('a', 'date', docs_fixture()) == []
    assert ex.inherited('nowhere', 'date', docs_fixture()) == []


def test_ancestry_path_runs_root_down():
    docs = docs_fixture()
    assert ex.ancestry_path('leaf', docs) == 'Tomb > Wall > Text'
    assert ex.ancestry_path('a', docs) == 'B > A'


def test_passport_values_finds_nested_leaves_and_skips_blanks():
    passport = {'children': [{'type': 'date', 'children': [{'type': 'date', 'children': [
        {'type': 'date', 'value': 'd1'}, {'type': 'date', 'value': ''},
        {'type': 'comment', 'value': 'x'}]}]}]}
    assert ex.passport_values(passport) == {'date': ['d1']}


def test_lemma_rows_keep_glosses_and_skip_incomplete_relations():
    row, relations = ex.lemma_rows({
        '_id': '1', 'name': 'ꜣ', 'type': 'substantive', 'subtype': 'substantive_masc',
        'revisionState': 'published',
        'translations': {'translations': [{'lang': 'de', 'value': 'Geier'},
                                          {'lang': 'en', 'value': 'vulture'}]},
        'words': [{'graphics': [{'code': 'G1'}]}],
        'relations': [{'type': 'rootOf', 'objectId': '9'}, {'objectId': '8'}, {'type': 'contains'}],
    })
    assert (row['gloss_de'], row['gloss_en'], row['sign_seq']) == ('Geier', 'vulture', 'G1')
    assert relations == [{'lemma_id': '1', 'relation': 'rootOf', 'target_id': '9'}]
