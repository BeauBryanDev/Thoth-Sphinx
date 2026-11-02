import json

import psycopg
import pytest

from evidence_store import graph, retrieve


def test_every_listed_god_relationship_is_reviewed():
    deities = json.loads(graph.GODS_FILE.read_text(encoding='utf-8'))['deities']
    _, links = graph.build_gods(deities)
    assert links


def test_an_unreviewed_relationship_stops_the_build():
    deity = {'name': 'Ra', 'relationships': [{'deity': 'Bes', 'relationship': 'Friend'}]}
    with pytest.raises(ValueError):
        graph.build_gods([deity])


def test_symmetric_links_have_one_direction():
    forward = graph.link('b', 'sibling_of', 'a', None, 's', 'curated')
    assert (forward['source_id'], forward['target_id']) == ('a', 'b')
    parent = graph.link('b', 'parent_of', 'a', None, 's', 'curated')
    assert (parent['source_id'], parent['target_id']) == ('b', 'a')


def test_verified_is_only_trusted_without_a_doubtful_source():
    assert graph.royal_reliability({'verified': True, 'source': 'confirmed on image'}) == 'verified'
    assert graph.royal_reliability({'verified': True,
                                    'source': 'claude-generated. Validate'}) == 'unverified'
    assert graph.royal_reliability({'verified': False, 'source': 'monument'}) == 'unverified'


def test_gods_in_a_name_come_from_its_parts():
    assert graph.gods_in_name('ra_ms_sw_ii') == ['Ra']
    assert graph.gods_in_name('twt_anx_jmn') == ['Amun']
    assert graph.gods_in_name('hat_shpsw') == []
    assert graph.gods_in_name('xafra') == ['Ra']


def test_german_names_use_the_list_spelling():
    assert graph.english_form('Thutmosis III.') == 'thutmose iii'
    assert graph.english_form('Amenophis IV.') == 'amenhotep iv'


@pytest.mark.parametrize('gloss_en, gloss_de, expected', [
    ('Nebkaure (throne name of Amenemhat II)', '[Thronname Amenemhets II.]', 'amenemhat ii'),
    ("Kheperkare", "[Thronname Sesostris' I.]", 'senusret i'),
    ('Sahure', 'Sahure', 'sahure'),
])
def test_king_lemmas_prefer_the_owner_in_the_gloss(gloss_en, gloss_de, expected):
    assert expected in graph.king_lemma_names({'gloss_en': gloss_en, 'gloss_de': gloss_de})


@pytest.mark.parametrize('gloss_en, gloss_de', [
    ('Neferkare', 'Neferkare; [Thronname Pepis II.]'),
    ('Nebmaatre (throne-name of several kings)', '[Thronname Nebmaatres]'),
    ('', 'Kleopatra'),
    ('[Horus-name of Ramesses II, III]', "[Horusname Ramses' II., Ramses' III.]"),
])
def test_shared_king_lemmas_are_skipped(gloss_en, gloss_de):
    assert not graph.king_lemma_names({'gloss_en': gloss_en, 'gloss_de': gloss_de})


@pytest.fixture(scope='module')
def stored():
    try:
        return retrieve.query('SELECT count(*) AS n FROM entities')[0]['n']
    except psycopg.Error:
        pytest.skip('evidence database is not running or has no graph')


def linked(source, relation, target):
    return retrieve.query(
        'SELECT 1 FROM entity_links WHERE source_id = %(s)s AND relation = %(r)s '
        'AND target_id = %(t)s', s=source, r=relation, t=target)


def temple(**fields):
    return {'Name': 'Temple', 'Added_by_audit': False, 'Aliases': [], 'Start_year': -1300,
            'End_year': -1250, 'Builders': [], 'Gods': [], 'Build_by': '', 'Description': '',
            'Location': '', 'Build_year': '', 'Other_builders': [], 'age': '', 'Offerings': [],
            'Offerings_status': '', 'Corpus_find_spots': [], 'Find_spot_covers_whole_site': False,
            'Audit_notes': [], 'Sources': [], **fields}


def test_a_new_temple_god_becomes_a_stub_and_links():
    entities, links, _ = graph.build_temples([temple(Gods=['Aten'])], {}, set(), {'Ra'})
    assert {'god-aten', 'temple-temple'} == {e['id'] for e in entities}
    assert links[0]['relation'] == 'dedicated_to'


def test_an_unknown_temple_builder_stops_the_build():
    with pytest.raises(ValueError):
        graph.build_temples([temple(Builders=['nobody'])], {}, set(), set())


def test_a_builder_outside_the_temple_dates_is_reported():
    ruler = {'name': 'Early king', 'start_year': -2500, 'end_year': -2480}
    _, _, mismatches = graph.build_temples([temple(Builders=['early'])], {'early': ruler}, set(),
                                           set())
    assert mismatches == ['Temple: Early king']


def test_the_temple_file_keeps_its_audit_fields():
    temples = json.loads(graph.TEMPLES_FILE.read_text(encoding='utf-8'))
    assert len(temples) == 21
    assert all('Gods' in item and 'Audit_notes' in item for item in temples)
    assert all(item['Sources'] for item in temples if item['Added_by_audit'])


def test_abu_simbel_was_made_by_ramesses_ii(stored):
    assert linked('temple-great-temple-of-abu-simbel', 'made_by', 'ramesses-ii-d19')
    assert linked('temple-great-temple-of-abu-simbel', 'dedicated_to', 'god-ptah')


def test_find_spots_reach_corpus_documents(stored):
    rows = retrieve.query(
        "SELECT e.name, s.value FROM entities e, "
        "jsonb_array_elements_text(e.attributes->'corpus_find_spots') AS s(value) "
        "WHERE e.type = 'temple' AND s.value NOT IN (SELECT find_spot FROM documents "
        "WHERE find_spot IS NOT NULL)")
    assert rows == []


def test_pyramid_years_and_aliases():
    assert graph.pyramid_years('c. 2670 BCE – 2650 BCE') == (-2670, -2650)
    assert graph.pyramid_years('c. 2600 BCE') == (-2600, -2600)
    assert graph.pyramid_aliases('Red Pyramid (North Pyramid)') == ['Red Pyramid', 'North Pyramid']
    assert graph.pyramid_aliases('Bent Pyramid') == []


def test_an_unreviewed_pyramid_stops_the_build():
    with pytest.raises(ValueError):
        graph.build_pyramids([{'Name': 'New Pyramid'}], {}, set())


def test_every_listed_pyramid_is_reviewed():
    pyramids = json.loads(graph.PYRAMIDS_FILE.read_text(encoding='utf-8'))
    assert {item['Name'] for item in pyramids} == set(graph.PYRAMID_BUILDERS)


def test_great_pyramid_was_made_by_khufu(stored):
    assert linked('pyramid-great-pyramid-of-giza-pyramid-of-khufu', 'made_by', 'khufu-d4')


def test_pyramid_links_to_the_temples_its_file_names(stored):
    assert linked('pyramid-pyramid-of-khafre', 'connected_to', 'temple-valley-temple-of-khafre')


def test_pyramid_texts_paths_reach_documents(stored):
    rows = retrieve.query(
        "SELECT e.name, (SELECT count(*) FROM documents d WHERE d.path LIKE "
        "e.attributes->>'corpus_path' || '%%') AS n FROM entities e "
        "WHERE e.attributes->>'corpus_path' IS NOT NULL")
    assert len(rows) == 3
    assert all(row['n'] > 0 for row in rows)


def test_family_directions_are_right(stored):
    assert linked('god-shu', 'parent_of', 'god-geb')
    assert not linked('god-geb', 'parent_of', 'god-shu')
    assert linked('god-nut', 'parent_of', 'god-osiris')
    assert linked('god-set', 'uncle_of', 'god-horus')


def test_chain_from_royal_name_to_real_sentences(stored):
    rows = retrieve.query(
        "SELECT p.name, p.lemma_ids FROM entity_links l JOIN entities p ON p.id = l.target_id "
        "WHERE l.source_id = 'royal-wnjs' AND l.relation = 'names'")
    assert rows[0]['name'] == 'Unas'
    mentions = retrieve.query('SELECT count(*) AS n FROM tokens WHERE lemma_id = ANY(%(ids)s)',
                              ids=rows[0]['lemma_ids'])
    assert mentions[0]['n'] > 0


def test_every_god_with_a_profile_reaches_the_corpus(stored):
    rows = retrieve.query(
        "SELECT name FROM entities WHERE type = 'god' AND lemma_ids = '{}' "
        "AND NOT coalesce((attributes->>'stub')::boolean, false)")
    assert rows == []


def test_dynasty_and_date_terms_reach_document_dates(stored):
    rows = retrieve.query(
        "SELECT t.term FROM entities e, unnest(e.date_terms) AS t(term) "
        "WHERE e.type = 'pharaoh' AND t.term NOT IN (SELECT term FROM document_dates)")
    assert rows == []


def test_only_one_royal_name_counts_as_verified(stored):
    rows = retrieve.query("SELECT name FROM entities WHERE type = 'royal_name' "
                          "AND reliability = 'verified'")
    assert [row['name'] for row in rows] == ['Unas (Dynasty V)']
