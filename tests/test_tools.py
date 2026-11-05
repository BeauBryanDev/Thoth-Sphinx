import psycopg
import pytest

from app.llm import tools
from corpus import confidence, retrieve

LONG_QUERY = 'ḥtp-ḏi̯ nswt wsjr nb ḏdw'


@pytest.fixture(scope='module', autouse=True)
def database():
    try:
        retrieve.query('SELECT 1 FROM entities LIMIT 1')
    except psycopg.Error:
        pytest.skip('evidence database is not running or has no graph')


def signals(share, run, rare=0.5):
    return {'word_share': share, 'word_run': run, 'rare_share': rare}


def test_labels_follow_the_measured_cutoffs():
    assert confidence.rate_hit(signals(0.7, 3), 6) == 'strong'
    assert confidence.rate_hit(signals(0.7, 3), 5) == 'possible'
    assert confidence.rate_hit(signals(0.5, 1), 5) == 'possible'
    assert confidence.rate_hit(signals(0.5, 1, rare=0.3), 9) == 'weak'


def test_overall_rating_takes_the_best_label():
    assert confidence.overall_rating({'weak', 'possible'}) == 'possible'
    assert confidence.overall_rating({'weak'}) == 'none'
    assert confidence.overall_rating(set()) == 'none'


def test_a_known_formula_is_strong_and_citable():
    result = tools.search_parallels(translit=LONG_QUERY)
    assert result['verdict'] == 'strong'
    hit = result['hits'][0]
    assert hit['rating'] == 'strong'
    assert hit['transliteration'] and hit['matched_by'] == ['transliteration']
    assert 'Thesaurus Linguae Aegyptiae' in result['credit']


def test_nonsense_is_reported_as_too_short_or_none_never_strong():
    assert tools.search_parallels(translit='xyz qqq')['verdict'] == 'too_short'
    verdict = tools.search_parallels(translit='xyz qqq vvv www jjj kkk')['verdict']
    assert verdict == 'none'


def test_a_short_phrase_is_not_called_no_parallel():
    assert tools.search_parallels(translit='ḥtp-ḏi̯ nswt')['verdict'] == 'too_short'


def test_signs_alone_are_unrated():
    result = tools.search_parallels(signs='D21 N35 X1 G17')
    assert result['verdict'] == 'unrated'
    assert result['hits'][0]['rating'] == 'unrated'


def test_strong_hits_come_before_weaker_ones():
    ratings = [hit['rating'] for hit in tools.search_parallels(translit=LONG_QUERY,
                                                               limit=10)['hits']]
    assert ratings == sorted(ratings, key=['strong', 'possible', 'weak'].index)


def test_search_needs_some_input():
    with pytest.raises(ValueError):
        tools.search_parallels(signs='  ', translit='')


def test_unknown_date_is_a_clear_error():
    with pytest.raises(ValueError, match='unknown date period'):
        tools.search_parallels(translit=LONG_QUERY, date='Altes Riech')


def test_date_filter_reaches_the_tool():
    hits = tools.search_parallels(translit=LONG_QUERY, date='Neues Reich', include_undated=False,
                                  limit=3)['hits']
    assert hits


def test_a_cohere_failure_falls_back_and_says_so(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('no credit')

    monkeypatch.setattr(tools.rerank, 'search_reranked', fail)
    result = tools.search_parallels(translit=LONG_QUERY, use_rerank=True)
    assert result['rerank'].startswith('failed')
    assert result['hits']


def test_the_tool_limit_is_capped():
    assert len(tools.search_parallels(translit=LONG_QUERY, limit=999)['hits']) <= tools.MAX_LIMIT


def test_lemma_lookup_prefers_real_entries():
    lemmas = tools.lookup_lemma('nṯr')['lemmas']
    assert lemmas[0]['gloss_en'] and lemmas[0]['pos'] not in ('undefined', 'root')


def test_context_rejects_unknown_ids():
    with pytest.raises(ValueError, match='Unknown sentence'):
        tools.get_context('no-such-sentence')


def test_context_keeps_the_hit_in_place():
    hit = tools.search_parallels(translit=LONG_QUERY, limit=1)['hits'][0]
    rows = tools.get_context(hit['sentence_id'], 1)['sentences']
    assert 0 in [row['offset_from_hit'] for row in rows]


def test_periods_are_listed():
    names = [row['period'] for row in tools.list_periods()['periods']]
    assert 'Altes Reich' in names and 'undated' not in names


def test_entity_lookup_finds_a_ruler_by_german_free_spelling():
    assert tools.lookup_entity('Ramses II')['entities'][0]['name'] == 'Ramesses II'


def test_entity_exact_match_ranks_first_and_type_filters():
    assert tools.lookup_entity('Khufu')['entities'][0]['type'] == 'pharaoh'
    assert tools.lookup_entity('Khufu', 'royal_name')['entities'][0]['type'] == 'royal_name'


def test_temple_links_and_corpus_mentions():
    temple = tools.lookup_entity('Abu Simbel')['entities'][0]
    relations = {(link['relation'], link['name']) for link in temple['links']}
    assert ('made_by', 'Ramesses II') in relations
    assert ('dedicated_to', 'Ptah') in relations
    assert temple['corpus']['by_find_spot']['sentences'] > 0


def test_god_reaches_real_sentences_and_says_it_is_background():
    god = tools.lookup_entity('Osiris')['entities'][0]
    assert god['corpus']['by_word']['sentences'] > 1000
    assert god['corpus']['by_word']['examples'][0]['translation_de']
    assert 'Do not present it as evidence' in god['reliability_guide']


def test_incoming_links_are_labelled():
    god = tools.lookup_entity('Osiris')['entities'][0]
    assert any(link['direction'] == 'incoming' and link['name'] == 'Geb'
               and link['relation'] == 'parent_of' for link in god['links'])


def test_a_pyramid_reaches_the_texts_carved_inside_it():
    pyramid = tools.lookup_entity('Pyramid of Unas', 'pyramid')['entities'][0]
    assert pyramid['corpus']['inscribed_here']['sentences'] > 0
    assert pyramid['corpus']['inscribed_here']['examples'][0]['transliteration']


def test_pyramid_finds_by_alias_and_lists_its_builder():
    pyramid = tools.lookup_entity('Great Pyramid of Giza')['entities'][0]
    assert pyramid['type'] == 'pyramid'
    assert 'Khufu' in [link['name'] for link in pyramid['links']]


def test_unknown_entity_finds_nothing():
    assert tools.lookup_entity('zzzzqqq')['found'] == 0


def test_mdc_query_is_converted_before_search():
    # Claude writes MdC ASCII; unconverted, this query found nothing (verdict none).
    result = tools.search_parallels(translit='anx Hr kA-nxt xa-m-WAst', limit=3)
    assert result['query_translit'] == 'ꜥnḫ ḥr kꜣ-nḫt ḫꜥ-m-wꜣst'
    assert result['verdict'] == 'strong'
    assert 'Kꜣ-nḫt-ḫꜥ-m-Wꜣs,t' in result['hits'][0]['transliteration']


def test_apparatus_only_query_is_rejected():
    with pytest.raises(ValueError, match='Give signs'):
        tools.search_parallels(translit='[...] ?')
