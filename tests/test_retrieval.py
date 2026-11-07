import psycopg
import pytest

from corpus import retrieve


@pytest.fixture(scope='module', autouse=True)
def database():
    try:
        retrieve.query('SELECT 1 FROM channel_stats LIMIT 1')
    except (psycopg.Error):
        pytest.skip('evidence database is not running or not indexed')


def test_translit_tokens_fold_spelling_conventions():
    rows = retrieve.query("SELECT translit_tokens(%(q)s) AS t", q='ḥm.pl.-nṯr ḏi̯ =sn')
    assert rows[0]['t'] == ['ḥm', 'nṯr', 'ḏi', 'sn']


def test_sign_grams_include_skip_pairs_and_ignore_unseen_signs():
    rows = retrieve.query("SELECT sign_grams(%(q)s) AS g", q='D21 N35 X1')
    assert sorted(rows[0]['g']) == ['d21 n35', 'd21 x1', 'n35 x1']


def test_sign_grams_use_only_the_vision_vocabulary():
    rows = retrieve.query("SELECT vision_tokens(%(q)s) AS t", q='D21 ZZ999')
    assert rows[0]['t'] == ['d21']


def test_hybrid_reports_which_channel_found_each_hit():
    hits = retrieve.search(translit='ḥtp-ḏi̯ nswt', limit=5)
    assert len(hits) == 5
    assert all(hit['translit_rank'] for hit in hits)


@pytest.mark.parametrize('channel, text', [
    ('sign', 'D21 N35 X1 G17'),
    ('translit', 'ḥtp-ḏi̯ nswt'),
])
def test_single_channel_search_returns_ranked_hits(channel, text):
    hits = retrieve.search_channel(channel, text, 5)
    scores = [hit['score'] for hit in hits]
    assert len(hits) == 5
    assert scores == sorted(scores, reverse=True)


def test_exclusions_remove_the_named_text():
    first = retrieve.search(translit='ḥtp-ḏi̯ nswt', limit=1)[0]
    text_id = retrieve.query("SELECT text_id FROM sentences WHERE id = %(i)s",
                             i=first['sentence_id'])[0]['text_id']
    hits = retrieve.search(translit='ḥtp-ḏi̯ nswt', limit=20, exclude_text=text_id)
    ids = [h['sentence_id'] for h in hits]
    assert first['sentence_id'] not in ids


def dates_of(hits):
    ids = [hit['sentence_id'] for hit in hits]
    rows = retrieve.query(
        'SELECT DISTINCT x.id, d.term FROM sentences x '
        'JOIN document_dates d ON d.document_id = x.text_id WHERE x.id = ANY(%(ids)s)', ids=ids)
    terms = {}
    for row in rows:
        terms.setdefault(row['id'], set()).add(row['term'])
    return [terms[i] for i in ids]


def test_date_filter_covers_child_periods():
    hits = retrieve.search(translit='ḥtp-ḏi̯ nswt', limit=50, date='Altes Reich',
                           include_undated=False)
    terms = dates_of(hits)
    assert len(hits) == 50
    assert all('Altes Reich' in t for t in terms)
    assert any('Pepi II.' in t for t in terms)


def test_two_date_document_matches_on_either_date():
    sentence = retrieve.query(
        "SELECT x.id, x.translit_norm FROM sentences x "
        "JOIN document_dates a ON a.document_id = x.text_id AND a.term = '19. Dyn.' "
        "JOIN document_dates b ON b.document_id = x.text_id AND b.term = '20. Dyn.' "
        "WHERE x.word_count >= 8 ORDER BY x.id LIMIT 1")[0]
    for date in ('19. Dyn.', '20. Dyn.'):
        hits = retrieve.search(translit=sentence['translit_norm'], limit=100, date=date,
                               include_undated=False)
        assert sentence['id'] in [hit['sentence_id'] for hit in hits]


def test_filtered_results_keep_undated_sentences_unless_asked():
    undated = retrieve.query(
        "SELECT x.translit_norm FROM sentences x JOIN document_dates d ON d.document_id = x.text_id "
        "WHERE d.term = 'undated' AND x.word_count >= 6 ORDER BY x.id LIMIT 1")[0]['translit_norm']
    kept = dates_of(retrieve.search(translit=undated, limit=20, date='Neues Reich'))
    dropped = dates_of(retrieve.search(translit=undated, limit=20, date='Neues Reich',
                                       include_undated=False))
    assert any('undated' in t for t in kept)
    assert not any('undated' in t for t in dropped)


def test_hybrid_shows_the_date_and_null_when_undated():
    dated = retrieve.search(translit='ḥtp-ḏi̯ nswt', limit=1, date='Altes Reich',
                            include_undated=False)[0]
    undated = retrieve.query(
        "SELECT x.translit_norm FROM sentences x JOIN document_dates d ON d.document_id = x.text_id "
        "WHERE d.term = 'undated' AND x.word_count >= 6 ORDER BY x.id LIMIT 1")[0]['translit_norm']
    hits = retrieve.search(translit=undated, limit=20, date='Neues Reich')
    assert dated['date']
    assert any(hit['date'] is None for hit in hits)


def test_every_channel_returns_a_full_page_under_a_filter():
    vector = retrieve.query('SELECT embedding::text AS e FROM sentence_embeddings LIMIT 1')[0]['e']
    embedding = [float(value) for value in vector.strip('[]').split(',')]
    filters = {'date': 'Neues Reich', 'include_undated': False}
    assert len(retrieve.search_channel('translit', 'ḥtp-ḏi̯ nswt', 10, **filters)) == 10
    assert len(retrieve.search_channel('sign', 'D21 N35 X1 G17', 10, **filters)) == 10
    assert len(retrieve.search_embedding(embedding, 10, **filters)) == 10


def test_embedding_search_finds_rows_under_a_rare_period():
    period = retrieve.query(
        'SELECT p.period FROM date_periods() p JOIN document_dates d ON d.term = p.period '
        'JOIN sentences x ON x.text_id = d.document_id '
        'JOIN sentence_embeddings e ON e.sentence_id = x.id '
        'WHERE p.sentences BETWEEN 20 AND 200 ORDER BY p.sentences, p.period LIMIT 1')[0]['period']
    vector = retrieve.query('SELECT embedding::text AS e FROM sentence_embeddings LIMIT 1')[0]['e']
    embedding = [float(value) for value in vector.strip('[]').split(',')]
    assert retrieve.search_embedding(embedding, 5, date=period, include_undated=False)


def test_unknown_date_raises_instead_of_returning_nothing():
    with pytest.raises(ValueError):
        retrieve.search(translit='ḥtp-ḏi̯ nswt', date='Altes Riech')


def test_context_returns_neighbours_around_the_hit():
    hit = retrieve.search(translit='ḥtp-ḏi̯ nswt', limit=1)[0]
    rows = retrieve.context(hit['sentence_id'], 1)
    assert 0 in [row['offset_from_hit'] for row in rows]


def test_lemma_lookup_works_by_english_gloss_and_transliteration():
    assert any(row['name'] == 'nṯr' for row in retrieve.lookup_lemma('nṯr'))
    assert retrieve.lookup_lemma('vulture')
