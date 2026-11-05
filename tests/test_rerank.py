from types import SimpleNamespace

import pytest

from corpus import rerank, retrieve

HITS = [{'sentence_id': name, 'score': 0.1, 'sign_rank': None, 'translit_rank': rank,
         'embedding_rank': None, 'date': None}
        for rank, name in enumerate('abc', 1)]
ROWS = [{'id': 'a', 'transliteration': 'ꜥnḫ', 'translation_de': 'Leben'},
        {'id': 'b', 'transliteration': 'nṯr', 'translation_de': None},
        {'id': 'c', 'transliteration': None, 'translation_de': 'Gott'}]


class FakeCohere:
    def __init__(self, order):
        self.order, self.request = order, None

    def rerank(self, **request):
        self.request = request
        results = [SimpleNamespace(index=i, relevance_score=1 - n / 10)
                   for n, i in enumerate(self.order[:request['top_n']])]
        return SimpleNamespace(results=results)


@pytest.fixture
def searched(monkeypatch):
    calls = []

    def fake_search(*args, **kwargs):
        calls.append((args, kwargs))
        return HITS

    monkeypatch.setattr(retrieve, 'search', fake_search)
    monkeypatch.setattr(retrieve, 'query', lambda *args, **kwargs: ROWS)
    return calls


def test_order_follows_cohere_and_channel_ranks_survive(searched):
    hits = rerank.search_reranked('q', translit='q', limit=2, api=FakeCohere([2, 0, 1]))
    assert [hit['sentence_id'] for hit in hits] == ['c', 'a']
    assert [hit['hybrid_rank'] for hit in hits] == [3, 1]
    assert hits[0]['translit_rank'] == 3
    assert hits[0]['rerank_score'] > hits[1]['rerank_score']


def test_request_sends_model_limit_and_both_languages(searched):
    api = FakeCohere([0])
    rerank.search_reranked('draft', translit='q', limit=1, api=api)
    assert api.request['model'] == 'rerank-v5.0-fast'
    assert api.request['query'] == 'draft'
    assert api.request['top_n'] == 1
    assert api.request['documents'] == ['ꜥnḫ\nLeben', 'nṯr', 'Gott']


def test_filters_reach_the_candidate_search(searched):
    rerank.search_reranked('q', translit='q', api=FakeCohere([0]), date='Altes Reich',
                           include_undated=False, exclude_text='t')
    args, kwargs = searched[0]
    assert args[2] == rerank.CANDIDATES
    assert kwargs == {'date': 'Altes Reich', 'include_undated': False, 'exclude_text': 't'}


def test_api_errors_are_raised_not_hidden(searched):
    class Failing:
        def rerank(self, **request):
            raise RuntimeError('rate limited')

    with pytest.raises(RuntimeError):
        rerank.search_reranked('q', translit='q', api=Failing())


def test_no_candidates_skips_the_api(monkeypatch):
    monkeypatch.setattr(retrieve, 'search', lambda *args, **kwargs: [])
    assert rerank.search_reranked('q', api=None) == []
