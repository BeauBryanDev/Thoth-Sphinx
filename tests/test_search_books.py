import pytest

from app.llm import agent_tools, tools
from corpus import embed
from corpus.books import search as book_search

HIT = {
    'id': 'plain:thoth-the-hermes-of-egypt:120',
    'doc_id': 'thoth-the-hermes-of-egypt',
    'title': 'Thoth, the Hermes of Egypt',
    'author': 'Boylan, Patrick',
    'section': 'Chapter XIII. Thoth in magic',
    'header': 'Thoth, the Hermes of Egypt › Chapter XIII. Thoth in magic › p. 124',
    'body': 'Thoth is the lord of magic. ' * 80,
    'page_start': 'p. 124',
    'page_end': 'p. 125',
    'text_rank': 1,
    'vector_rank': 3,
}


@pytest.mark.parametrize(('best', 'embedding_up', 'supported', 'mode'), [
    (0.71, True, True, 'hybrid'),
    (0.31, True, False, 'hybrid'),
    (None, False, True, 'full text only (meaning search unavailable)'),
])
def test_search_books_support_flag_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
    best: float | None,
    embedding_up: bool,
    supported: bool,
    mode: str,
) -> None:
    """A weak best match sets no_book_support; an embedding outage falls back to full text; text is clipped."""
    def fake_embed(api: object, texts: list[str]) -> tuple[list[list[float]], int]:
        if not embedding_up:
            raise ConnectionError('openai down')
        return [[0.1] * 4], 3

    monkeypatch.setattr(embed, 'client', lambda: None)
    monkeypatch.setattr(embed, 'embed_texts', fake_embed)
    monkeypatch.setattr(book_search, 'search_hybrid', lambda q, v, limit: [HIT])
    monkeypatch.setattr(book_search, 'search_vector', lambda v, limit: [{**HIT, 'score': best}])
    monkeypatch.setattr(book_search, 'search_text', lambda q, limit: [{k: v for k, v in HIT.items() if 'rank' not in k}])

    result = agent_tools.run_search_books('What is Thoth the lord of?')
    assert result['no_book_support'] is not supported and result['mode'] == mode
    passage = result['passages'][0]
    assert passage['citation'] == HIT['header'] and passage['pages'] == 'p. 124–p. 125'
    assert len(passage['text']) == agent_tools.MAX_BOOK_TEXT + 1
    assert passage['matched_by'] == (['words', 'meaning'] if embedding_up else ['words'])
    with pytest.raises(ValueError):
        tools.search_books('  ')


def test_chat_book_passages_dedup_weak_and_cap() -> None:
    """Passages from search_books calls only, no repeats, weak when unsupported, at most five."""
    from app.llm import ToolCall
    from app.services.sphinx_chat import MAX_PASSAGES, book_passages

    def passage(n: int) -> dict[str, object]:
        return {'chunk_id': f'c{n}', 'book': 'Thoth', 'author': None, 'section': 'S', 'pages': 'p. 1',
                'citation': 'Thoth › S › p. 1', 'matched_by': ['words'], 'text': 'x'}

    def call(tool: str, ok: bool, passages: list[int], weak: bool = False) -> ToolCall:
        result = {'no_book_support': weak, 'passages': [passage(n) for n in passages]} if ok else None
        return ToolCall(model='m', tool=tool, input={}, summary='', ok=ok, latency_ms=1, result=result)

    out = book_passages([
        call('lookup_entity', True, [9]),
        call('search_books', True, [1, 2, 3], weak=True),
        call('search_books', False, []),
        call('search_books', True, [3, 4, 5, 6]),
    ])
    assert [p.chunk_id for p in out] == ['c1', 'c2', 'c3', 'c4', 'c5'] and len(out) == MAX_PASSAGES
    assert [p.weak for p in out] == [True, True, True, False, False]
