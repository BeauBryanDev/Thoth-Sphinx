import psycopg
import pytest

from app.llm import agent_tools
from corpus import retrieve


def test_compact_hit_clips_and_prefers_english() -> None:
    hit = {'sentence_id': 'X', 'rating': 'strong', 'matched_by': ['sign'],
           'transliteration': 'a' * 500, 'translation_de': 'Horus', 'translation_en': 'Horus EN',
           'date': 'Sethos I.', 'text_name': 'Abydos', 'find_spot': 'not sent'}
    out = agent_tools.compact_hit(hit)
    assert len(out['transliteration']) == agent_tools.MAX_HIT_TEXT + 1
    assert out['translation'] == 'Horus EN' and out['translation_lang'] == 'en'
    assert 'find_spot' not in out


def test_definitions_are_api_shaped() -> None:
    for tool in agent_tools.READING_TOOLS:
        d = tool.definition()
        assert set(d) == {'name', 'description', 'input_schema'}
        assert d['input_schema']['type'] == 'object' and d['input_schema']['required']


@pytest.mark.parametrize(('name', 'result', 'expected'), [
    ('search_parallels', {'line': 2, 'verdict': 'strong', 'hits': [
        {'transliteration': 'Ḥr,w Kꜣ-nḫt', 'date': 'Sethos I.'}]}, 'line 2: strong — best: Ḥr,w Kꜣ-nḫt (Sethos I.)'),
    ('search_parallels', {'line': 1, 'verdict': 'none', 'hits': []}, 'line 1: none'),
    ('lookup_lemma', {'found': 2, 'lemmas': [{'name': 'ḫꜥ', 'gloss_en': 'hill'}]}, '2 lemmas — ḫꜥ: hill'),
    ('lookup_entity', {'found': 0, 'entities': []}, '0 entities'),
])
def test_summaries(name: str, result: dict, expected: str) -> None:
    assert agent_tools.summarize(name, result) == expected


def test_search_parallels_tool_on_the_corpus() -> None:
    try:
        retrieve.query('SELECT 1')
    except psycopg.Error:
        pytest.skip('evidence database is not running')
    result = agent_tools.run_search_parallels(1, translit='anx Hr kA-nxt xa-m-WAst')
    assert result['line'] == 1 and result['verdict'] == 'strong'
    assert result['query_translit'] == 'ꜥnḫ ḥr kꜣ-nḫt ḫꜥ-m-wꜣst'
    assert len(result['hits']) <= 3 and result['hits'][0]['sentence_id']
