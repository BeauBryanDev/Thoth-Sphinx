import psycopg
import pytest

from app.llm import agent_tools, tools
from corpus import retrieve


@pytest.mark.parametrize(('span', 'spelling', 'edit'), [
    ('G39 N5', 'G39 N5', 'exact'),
    ('N5 G39', 'G39 N5', 'swap N5 and G39'),
    ('M17 U33 N35', 'M17 Y5 N35', 'read U33 as Y5'),
    ('I10 X1', 'I10 X1 N17', 'add missing N17'),
    ('M17 U33 N35', 'M17 N35', 'drop extra U33'),
    # An extra sign at an end is the shorter span, which is looked up on its own.
    ('G39 D4 Q1', 'D4 Q1', None),
    ('A1 B1 C1', 'C1 B1 A1', None),
])
def test_spelling_edit_names_one_change(span: str, spelling: str, edit: str | None) -> None:
    assert tools.spelling_edit(span.split(), spelling.split()) == edit


def test_unknown_splits_runs_and_junk_is_reported() -> None:
    runs, ignored = tools.sign_runs('g5 Unknown Aa1, D21 cartouche N5')
    assert runs == [['G5'], ['AA1', 'D21'], ['N5']]
    assert ignored == ['cartouche']


def test_span_keys_cover_swaps_and_deletions() -> None:
    keys = tools.span_keys(['A1', 'B1', 'C1'])
    assert {'A1 B1 C1', 'B1 A1 C1', 'A1 C1 B1', 'B1 C1', 'A1 C1', 'A1 B1'} == keys


def test_empty_and_oversized_input_are_clear_errors() -> None:
    with pytest.raises(ValueError, match='required'):
        tools.lookup_signs('  ')
    with pytest.raises(ValueError, match='No Gardiner codes'):
        tools.lookup_signs('Unknown cartouche')
    with pytest.raises(ValueError, match='At most'):
        tools.lookup_signs(' '.join(['N35'] * (tools.MAX_SIGNS + 1)))


def test_summary_names_the_best_word_and_its_fix() -> None:
    result = {'found': 2, 'matches': [{'signs': 'N5 G39', 'change': 'swap N5 and G39',
                                       'words': [{'name': 'zꜣ-Rꜥw', 'gloss_en': 'son of Re'}]}]}
    assert agent_tools.summarize('lookup_signs', result) == \
        '2 spellings — N5 G39 = zꜣ-Rꜥw: son of Re, swap N5 and G39'


@pytest.fixture(scope='module')
def vocabulary() -> None:
    try:
        retrieve.query('SELECT 1 FROM sign_spellings LIMIT 1')
    except psycopg.Error:
        pytest.skip('evidence database is not running or db/vocabulary.sql has not run')


def test_son_of_re_in_both_orders(vocabulary: None) -> None:
    matches = tools.lookup_signs('N5 G39')['matches']
    assert matches[0]['match'] == 'exact' and matches[0]['words'][0]['name'] == 'zꜣ-Rꜥw'
    swap = next(m for m in matches if m['match'] == 'corrected')
    assert swap['spelling'] == 'G39 N5' and swap['change'] == 'swap N5 and G39'


def test_a_missing_determinative_is_proposed(vocabulary: None) -> None:
    matches = tools.lookup_signs('I10 X1')['matches']
    assert any(m['change'] == 'add missing N17' and m['words'][0]['name'] == 'ḏ.t'
               for m in matches)


def test_a_line_returns_words_in_reading_order(vocabulary: None) -> None:
    result = agent_tools.run_lookup_signs('M17 U33 N35 G5 G39 D4 Q1 M23 N5 U1 D36 T22 G43 Aa1 D21')
    starts = [m['start'] for m in result['matches']]
    assert starts == sorted(starts)
    assert any(m['signs'] == 'D4 Q1' and m['words'][0]['name'] == 'Wsjr' for m in result['matches'])


def test_middle_egyptian_only_counts_only_those_texts(vocabulary: None) -> None:
    for match in tools.lookup_signs('G39 N5', middle_egyptian_only=True)['matches']:
        assert all(word['me_tokens'] > 0 for word in match['words'])
