import json
from pathlib import Path

import pytest

from vision.sphinx_corrector import apply_trie_additions, correct
from vision.sphinx_trie import SourceRecord, SphinxTrie


def write_additions(tmp_path: Path) -> Path:
    path = tmp_path / 'additions.json'
    path.write_text(json.dumps({'entries': [{
        'translit': 'sA ra', 'translation': 'son of Re', 'freq': 20,
        'spellings': [['G39', 'N5'], ['N5', 'G39']]}]}), encoding='utf-8')
    return path


def test_additions_insert_every_spelling_once(tmp_path: Path) -> None:
    trie = SphinxTrie()
    trie.insert(['G39', 'N5'], SourceRecord(source='dickson', translit='sA ra', translation=''))
    path = write_additions(tmp_path)
    assert apply_trie_additions(trie, path) == 2
    # Rerunning is a no-op.
    assert apply_trie_additions(trie, path) == 0
    assert [(r.source, r.freq) for r in trie.search(['N5', 'G39'])] == [('curated', 20)]
    assert [r.source for r in trie.search(['G39', 'N5'])] == ['dickson', 'curated']


def test_missing_additions_file_adds_nothing(tmp_path: Path) -> None:
    assert apply_trie_additions(SphinxTrie(), tmp_path / 'absent.json') == 0


@pytest.mark.parametrize('codes', [['N5', 'G39'], ['G39', 'N5']])
def test_son_of_re_reads_as_one_word_before_amun(corrector: tuple, codes: list[str]) -> None:
    trie, log_prob, unigrams, sub_cost = corrector
    result = correct([[(c, 0.9)] for c in codes + ['M17', 'Y5', 'N35']],
                     trie, log_prob, unigrams, sub_cost_matrix=sub_cost)
    assert result.segmented_words[0].codes == codes
    assert result.segmented_words[0].translit == 'sA ra'
