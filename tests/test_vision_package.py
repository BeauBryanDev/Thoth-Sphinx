import pickle
import sys
from pathlib import Path

import pytest

from vision import sphinx_corrector
from vision import sphinx_trie as trie_module
from vision.cartouche_matcher import load_royal_names
from vision.sphinx_trie import SphinxTrie

TRIE_PKL = Path(__file__).parent.parent / 'artifacts' / 'sphinx_trie_v4.pkl'


def test_corrector_imports_the_real_trie() -> None:
    # sphinx_corrector falls back to None on ImportError, which would hide a broken import.
    assert sphinx_corrector.SphinxTrie is SphinxTrie


def test_lexicon_priors_load_from_vision() -> None:
    assert 'wnjs' in load_royal_names()
    assert sphinx_corrector.load_determinatives()
    assert sphinx_corrector.load_initial_particles()


def test_trie_pickled_under_old_module_name_still_loads(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # Pickle the way tries were saved before the move: from a top-level 'sphinx_trie' module.
    monkeypatch.setitem(sys.modules, 'sphinx_trie', trie_module)
    for cls in (trie_module.SphinxTrie, trie_module.TrieNode, trie_module.SourceRecord):
        monkeypatch.setattr(cls, '__module__', 'sphinx_trie')
    data = pickle.dumps(SphinxTrie(), protocol=pickle.HIGHEST_PROTOCOL)
    monkeypatch.undo()

    assert b'sphinx_trie' in data and b'vision.sphinx_trie' not in data
    assert 'sphinx_trie' not in sys.modules
    path = tmp_path / 'old.pkl'
    path.write_bytes(data)
    assert isinstance(SphinxTrie.from_pickle(path), SphinxTrie)


@pytest.mark.slow
def test_saved_trie_artifact_loads() -> None:
    if not TRIE_PKL.exists():
        pytest.skip('trie artifact missing')
    assert isinstance(SphinxTrie.from_pickle(TRIE_PKL), SphinxTrie)
