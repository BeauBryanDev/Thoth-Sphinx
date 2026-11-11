# Gardiner-keyed prefix trie: the lexicon the corrector reads YOLO's codes
# against. An edge is one Gardiner code (or Unknown); a terminal node holds
# one SourceRecord per observation in Dickson, BBAW or TLA.
#
# Building lives in trie_builder, dumps in trie_inspect. sphinx_trie keeps
# the old import path.

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

UNKNOWN_TOKEN = 'Unknown'

# Module names older trie pickles carry for the classes below.
OLD_MODULE_NAMES = ('sphinx_trie', 'vision.sphinx_trie')

# A fuzzy hit while walking: (distance, total freq, codes, terminal node).
FuzzyHit = tuple[int, int, list[str], 'TrieNode']


class _TrieUnpickler(pickle.Unpickler):
    """Resolves the classes of a trie pickled under an old module name."""

    def find_class(self, module: str, name: str) -> Any:
        if module in OLD_MODULE_NAMES:
            module = __name__

        return super().find_class(module, name)


@dataclass
class SourceRecord:
    # dickson, bbaw, TLA or curated
    source: str
    translit: str
    translation: str
    freq: int = 1
    # The sequence as the corpus had it, before out-of-vocabulary codes
    # became Unknown. Several real words can share one trie path that way.
    gardiner_orig: Optional[str] = None


class TrieNode:
    __slots__ = ('children', 'is_end', 'records')

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.records: list[SourceRecord] = []


def _edit_row(prev_row: list[int], edge: str, target: list[str]) -> list[int]:
    """Next Levenshtein row after following one trie edge."""
    row = [prev_row[0] + 1]

    for i in range(1, len(target) + 1):
        substitution = prev_row[i - 1]
        if target[i - 1] != edge:
            substitution += 1

        row.append(min(row[i - 1] + 1, prev_row[i] + 1, substitution))

    return row


def _rank_hit(hit: FuzzyHit) -> tuple[int, int]:
    """Closest first, then the most frequent."""
    return hit[0], -hit[1]


class SphinxTrie:
    """Gardiner-sequence prefix trie."""

    def __init__(self, allowed_codes: Optional[set[str]] = None) -> None:
        self.root = TrieNode()
        self.total_entries = 0
        self.allowed_codes = set(allowed_codes) if allowed_codes else None

    def insert(self, codes: list[str], record: SourceRecord) -> bool:
        """Add one record; False for an empty or out-of-vocabulary path."""
        if not codes:
            return False

        if self.allowed_codes is not None:
            for code in codes:
                if code not in self.allowed_codes:
                    return False

        node = self.root
        for code in codes:
            if code not in node.children:
                node.children[code] = TrieNode()

            node = node.children[code]

        if not node.is_end:
            self.total_entries += 1
            node.is_end = True

        node.records.append(record)

        return True

    def _walk(self, codes: list[str]) -> Optional[TrieNode]:
        node = self.root

        for code in codes:
            node = node.children.get(code)
            if node is None:
                return None

        return node

    def search(self, codes: list[str]) -> Optional[list[SourceRecord]]:
        """Records of the exact sequence, or None when it is not a word."""
        node = self._walk(codes)
        if node is None or not node.is_end:
            return None

        return list(node.records)

    def starts_with(self, codes: list[str]) -> bool:
        return self._walk(codes) is not None

    def autocomplete(
        self,
        prefix: list[str],
        max_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Up to max_results words under the prefix, depth first."""
        node = self._walk(prefix)
        if node is None:
            return []

        out: list[dict[str, Any]] = []
        self._collect(node, list(prefix), out, max_results)

        return out

    def _collect(
        self,
        node: TrieNode,
        path: list[str],
        out: list[dict[str, Any]],
        max_results: int,
    ) -> None:
        if len(out) >= max_results:
            return

        if node.is_end:
            out.append({
                'gardiner_seq': ' '.join(path),
                'records': [asdict(r) for r in node.records],
            })

        for code, child in node.children.items():
            path.append(code)
            self._collect(child, path, out, max_results)
            path.pop()

            if len(out) >= max_results:
                return

    def fuzzy_search(
        self,
        codes: list[str],
        max_distance: int = 2,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Words within max_distance edits, closest then most frequent.
        The records are SourceRecord instances, not dicts."""
        if not codes:
            return []

        hits: list[FuzzyHit] = []
        first_row = list(range(len(codes) + 1))
        path: list[str] = []

        for code, child in self.root.children.items():
            path.append(code)
            self._fuzzy_walk(child, codes, first_row, path, max_distance, hits)
            path.pop()

        hits.sort(key=_rank_hit)

        results: list[dict[str, Any]] = []
        for distance, freq, hit_path, node in hits[:max_results]:
            results.append({
                'gardiner_seq': ' '.join(hit_path),
                'distance': distance,
                'freq': freq,
                'records': list(node.records),
            })

        return results

    def _fuzzy_walk(
        self,
        node: TrieNode,
        target: list[str],
        prev_row: list[int],
        path: list[str],
        max_distance: int,
        hits: list[FuzzyHit],
    ) -> None:
        row = _edit_row(prev_row, path[-1], target)

        if node.is_end and row[-1] <= max_distance:
            freq = sum(r.freq for r in node.records)
            hits.append((row[-1], freq, list(path), node))

        # No cell of the row can recover once all are past the bound.
        if min(row) > max_distance:
            return

        for code, child in node.children.items():
            path.append(code)
            self._fuzzy_walk(child, target, row, path, max_distance, hits)
            path.pop()

    def to_pickle(self, path: str | Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: str | Path) -> SphinxTrie:
        """Load a trie pickle. Only for our own artifact, never for uploads."""
        with open(path, 'rb') as f:
            obj = _TrieUnpickler(f).load()

        if not isinstance(obj, cls):
            raise TypeError(f'Not a SphinxTrie: {type(obj)}')

        return obj

    def __repr__(self) -> str:
        allowed = len(self.allowed_codes) if self.allowed_codes else 'any'

        return f'SphinxTrie(entries={self.total_entries}, allowed={allowed})'
