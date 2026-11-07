#!/usr/bin/env python3
"""
sphinx_trie.py — Gardiner-keyed prefix trie for the SphinxEyes inference
pipeline.

Role: validate / correct YOLOv11 hieroglyph predictions before passing the
phonetic transliteration to GPT for translation.

Edges = single Gardiner codes (your 148 trained classes + 'Unknown').
Terminal nodes carry a list of SourceRecord, one per observation in the
source corpora (Dickson dictionary, BBAW corpus, TLA corpus).

Module-level class so pickled tries are loadable from FastAPI without
__main__-shenanigans.
"""
from __future__ import annotations

import csv
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


UNKNOWN_TOKEN = 'Unknown'


class _TrieUnpickler(pickle.Unpickler):
    """Tries pickled before the move to vision/ name their module 'sphinx_trie'."""

    def find_class(self, module: str, name: str) -> Any:
        if module == 'sphinx_trie':
            module = __name__
        return super().find_class(module, name)


@dataclass
class SourceRecord:
    source: str                     # "dickson" | "bbaw" | "tla"
    translit: str
    translation: str
    freq: int = 1
    # The canonical Gardiner sequence as it was in the source corpus,
    # BEFORE any OOV->Unknown substitution. Useful when several real
    # sequences collapse onto the same trie path through Unknown.
    gardiner_orig: Optional[str] = None


class TrieNode:
    __slots__ = ('children', 'is_end', 'records')

    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.records: list[SourceRecord] = []


class SphinxTrie:
    """Gardiner-sequence prefix trie."""

    def __init__(self, allowed_codes: Optional[set[str]] = None):
        self.root = TrieNode()
        self.total_entries = 0
        self.allowed_codes = set(allowed_codes) if allowed_codes else None

    # ── Insertion ──────────────────────────────────────────────────────
    def insert(self, codes: list[str], record: SourceRecord) -> bool:
        if not codes:
            return False
        if self.allowed_codes is not None:
            if any(c not in self.allowed_codes for c in codes):
                return False
        node = self.root
        for c in codes:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        if not node.is_end:
            self.total_entries += 1
            node.is_end = True
        node.records.append(record)
        return True

    # ── Lookup ─────────────────────────────────────────────────────────
    def _walk(self, codes: list[str]) -> Optional[TrieNode]:
        node = self.root
        for c in codes:
            child = node.children.get(c)
            if child is None:
                return None
            node = child
        return node

    def search(self, codes: list[str]) -> Optional[list[SourceRecord]]:
        node = self._walk(codes)
        return list(node.records) if (node and node.is_end) else None

    def starts_with(self, codes: list[str]) -> bool:
        return self._walk(codes) is not None

    def autocomplete(self, prefix: list[str], max_results: int = 10) -> list[dict]:
        node = self._walk(prefix)
        if node is None:
            return []
        out: list[dict] = []
        self._dfs(node, list(prefix), out, max_results)
        return out

    def _dfs(self, node, path, out, max_results):
        if len(out) >= max_results:
            return
        if node.is_end:
            out.append({
                'gardiner_seq': ' '.join(path),
                'records'     : [asdict(r) for r in node.records],
            })
        for c, child in node.children.items():
            self._dfs(child, path + [c], out, max_results)
            if len(out) >= max_results:
                return

    # ── Bounded Levenshtein over the trie ──────────────────────────────
    def fuzzy_search(self,
                     codes: list[str],
                     max_distance: int = 2,
                     max_results: int = 5) -> list[dict]:
        """
        Bounded edit distance over Gardiner-code sequences, traversing the
        trie and pruning branches whose minimum row > max_distance.

        Ranking: distance asc, total record freq desc.
        """
        n = len(codes)
        if n == 0:
            return []
        candidates: list[tuple[int, int, list[str], TrieNode]] = []
        first_row = list(range(n + 1))
        for c, child in self.root.children.items():
            self._fuzzy_walk(child, c, codes, first_row, [c],
                             max_distance, candidates)
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return [
            {
                'gardiner_seq': ' '.join(path),
                'distance'    : dist,
                'freq'        : freq,
                'records'     : list(node.records),    # SourceRecord instances
            }
            for dist, freq, path, node in candidates[:max_results]
        ]

    def _fuzzy_walk(self, node, edge, target, prev_row, path,
                    max_dist, candidates):
        n = len(target)
        cur_row = [prev_row[0] + 1]
        for i in range(1, n + 1):
            ins = cur_row[i - 1] + 1
            dele = prev_row[i] + 1
            sub = prev_row[i - 1] + (0 if target[i - 1] == edge else 1)
            cur_row.append(min(ins, dele, sub))
        if node.is_end and cur_row[-1] <= max_dist:
            freq = sum(r.freq for r in node.records)
            candidates.append((cur_row[-1], freq, list(path), node))
        if min(cur_row) <= max_dist:
            for c, child in node.children.items():
                self._fuzzy_walk(child, c, target, cur_row, path + [c],
                                 max_dist, candidates)

    # ── Build from CSV ─────────────────────────────────────────────────
    def _ingest_row(self, row, source, oov_strategy,
                    gardiner_col, translit_col, translation_col, freq_col,
                    counters):
        """
        Process one row dict. Mutates `counters` and inserts on success.
        """
        seq_str = (row.get(gardiner_col) or '').strip()
        if not seq_str:
            counters['empty_skipped'] += 1
            return
        codes = seq_str.split()
        gardiner_orig = None
        if self.allowed_codes is not None:
            has_oov = any(c not in self.allowed_codes for c in codes)
            if has_oov:
                if oov_strategy == 'skip':
                    counters['oov_dropped'] += 1
                    return
                gardiner_orig = ' '.join(codes)
                new_codes = []
                for c in codes:
                    if c in self.allowed_codes:
                        new_codes.append(c)
                    else:
                        new_codes.append(UNKNOWN_TOKEN)
                        counters['codes_subbed'] += 1
                codes = new_codes
                counters['oov_substituted'] += 1
        translit = (row.get(translit_col) or '').strip() if translit_col else ''
        translation = (row.get(translation_col) or '').strip() if translation_col else ''
        if freq_col and row.get(freq_col):
            try:
                freq = int(row[freq_col])
            except (ValueError, TypeError):
                freq = 1
        else:
            freq = 1
        rec = SourceRecord(
            source=source,
            translit=translit,
            translation=translation,
            freq=freq,
            gardiner_orig=gardiner_orig,
        )
        if self.insert(codes, rec):
            counters['inserted'] += 1
        else:
            counters['empty_skipped'] += 1

    def build_from_csv(self,
                       csv_path: str | Path,
                       source: str,
                       gardiner_col: str = 'gardiner_seq',
                       translit_col: str = 'translit',
                       translation_col: str = 'translation',
                       freq_col: Optional[str] = None,
                       oov_strategy: str = 'skip') -> dict:
        """
        Insert every row of csv_path into the trie. Returns build stats.

        oov_strategy
            'skip'    : drop the entire row if any code is outside
                        allowed_codes (default).
            'unknown' : replace each OOV code with the literal 'Unknown'
                        token; row is still inserted. The original
                        sequence is preserved on the record as
                        record.gardiner_orig.
        """
        if oov_strategy not in ('skip', 'unknown'):
            raise ValueError(f"oov_strategy must be 'skip' or 'unknown', got {oov_strategy!r}")
        counters = {'inserted': 0, 'oov_dropped': 0, 'oov_substituted': 0,
                    'codes_subbed': 0, 'empty_skipped': 0}
        path = Path(csv_path)
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                self._ingest_row(row, source, oov_strategy,
                                 gardiner_col, translit_col, translation_col, freq_col,
                                 counters)
        return {'csv': str(path), 'source': source,
                'oov_strategy': oov_strategy,
                **counters,
                'total_entries': self.total_entries}

    def build_from_dataframe(self,
                             df,
                             source: str,
                             gardiner_col: str = 'gardiner_seq',
                             translit_col: Optional[str] = 'translit',
                             translation_col: Optional[str] = 'translation',
                             freq_col: Optional[str] = None,
                             oov_strategy: str = 'skip') -> dict:
        """
        Same as build_from_csv but takes a pandas DataFrame.
        translit_col / translation_col may be None when the source has
        no such column.
        """
        if oov_strategy not in ('skip', 'unknown'):
            raise ValueError(f"oov_strategy must be 'skip' or 'unknown', got {oov_strategy!r}")
        counters = {'inserted': 0, 'oov_dropped': 0, 'oov_substituted': 0,
                    'codes_subbed': 0, 'empty_skipped': 0}
        wanted_cols = [c for c in (gardiner_col, translit_col, translation_col, freq_col) if c]
        records = df[wanted_cols].to_dict(orient='records') if wanted_cols else df.to_dict(orient='records')
        for row in records:
            self._ingest_row(row, source, oov_strategy,
                             gardiner_col, translit_col, translation_col, freq_col,
                             counters)
        return {'dataframe_rows': len(df), 'source': source,
                'oov_strategy': oov_strategy,
                **counters,
                'total_entries': self.total_entries}

    # ── Persistence ────────────────────────────────────────────────────
    def to_pickle(self, path: str | Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, path: str | Path) -> 'SphinxTrie':
        with open(path, 'rb') as f:
            obj = _TrieUnpickler(f).load()
        if not isinstance(obj, cls):
            raise TypeError(f'Not a SphinxTrie: {type(obj)}')
        return obj

    def to_json(self, path: str | Path) -> None:
        """For human inspection only — not for runtime.

        Bumps recursion limit since some real corpora (BBAW passages)
        produce trie paths well past Python's default 1000.
        """
        import sys as _sys
        def enc(node):
            return {
                'is_end' : node.is_end,
                'records': [asdict(r) for r in node.records],
                'children': {k: enc(v) for k, v in node.children.items()},
            }
        data = {
            'total_entries': self.total_entries,
            'allowed_codes': sorted(self.allowed_codes) if self.allowed_codes else None,
            'root': enc(self.root),
        }
        prev_limit = _sys.getrecursionlimit()
        # need ~3x max_depth headroom for json's iterencode internals
        _sys.setrecursionlimit(max(prev_limit, 5000))
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        finally:
            _sys.setrecursionlimit(prev_limit)

    def to_summary_json(self, path: str | Path,
                        sample_per_source: int = 20) -> None:
        """
        Compact, depth-bounded inspection artifact: stats plus a small
        sample of entries per source. Use this instead of to_json() when
        the trie is large.
        """
        n_nodes = max_depth = 0
        per_source: dict[str, int] = {}
        samples: dict[str, list] = {}
        terminal_paths_by_depth: dict[int, int] = {}

        def visit(node, path):
            nonlocal n_nodes, max_depth
            n_nodes += 1
            d = len(path)
            if d > max_depth:
                max_depth = d
            if node.is_end:
                terminal_paths_by_depth[d] = terminal_paths_by_depth.get(d, 0) + 1
                for r in node.records:
                    per_source[r.source] = per_source.get(r.source, 0) + 1
                    bucket = samples.setdefault(r.source, [])
                    if len(bucket) < sample_per_source:
                        bucket.append({
                            'gardiner_seq': ' '.join(path),
                            **asdict(r),
                        })
            for c, child in node.children.items():
                visit(child, path + [c])

        visit(self.root, [])

        # depth histogram (binned)
        bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, 200), (201, 1000)]
        depth_hist = []
        for lo, hi in bins:
            cnt = sum(v for d, v in terminal_paths_by_depth.items() if lo <= d <= hi)
            depth_hist.append({'range': f'{lo}..{hi}', 'count': cnt})

        data = {
            'total_entries'    : self.total_entries,
            'total_nodes'      : n_nodes,
            'max_depth'        : max_depth,
            'records_per_source': per_source,
            'terminal_depth_histogram': depth_hist,
            'allowed_codes'    : sorted(self.allowed_codes) if self.allowed_codes else None,
            'samples_per_source': samples,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Stats / introspection ──────────────────────────────────────────
    def stats(self) -> dict:
        n_nodes = 0
        max_depth = 0
        per_source: dict[str, int] = {}

        def visit(node, d):
            nonlocal n_nodes, max_depth
            n_nodes += 1
            if d > max_depth:
                max_depth = d
            for r in node.records:
                per_source[r.source] = per_source.get(r.source, 0) + 1
            for child in node.children.values():
                visit(child, d + 1)

        visit(self.root, 0)
        return {
            'total_entries'     : self.total_entries,
            'total_nodes'       : n_nodes,
            'max_depth'         : max_depth,
            'records_per_source': per_source,
        }

    def __repr__(self) -> str:
        return (f'SphinxTrie(entries={self.total_entries}, '
                f'allowed={len(self.allowed_codes) if self.allowed_codes else "any"})')
