# Inspection dumps for the trie: stats, a compact summary and a full JSON
# tree. For a human to read, never for the runtime.

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vision.lexicon_trie import SphinxTrie, TrieNode

# Terminal depth ranges of the summary histogram, both ends included.
DEPTH_BINS = (
    (1, 5),
    (6, 10),
    (11, 20),
    (21, 30),
    (31, 50),
    (51, 100),
    (101, 200),
    (201, 1000),
)

# BBAW passages make paths past Python's default limit of 1000 frames, and
# the json encoder needs about three frames per level.
JSON_RECURSION_LIMIT = 5000


def iter_nodes(trie: SphinxTrie) -> Iterator[tuple[tuple[str, ...], TrieNode]]:
    """Every node with its path, depth first, without recursion."""
    stack: list[tuple[tuple[str, ...], TrieNode]] = [((), trie.root)]

    while stack:
        path, node = stack.pop()
        yield path, node

        # Reversed, so children come off the stack in insertion order.
        for code in reversed(node.children):
            stack.append((path + (code,), node.children[code]))


def trie_stats(trie: SphinxTrie) -> dict[str, Any]:
    """Entry, node and depth counts, plus the records per source."""
    n_nodes = 0
    max_depth = 0
    per_source: Counter[str] = Counter()

    for path, node in iter_nodes(trie):
        n_nodes += 1
        max_depth = max(max_depth, len(path))

        for record in node.records:
            per_source[record.source] += 1

    return {
        'total_entries': trie.total_entries,
        'total_nodes': n_nodes,
        'max_depth': max_depth,
        'records_per_source': dict(per_source),
    }


def _sorted_codes(trie: SphinxTrie) -> list[str] | None:
    if not trie.allowed_codes:
        return None

    return sorted(trie.allowed_codes)


def _write_json(data: dict[str, Any], path: str | Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _depth_histogram(terminals: Counter[int]) -> list[dict[str, Any]]:
    histogram: list[dict[str, Any]] = []

    for low, high in DEPTH_BINS:
        count = 0
        for depth, n_terminals in terminals.items():
            if low <= depth <= high:
                count += n_terminals

        histogram.append({'range': f'{low}..{high}', 'count': count})

    return histogram


def write_summary_json(
    trie: SphinxTrie,
    path: str | Path,
    sample_per_source: int = 20,
) -> None:
    """Stats, a depth histogram and a few sample entries per source."""
    terminals: Counter[int] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}

    for node_path, node in iter_nodes(trie):
        if not node.is_end:
            continue

        terminals[len(node_path)] += 1

        for record in node.records:
            bucket = samples.setdefault(record.source, [])
            if len(bucket) < sample_per_source:
                bucket.append({
                    'gardiner_seq': ' '.join(node_path),
                    **asdict(record),
                })

    stats = trie_stats(trie)

    _write_json(
        {
            'total_entries': stats['total_entries'],
            'total_nodes': stats['total_nodes'],
            'max_depth': stats['max_depth'],
            'records_per_source': stats['records_per_source'],
            'terminal_depth_histogram': _depth_histogram(terminals),
            'allowed_codes': _sorted_codes(trie),
            'samples_per_source': samples,
        },
        path,
    )


def _encode_node(node: TrieNode) -> dict[str, Any]:
    children: dict[str, Any] = {}
    for code, child in node.children.items():
        children[code] = _encode_node(child)

    return {
        'is_end': node.is_end,
        'records': [asdict(r) for r in node.records],
        'children': children,
    }


def write_full_json(trie: SphinxTrie, path: str | Path) -> None:
    """The whole tree as nested JSON. Large: prefer write_summary_json."""
    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(previous_limit, JSON_RECURSION_LIMIT))

    try:
        _write_json(
            {
                'total_entries': trie.total_entries,
                'allowed_codes': _sorted_codes(trie),
                'root': _encode_node(trie.root),
            },
            path,
        )
    finally:
        sys.setrecursionlimit(previous_limit)
