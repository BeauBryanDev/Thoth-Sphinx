# Build-time loaders for the trie: insert corpus rows from a CSV or a pandas
# DataFrame. Not used at runtime, which only loads the pickle.

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from vision.lexicon_trie import UNKNOWN_TOKEN, SourceRecord, SphinxTrie

# skip drops a row with any out-of-vocabulary code; unknown inserts it with
# those codes replaced by Unknown, the original kept in gardiner_orig.
OOV_STRATEGIES = ('skip', 'unknown')


@dataclass(frozen=True)
class RowColumns:
    gardiner: str = 'gardiner_seq'
    translit: Optional[str] = 'translit'
    translation: Optional[str] = 'translation'
    freq: Optional[str] = None


def _new_counters() -> dict[str, int]:
    return {
        'inserted': 0,
        'oov_dropped': 0,
        'oov_substituted': 0,
        'codes_subbed': 0,
        'empty_skipped': 0,
    }


def _check_strategy(oov_strategy: str) -> None:
    if oov_strategy not in OOV_STRATEGIES:
        raise ValueError(
            f'oov_strategy must be one of {OOV_STRATEGIES}, '
            f'got {oov_strategy!r}'
        )


def _cell(row: dict[str, Any], column: Optional[str]) -> str:
    """Stripped text of a cell; empty for a missing or non-text value."""
    if not column:
        return ''

    value = row.get(column)
    if not isinstance(value, str):
        return ''

    return value.strip()


def _row_freq(row: dict[str, Any], column: Optional[str]) -> int:
    if not column or not row.get(column):
        return 1

    try:
        return int(row[column])
    except (ValueError, TypeError):
        return 1


def _mask_oov(
    codes: list[str],
    allowed: set[str],
    counters: dict[str, int],
) -> list[str]:
    """The codes with every out-of-vocabulary one replaced by Unknown."""
    masked: list[str] = []

    for code in codes:
        if code in allowed:
            masked.append(code)
        else:
            masked.append(UNKNOWN_TOKEN)
            counters['codes_subbed'] += 1

    return masked


def _ingest_row(
    trie: SphinxTrie,
    row: dict[str, Any],
    source: str,
    columns: RowColumns,
    oov_strategy: str,
    counters: dict[str, int],
) -> None:
    """Insert one corpus row and count what happened to it."""
    codes = _cell(row, columns.gardiner).split()
    if not codes:
        counters['empty_skipped'] += 1
        return

    gardiner_orig = None
    allowed = trie.allowed_codes

    if allowed is not None and not allowed.issuperset(codes):
        if oov_strategy == 'skip':
            counters['oov_dropped'] += 1
            return

        gardiner_orig = ' '.join(codes)
        codes = _mask_oov(codes, allowed, counters)
        counters['oov_substituted'] += 1

    record = SourceRecord(
        source=source,
        translit=_cell(row, columns.translit),
        translation=_cell(row, columns.translation),
        freq=_row_freq(row, columns.freq),
        gardiner_orig=gardiner_orig,
    )

    if trie.insert(codes, record):
        counters['inserted'] += 1
    else:
        counters['empty_skipped'] += 1


def build_from_csv(
    trie: SphinxTrie,
    csv_path: str | Path,
    source: str,
    columns: RowColumns = RowColumns(),
    oov_strategy: str = 'skip',
) -> dict[str, Any]:
    """Insert every row of the CSV into the trie; returns the build stats."""
    _check_strategy(oov_strategy)
    counters = _new_counters()

    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            _ingest_row(trie, row, source, columns, oov_strategy, counters)

    return {
        'csv': str(csv_path),
        'source': source,
        'oov_strategy': oov_strategy,
        **counters,
        'total_entries': trie.total_entries,
    }


def build_from_dataframe(
    trie: SphinxTrie,
    df: Any,
    source: str,
    columns: RowColumns = RowColumns(),
    oov_strategy: str = 'skip',
) -> dict[str, Any]:
    """Same as build_from_csv for a pandas DataFrame."""
    _check_strategy(oov_strategy)
    counters = _new_counters()

    for row in df.to_dict(orient='records'):
        _ingest_row(trie, row, source, columns, oov_strategy, counters)

    return {
        'dataframe_rows': len(df),
        'source': source,
        'oov_strategy': oov_strategy,
        **counters,
        'total_entries': trie.total_entries,
    }
