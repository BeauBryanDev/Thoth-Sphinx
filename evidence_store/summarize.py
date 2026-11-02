import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evidence_store.paths import REPORTS_DIR

# Combine every saved evaluation run into reports/evaluation_summary.csv and .json.

# Usage: python -m evidence_store.summarize

METRICS = ('queries', 'hit@1', 'hit@5', 'hit@10', 'recall@10', 'mrr@10', 'median_ms')
RUNS = (
    ('retrieval_baseline.json', 'baseline', '2026-10-31', 'valid',
     'Search channels without embeddings, 300 strict queries.'),
    ('retrieval_embeddings.json', 'embeddings', '2026-10-31', 'valid',
     'Adds embeddings of English gloss queries. Measured before the vector speed fix, so '
     'embedding and three-channel median_ms are out of date.'),
    ('retrieval_reranked.json', 'reranker', '2026-11-01', 'valid',
     'Cohere rerank-v5.0-fast on a smaller sample, 150 strict and 100 loose queries.'),
    ('retrieval_haiku_unclean_drafts.json', 'haiku_unclean', '2026-11-01', 'invalid',
     'Do not use the Haiku modes: many drafts contained Haiku\'s working notes. Baselines are fine.'),
    ('retrieval_haiku.json', 'haiku', '2026-11-01', 'valid',
     'Clean Haiku drafts from structured output, on the same queries as the embeddings run.'),
)


def load_runs(reports: Path) -> list[dict[str, Any]]:
    """Return one summary per saved run, skipping any report that doesn't exist."""
    runs: list[dict[str, Any]] = []
    for filename, name, date, status, note in RUNS:
        path = reports / filename
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        
        runs.append({'run': name, 
                     'date': date,
                     'status': status,
                     'note': note, 
                     'file': filename,
                     'strict_coverage': report.get('coverage'),
                     'loose_coverage': report.get('loose_coverage'),
                     'haiku_usage': report.get('haiku_usage'),
                     'strict': report.get('modes', {}), 
                     'loose': report.get('loose_modes', {})
                     })
    # returns a list of dicts, each dict contains the summary of a run, 
    # # including its name, date, status, note, file name, coverage metrics,
    # and mode metrics for both strict and loose tests.
    return runs


def rows(runs: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Flatten runs into one row per run, test and search mode."""
    for run in runs:
        for test in ('strict', 'loose'):
            for mode, values in run[test].items():
                yield {'run': run['run'],
                       'date': run['date'],
                       'status': run['status'],
                       'test': test, 
                       'mode': mode,
                       **{m: values.get(m) for m in METRICS},
                       'note': run['note']
                       }


def main() -> None:
    parser = argparse.ArgumentParser(description='Combine every saved evaluation run into reports/evaluation_summary.csv and .json.')
    parser.add_argument('--reports', type=Path, default=REPORTS_DIR)
    parser.add_argument('--out', type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    runs = load_runs(args.reports)
    table = list(rows(runs))
    args.out.mkdir(exist_ok=True)
    
    with open(args.out / 'evaluation_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, ['run', 'date', 'status', 'test', 'mode', *METRICS, 'note'])
        writer.writeheader()
        writer.writerows(table)
    (args.out / 'evaluation_summary.json').write_text(
        json.dumps({'runs': runs}, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(runs)} runs, {len(table)} rows written to {args.out}')


# Main entry point for the script. 
# When executed directly, it will call the main function to perform the summarization of evaluation runs.
if __name__ == '__main__':
    
    main()
