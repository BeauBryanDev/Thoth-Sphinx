# Score a saved agent eval run against the ground truth. No LLM calls, so
# it is free to rerun. Writes a JSON and a CSV summary and prints a table.

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import agent_eval_metrics as M  # noqa: E402

ROYAL_NAMES = Path('vision/data/royal_names.json')
CSV_COLUMNS = (
    'file', 'condition', 'verdict', 'kings_named', 'invented_kings',
    'translit_similarity', 'translit_strict', 'translit_coverage',
    'truth_incomplete', 'facts_recall', 'sign_error_rate',
    'sign_recall_any_order', 'cartouches_named_right', 'escalated',
    'tool_calls', 'cost_usd', 'read_s',
)


def king_catalog(golden: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Every king we can recognise by name: the lexicon plus the golden set."""
    catalog: dict[str, set[str]] = {}
    names = json.loads(ROYAL_NAMES.read_text(encoding='utf-8'))['names']

    for meta in names.values():
        english = re.sub(r'\(.*?\)', '', meta.get('english', ''))
        english = english.split(',')[0]
        variants = [v.strip() for v in english.split('/') if v.strip()]
        if variants:
            catalog.setdefault(variants[0], set()).update(variants)

    for record in golden:
        for cartouche in record['cartouches']:
            plain = [a for a in cartouche['aliases'] if '-' not in a]
            catalog.setdefault(cartouche['king'], set()).update(plain)

    return catalog


def reading_text(reading: dict[str, Any]) -> str:
    """The English the visitor reads: translation, glosses, second version."""
    parts = [reading.get('full_translation') or '']
    for chunk in reading.get('chunks', []):
        parts.append(chunk.get('english_gloss') or '')

    # Only the second translation itself: its note and passage id cite other
    # kings' monuments, which is a source, not a claim about this wall.
    semantic = reading.get('semantic') or {}
    parts.append(semantic.get('translation') or '')

    return '\n'.join(parts)


def score_run(
    record: dict[str, Any],
    run: dict[str, Any],
    catalog: dict[str, set[str]],
) -> dict[str, Any]:
    """One row of the summary: headline checks, verdict, diagnostics."""
    row: dict[str, Any] = {
        'file': run['file'],
        'condition': run['condition'],
        'read_s': run.get('read_s'),
    }
    reading = run.get('reading')
    errored = reading is None or bool(reading.get('error'))

    if run.get('predict'):
        row.update(M.vision_diagnostics(record, run['predict']))

    if reading is None:
        row.update({'verdict': 'fail', 'error': run.get('error')})
        return row

    text = reading_text(reading)
    translit = reading.get('full_transliteration') or ''
    kings = M.kings_found(record, text, translit)
    invented = M.invented_kings(record, text, catalog)
    strict = M.translit_similarity(record['transliteration'], translit)
    coverage = M.translit_coverage(record['transliteration'], translit)
    # Where the truth does not transcribe everything on the wall, the
    # reading is judged on how much of the truth it covers.
    incomplete = bool(record.get('truth_incomplete'))
    similarity = coverage if incomplete else strict
    recall, missing = M.facts_recall(record, text)

    row.update({
        'verdict': M.verdict(kings, invented, similarity, recall, errored),
        'kings_named': f'{sum(kings.values())}/{len(kings)}',
        'kings_missed': [k for k, hit in kings.items() if not hit],
        'invented_kings': invented,
        'translit_similarity': round(similarity, 3),
        'translit_strict': round(strict, 3),
        'translit_coverage': round(coverage, 3),
        'truth_incomplete': incomplete,
        'facts_recall': round(recall, 3),
        'facts_missing': missing,
        'error': reading.get('error'),
        'agent_transliteration': translit,
        'agent_translation': reading.get('full_translation'),
    })
    row.update(M.process_diagnostics(reading))

    return row


def print_table(rows: list[dict[str, Any]]) -> None:
    print(
        f"{'image':28s} {'cond':9s} {'verdict':8s} {'kings':6s} "
        f"{'translit':>8s} {'facts':>6s} {'signErr':>8s} {'esc':4s} "
        f"{'cost':>7s}"
    )

    for row in rows:
        sign_error = row.get('sign_error_rate')
        print(
            f"{row['file'][:28]:28s} {row['condition']:9s} "
            f"{row['verdict']:8s} {row.get('kings_named', '-'):6s} "
            f"{row.get('translit_similarity', 0):8.2f} "
            f"{row.get('facts_recall', 0):6.2f} "
            f"{'-' if sign_error is None else format(sign_error, '.2f'):>8s} "
            f"{'yes' if row.get('escalated') else 'no':4s} "
            f"{row.get('cost_usd') or 0:7.4f}"
        )


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Verdict counts and spend per condition."""
    out: dict[str, Any] = {}

    for condition in sorted({r['condition'] for r in rows}):
        subset = [r for r in rows if r['condition'] == condition]
        out[condition] = {
            'images': len(subset),
            'verdicts': dict(Counter(r['verdict'] for r in subset)),
            'escalated': sum(1 for r in subset if r.get('escalated')),
            'cost_usd': round(sum(r.get('cost_usd') or 0 for r in subset), 4),
        }

    return out


def verdict_changes(
    rows: list[dict[str, Any]],
    baseline: Path,
) -> list[str]:
    """Images whose verdict differs from the baseline summary."""
    old_rows = json.loads(baseline.read_text(encoding='utf-8'))['rows']
    old = {(r['file'], r['condition']): r['verdict'] for r in old_rows}
    changes: list[str] = []

    for row in rows:
        before = old.get((row['file'], row['condition']))
        if before is not None and before != row['verdict']:
            changes.append(
                f"{row['file']} [{row['condition']}]: "
                f"{before} -> {row['verdict']}"
            )

    return changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Score a saved agent eval run against the ground truth.'
    )
    parser.add_argument('run_dir', type=Path)
    parser.add_argument(
        '--golden', type=Path, default=Path('reports/agent_golden_chosen.json')
    )
    parser.add_argument('--baseline', type=Path, default=None)
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding='utf-8'))['images']
    by_file = {r['file']: r for r in golden}
    catalog = king_catalog(golden)

    rows: list[dict[str, Any]] = []
    for path in sorted(args.run_dir.glob('*.json')):
        if path.name == 'manifest.json' or path.name.startswith('summary'):
            continue

        run = json.loads(path.read_text(encoding='utf-8'))
        rows.append(score_run(by_file[run['file']], run, catalog))

    summary = {
        'run': args.run_dir.name,
        'thresholds': {
            'translit_pass': M.TRANSLIT_PASS,
            'facts_pass': M.FACTS_PASS,
        },
        'totals': totals(rows),
        'rows': rows,
    }
    (args.run_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    csv_path = args.run_dir / 'summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    print_table(rows)
    print()
    print(json.dumps(summary['totals'], indent=2))

    if args.baseline:
        changes = verdict_changes(rows, args.baseline)
        print('\nverdict changes vs baseline:', changes or 'none')


if __name__ == '__main__':
    main()
