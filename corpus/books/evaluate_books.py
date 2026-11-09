import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from corpus import embed
from corpus.books.judge import Judge
from corpus.books.search import search_hybrid, search_text, search_vector
from corpus.paths import REPORTS_DIR

# Scores book search against the golden questions. A strict hit is a chunk of the right book that
# contains the evidence quote; a loose hit is a chunk of the right book whose pages overlap the gold pages.
# With --judge, Claude Sonnet grades whether each top-5 passage answers the question, from any book.

GOLDEN = REPORTS_DIR / 'book_golden_questions.jsonl'
MODES = ('text', 'vector', 'hybrid')
K = 10
JUDGE_K = 5


def squash(text: str) -> str:
    """Whitespace and quote style folded, the same way the golden set was checked."""
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', text).strip().lower()


def page_range(start: str | None, end: str | None) -> tuple[str, int, int] | None:
    """('p.', 12, 14) from 'p. 12' and 'p. 14'; the prefix tells printed from PDF pages."""
    if not start:
        return None
    kind, first = start.rsplit(' ', 1)
    last = (end or start).rsplit(' ', 1)[1]

    return kind, int(first), int(last)


def is_hit(hit: dict[str, Any], gold: dict[str, Any], strict: bool) -> bool:
    """Right book, and the quote inside (strict) or overlapping pages (loose)."""
    if hit['doc_id'] != gold['doc_id']:
        return False
    if strict:
        return squash(gold['evidence']) in squash(hit['body'])
    a = page_range(hit['page_start'], hit['page_end'])
    b = page_range(gold['page_start'], gold['page_end'])

    return bool(a and b and a[0] == b[0] and a[1] <= b[2] and b[1] <= a[2])


def first_hit(hits: list[dict[str, Any]], gold: dict[str, Any], strict: bool) -> int | None:
    """1-based rank of the first hit, or None."""
    return next((i for i, h in enumerate(hits, start=1) if is_hit(h, gold, strict)), None)


def summary(ranks: list[int | None]) -> dict[str, float]:
    """hit@1/5/10 and MRR@10."""
    n = len(ranks)
    return {
        'hit@1': round(sum(r is not None and r <= 1 for r in ranks) / n, 3),
        'hit@5': round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
        'hit@10': round(sum(r is not None and r <= 10 for r in ranks) / n, 3),
        'mrr@10': round(sum(1 / r for r in ranks if r is not None) / n, 3),
    }


def run(
    golden: list[dict[str, Any]],
    vectors: dict[str, list[float]],
    variant: str,
) -> tuple[dict[str, list[list[dict[str, Any]]]], 
            dict[str, float]]:
    """Top-K hits per mode per question, and mean latency per mode in ms."""
    results: dict[str, list[list[dict[str, Any]]]] = {m: [] for m in MODES}
    timing: dict[str, list[float]] = {m: [] for m in MODES}
    for q in golden:
        for mode in MODES:
            start = time.perf_counter()
            if mode == 'text':
                hits = search_text(q['question'], variant, K)
            elif mode == 'vector':
                hits = search_vector(vectors[q['id']], variant, K)
            else:
                hits = search_hybrid(q['question'], vectors[q['id']], variant, K)
            timing[mode].append(1000 * (time.perf_counter() - start))
            results[mode].append(hits)

    return results, {m: round(statistics.mean(t), 1) for m, t in timing.items()}


def negative_check(
    golden: list[dict[str, Any]],
    results: dict[str, list[list[dict[str, Any]]]],
) -> dict[str, dict[str, Any]]:
    """Top-1 vector score of negatives against positives: can a threshold say 'not in these books'?"""
    tops = [hits[0]['score'] if hits else 0.0 for hits in results['vector']]
    pos = sorted(t for t, q in zip(tops, golden) if not q['negative'])
    neg = sorted(t for t, q in zip(tops, golden) if q['negative'])
    if not pos or not neg:
        return {}

    floor = pos[max(0, int(0.1 * len(pos)) - 1)]
    return {
        'vector_top1': {
            'positives_median': round(statistics.median(pos), 3),
            'positives_p10': round(floor, 3),
            'negatives': [round(t, 3) for t in neg],
            'negatives_below_positive_p10': sum(t < floor for t in neg),
        },
    }


def judged(
    judge: Judge,
    golden: list[dict[str, Any]],
    results: dict[str, list[list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Sonnet verdicts on every top-5 passage, then answered@1/@5 per mode and two checks on the judge itself."""
    for n, q in enumerate(golden, start=1):
        seen: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            for hit in results[mode][n - 1][:JUDGE_K]:
                seen.setdefault(hit['id'], hit)
        judge.judge(q, list(seen.values()))
        if n % 10 == 0:
            print(f'  judged {n}/{len(golden)} questions, ${judge.spent:.2f} so far', flush=True)

    def verdict(q: dict[str, Any], hit: dict[str, Any]) -> str:
        found = judge.known(q['question'], hit['body'])
        return found['verdict'] if found else 'unjudged'

    out: dict[str, Any] = {'modes': {}}
    for mode in MODES:
        out['modes'][mode] = {}
        for accept in (('yes',), ('yes', 'partial')):
            ranks = []
            for q, hits in zip(golden, results[mode]):
                if q['negative']:
                    continue
                marks = [verdict(q, h) in accept for h in hits[:JUDGE_K]]
                ranks.append(marks.index(True) + 1 if any(marks) else None)
            n = len(ranks)
            out['modes'][mode]['answered' if len(accept) == 1 else 'answered_or_partial'] = {
                'hit@1': round(sum(r == 1 for r in ranks) / n, 3),
                'hit@5': round(sum(r is not None for r in ranks) / n, 3),
                'mrr@5': round(sum(1 / r for r in ranks if r is not None) / n, 3),
            }
    strict = [
        verdict(q, h)
        for q, hits in zip(golden, results['hybrid'])
        if not q['negative']
        for h in hits[:JUDGE_K]
        if is_hit(h, q, True)
    ]
    negative = [
        verdict(q, h)
        for q, hits in zip(golden, results['hybrid'])
        if q['negative']
        for h in hits[:JUDGE_K]
    ]
    out['judge_checks'] = {
        'strict_hits_judged_yes': f'{strict.count("yes")}/{len(strict)}',
        'negative_passages_judged_yes': f'{negative.count("yes")}/{len(negative)}',
    }
    out['judge_cost_usd'] = round(judge.spent, 3)
    out['judge_calls'] = judge.calls
    
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Score book search (text, vector, hybrid) on the golden questions.')
    parser.add_argument('--variant', default='plain', help='which chunking in book_chunks to search')
    parser.add_argument('--golden', type=Path, default=GOLDEN, help=f'default {GOLDEN}')
    parser.add_argument('--out', type=Path, help='report JSON (default reports/book_retrieval_<variant>.json)')
    parser.add_argument('--judge', action='store_true', help='have Claude Sonnet grade every top-5 passage')
    parser.add_argument('--judge-max-usd', type=float, default=5.0, help='spending guard for --judge')
    args = parser.parse_args()

    golden = [json.loads(line) for line in args.golden.read_text().splitlines() if line.strip()]
    positives = [q for q in golden if not q['negative']]
    vectors_list, tokens = embed.embed_texts(embed.client(), [q['question'] for q in golden])
    vectors = {q['id']: v for q, v in zip(golden, vectors_list)}
    results, latency = run(golden, vectors, args.variant)

    report: dict[str, Any] = {
        'variant': args.variant,
        'questions': len(positives),
        'negatives': len(golden) - len(positives),
        'query_embedding_tokens': tokens,
        'latency_ms': latency,
        'modes': {},
        'misses': {},
    }
    print(f'{args.variant}: {len(positives)} questions, {len(golden) - len(positives)} negatives\n')
    print(f'{"mode":<8} {"match":<7} {"hit@1":>6} {"hit@5":>6} {"hit@10":>6} {"mrr@10":>7}  latency')
    for mode in MODES:
        report['modes'][mode] = {}
        hits_for = [h for h, q in zip(results[mode], golden) if not q['negative']]
        for match in ('strict', 'loose'):
            ranks = [first_hit(h, q, match == 'strict') for h, q in zip(hits_for, positives)]
            scores = summary(ranks)
            report['modes'][mode][match] = scores
            if match == 'strict':
                report['misses'][mode] = [q['id'] for q, r in zip(positives, ranks) if r is None or r > 5]
            print(f'{mode:<8} {match:<7} {scores["hit@1"]:>6} {scores["hit@5"]:>6} {scores["hit@10"]:>6} '
                  f'{scores["mrr@10"]:>7}  {latency[mode]} ms')
    report['negative_check'] = negative_check(golden, results)
    neg = report['negative_check'].get('vector_top1')
    if neg:
        print(f'\nvector top-1 score: positives median {neg["positives_median"]}, p10 {neg["positives_p10"]}; '
              f'negatives {neg["negatives"]} ({neg["negatives_below_positive_p10"]} of {len(neg["negatives"])} '
              'below the positives p10)')
    print(f'\nmissed at 5 by hybrid (strict): {", ".join(report["misses"]["hybrid"]) or "none"}')
    if args.judge:
        print('\njudging top-5 passages with Claude Sonnet...')
        report['judged'] = judged(Judge(max_usd=args.judge_max_usd), golden, results)
        print(f'\n{"mode":<8} {"accept":<20} {"hit@1":>6} {"hit@5":>6} {"mrr@5":>6}')
        for mode, scores in report['judged']['modes'].items():
            for accept, s in scores.items():
                print(f'{mode:<8} {accept:<20} {s["hit@1"]:>6} {s["hit@5"]:>6} {s["mrr@5"]:>6}')
        checks = report['judged']['judge_checks']
        print(f'\njudge checks: strict hits judged yes {checks["strict_hits_judged_yes"]}; '
              f'negative-question passages judged yes {checks["negative_passages_judged_yes"]}')
        print(f'judge: {report["judged"]["judge_calls"]} calls, ${report["judged"]["judge_cost_usd"]:.3f}')
    out = args.out or REPORTS_DIR / f'book_retrieval_{args.variant}.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
