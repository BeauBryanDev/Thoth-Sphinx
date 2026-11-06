import argparse
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

from corpus import embed
from corpus.paths import REPORTS_DIR
from corpus.translations import search
from corpus.translations.concordance import unis_concordance
from corpus.translations.extract import Passage, allen, unas_online

# Cross-translation retrieval test: Allen's English of a Unis spell must find Piankoff's English of the same spell.
# Ground truth is Allen's own concordance (his spell -> PT utterance), so no LLM judge is needed.

REPORT = REPORTS_DIR / 'translation_retrieval.json'
TARGET_SOURCE = ['unas_online']
KING = 'UNIS'
DEPTH = 10
SENTENCE_WORDS = 25               # the hard query: the spell's opening, about one sentence


def golden() -> list[dict[str, Any]]:
    """One case per Allen Unis spell whose PT utterance is in the Piankoff text: full and one-sentence queries."""
    pairs = unis_concordance()
    targets = {p.spell for p in unas_online()}
    first: dict[int, Passage] = {}
    for p in allen():
        if p.king == KING and p.spell not in first:
            first[p.spell] = p
    cases = []
    for spell, passage in sorted(first.items()):
        wanted = [u for u in pairs.get(spell, []) if u in targets]
        if not wanted:
            continue
        body = re.sub(r'^RECITATION\.\s*', '', passage.body)
        cases.append({
            'allen_spell': spell,
            'targets': wanted,
            'full': body,
            'sentence': ' '.join(body.split()[:SENTENCE_WORDS]),
        })
    return cases


def spells_of(hits: list[dict[str, Any]]) -> list[int | None]:
    """The spell number of each hit, in rank order."""
    return [h['spell'] for h in hits]


def score(ranked: list[list[int | None]], cases: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """hit@1, hit@3, MRR@10 at spell level, and how many distinct spells the top 3 shows."""
    first: list[int | None] = []
    for spells, case in zip(ranked, cases):
        rank = next((i + 1 for i, s in enumerate(spells[:DEPTH]) if s in case['targets']), None)
        first.append(rank)
    n = len(cases)
    return {
        'query': kind,
        'n': n,
        'hit@1': round(sum(r == 1 for r in first) / n, 3),
        'hit@3': round(sum(r is not None and r <= 3 for r in first) / n, 3),
        'mrr@10': round(sum(1 / r for r in first if r) / n, 3),
        'distinct_spells_top3': round(sum(len(set(s[:3])) for s in ranked) / n, 2),
        'misses': [c['allen_spell'] for c, r in zip(cases, first) if r is None],
    }


def with_retry(call: Callable[[], list[dict[str, Any]]], tries: int = 6) -> list[dict[str, Any]]:
    """Cohere trial keys are rate limited; wait and retry."""
    for attempt in range(tries):
        try:
            return call()
        except Exception as error:
            if attempt == tries - 1 or 'rate' not in str(error).lower() and '429' not in str(error):
                raise
            time.sleep(7 * (attempt + 1))
    return []


def run(cases: list[dict[str, Any]], kind: str, use_rerank: bool) -> list[dict[str, Any]]:
    """Every mode on one query kind."""
    queries = [c[kind] for c in cases]
    variants = [search.name_variants(q, KING) for q in queries]
    flat = [v for vs in variants for v in vs]
    api = embed.client()
    flat_vectors: list[list[float]] = []
    for i in range(0, len(flat), 200):
        flat_vectors += embed.embed_texts(api, flat[i:i + 200])[0]
    vectors: list[list[list[float]]] = []
    at = 0
    for vs in variants:
        vectors.append(flat_vectors[at:at + len(vs)])
        at += len(vs)
    modes: dict[str, Callable[[str, list[list[float]]], list[dict[str, Any]]]] = {
        'vector_parts': lambda q, v: search.search_vector(v[0], KING, DEPTH, TARGET_SOURCE),
        'hybrid_parts': lambda q, v: search.search_hybrid(q, v[0], KING, DEPTH, TARGET_SOURCE),
        'hybrid_grouped': lambda q, v: search.group_by_spell(
            search.search_hybrid(q, v[0], KING, search.CANDIDATES * 2, TARGET_SOURCE))[:DEPTH],
        'hybrid_grouped_names': lambda q, v: search.search_spells(q, v, KING, DEPTH, TARGET_SOURCE),
    }
    if use_rerank:
        modes['hybrid_grouped_rerank'] = lambda q, v: with_retry(lambda: search.rerank(
            q,
            search.group_by_spell(search.search_hybrid(q, v[0], KING, search.CANDIDATES * 2, TARGET_SOURCE))
            [:search.RERANK_CANDIDATES],
            DEPTH,
        ))
        modes['hybrid_grouped_names_rerank'] = lambda q, v: with_retry(
            lambda: search.search_spells(q, v, KING, DEPTH, TARGET_SOURCE, use_rerank=True))
    results = []
    for name, mode in modes.items():
        started = time.monotonic()
        ranked = [spells_of(mode(q, v)) for q, v in zip(queries, vectors)]
        row = {'mode': name, **score(ranked, cases, kind),
               'ms_per_query': round((time.monotonic() - started) * 1000 / len(queries))}
        results.append(row)
        print(f'  {kind:8} {name:22} hit@1 {row["hit@1"]:.3f}  hit@3 {row["hit@3"]:.3f}  '
              f'mrr@10 {row["mrr@10"]:.3f}  distinct@3 {row["distinct_spells_top3"]:.2f}  '
              f'{row["ms_per_query"]} ms', flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate retrieval over the Unas translations with Allen as queries.')
    parser.add_argument('--no-rerank', action='store_true', help='skip the Cohere modes (no cost)')
    parser.add_argument('--rerank-all', action='store_true', help='also rerank the full-passage queries (~$0.90 more)')
    args = parser.parse_args()

    cases = golden()
    print(f'{len(cases)} golden cases (Allen Unis spell -> Piankoff utterance)')
    results = (run(cases, 'full', args.rerank_all and not args.no_rerank)
               + run(cases, 'sentence', not args.no_rerank))
    REPORT.write_text(json.dumps({
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'question': "Does Allen's English of a Unis spell retrieve Piankoff's English of the same spell?",
        'ground_truth': "Allen's concordance, Unis spell -> PT utterance (corpus/translations/concordance.py)",
        'search_scope': {'king': KING, 'sources': TARGET_SOURCE},
        'queries': {'full': 'first part of the Allen spell (<= 420 tokens)',
                    'sentence': f'its first {SENTENCE_WORDS} words'},
        'n_cases': len(cases),
        'results': results,
    }, indent=2))
    print(f'wrote {REPORT}')


if __name__ == '__main__':
    main()
