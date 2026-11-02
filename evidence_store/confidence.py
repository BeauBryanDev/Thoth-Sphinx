import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Collection
from typing import Any

from evidence_store import evaluate as ev
from evidence_store import retrieve
from evidence_store.paths import BUILD_DIR, REPORTS_DIR, VISION_SIGNS

ROWS_FILE = BUILD_DIR / 'confidence_rows.jsonl'
REPORT_FILE = REPORTS_DIR / 'parallel_confidence.json'
DEV_SIZE = 500
MIN_RATED_WORDS = 5  # below this, a missing strong hit says nothing about the sentence


def rate_hit(signals: dict[str, Any], 
             query_words: int
             ) -> str:
    """Label one hit. Cutoffs were chosen on the first half of the sample only."""
    if (signals['word_share'] >= 0.7 and signals['word_run'] >= 3 and query_words >= 6):
        return 'strong'
    
    if (signals['word_share'] >= 0.5 and signals['rare_share'] >= 0.4 and query_words >= 5):
        return 'possible'
    
    return 'weak'


def overall_rating(ratings: Collection[str | None]) -> str:
    """Return a search's verdict from its hits' labels."""
    for level in ('strong', 'possible'):
        if level in ratings:
            return level
    return 'none'


def word_weights(words: set[str]) -> dict[str, float]:
    """Return {word: rarity}, using the search index's document frequencies."""
    stats = retrieve.query("SELECT n FROM channel_stats WHERE channel = 'translit'")[0]['n']
    rows = retrieve.query(
        "SELECT gram, df FROM gram_stats WHERE channel = 'translit' AND gram = ANY(%(words)s)",
        words=list(words))
    known = {row['gram']: row['df'] for row in rows}
    
    return {word: ev.math.log(1 + stats / (1 + known.get(word, 0))) for word in words}


def tokens(text: str | None) -> list[str]:
    return retrieve.query('SELECT translit_tokens(%(text)s) AS t', text=text or '')[0]['t']


def sign_tokens(text: str | None) -> list[str]:
    return retrieve.query('SELECT vision_tokens(%(text)s) AS t', text=text or '')[0]['t']


def hit_signals(query_words: list[str],
                query_signs: list[str],
                hit: dict[str, Any],
                weights: dict[str, float]
                ) -> dict[str, Any]:
    """Return candidate signals comparing a query with one hit."""
    words, signs = hit['tokens'], hit['signs']
    shared = set(query_words) & set(words)
    total = sum(weights[word] for word in set(query_words)) or 1
    
    return {'word_run': ev.longest_shared_run(query_words, words),
            'sign_run': ev.longest_shared_run(query_signs, signs),
            'word_share': len(shared) / max(len(set(query_words)), 1),
            'rare_share': sum(weights[word] for word in shared) / total,
            'length_ratio': len(words) / max(len(query_words), 1),
            'both_channels': bool(hit['sign_rank'] and hit['translit_rank']),
            'rrf': hit['score'], 'translit_rank': hit['translit_rank'],
            'sign_rank': hit['sign_rank']
            }


def enrich(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add each hit's folded words and vision signs in one query."""
    if not hits:
        return []
    rows = retrieve.query(
        'SELECT id, translit_tokens(translit_norm) AS tokens, '
        "vision_tokens(coalesce(sign_seq, '')) AS signs FROM sentences WHERE id = ANY(%(ids)s)",
        ids=[hit['sentence_id'] for hit in hits])
    found = {row['id']: row for row in rows}
    
    return [{**hit, 'tokens': found[hit['sentence_id']]['tokens'] or [],
             'signs': found[hit['sentence_id']]['signs'] or []} for hit in hits]


def query_row(sentence_id: str,
              table: dict[str, dict[str, Any]],
              truth: set[str],
              kind: str,
              vocabulary: set[str],
              rng: random.Random
              ) -> dict[str, Any]:
    
    row = table[sentence_id]
    clean = ev.vision_tokens(row['signs'], vocabulary)
    signs = ' '.join(ev.noisy_signs(clean, sorted(vocabulary), rng))
    translit = ev.noisy_translit(row['norm'], rng)
    
    hits = enrich(retrieve.search(signs, translit, 10, 
                                  exclude_text=row['text_id'],
                                  exclude_norm=row['norm'])
                                )
    query_words, query_signs = tokens(translit), sign_tokens(signs)
    all_words = set(query_words) | {word for hit in hits for word in hit['tokens']}
    weights = word_weights(all_words) if all_words else {}
    
    return {'id': sentence_id, 
            'kind': kind, 
            'query_words': len(query_words),
            'query_signs': len(query_signs),
            'hits': [{'true': hit['sentence_id'] in truth,
                      **hit_signals(query_words, 
                                    query_signs, 
                                    hit, weights)
                      } for hit in hits]
            }


def collect(count: int, seed: int) -> None:
    rng = random.Random(seed)
    vocabulary = set(VISION_SIGNS.read_text().split())
    table, sequences, index, _ = ev.load_corpus(BUILD_DIR)
    lemma_index = ev.build_lemma_index(sequences)
    order = ev.eligible_sentences(table, sequences,
                                  vocabulary, rng)
    ROWS_FILE.parent.mkdir(exist_ok=True)
    
    with open(ROWS_FILE, 'w', encoding='utf-8') as handle:
        for done, sentence_id in enumerate(order[:count], 1):
            strict = ev.find_parallels(sentence_id, sequences, index, table)
            loose = ev.find_loose_parallels(sentence_id, sequences, lemma_index, table)
            kind = 'strict' if strict else 'loose' if loose else 'none'
            row = query_row(sentence_id, table, strict | loose, kind, vocabulary, rng)
            handle.write(json.dumps(row) + '\n')
            if done % 100 == 0:
                print(f'{done}/{count}', file=sys.stderr, flush=True)


def measure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return hit-level precision and query-level verdicts for one half of the sample."""
    hits = [(hit, rate_hit(hit, row['query_words'])) for row in rows for hit in row['hits']]
    levels: dict[str, dict[str, float]] = {}
    
    for level in ('strong', 'possible', 'weak'):
        chosen = [hit for hit, rating in hits if rating == level]
        levels[level] = {'hits': len(chosen), 'true': sum(hit['true'] for hit in chosen),
                         'precision': round(sum(hit['true'] for hit in chosen) / max(len(chosen), 1), 3)
                         }
    verdicts: dict[str, dict[str, int]] = {}
    for kind in ('strict', 'loose', 'none'):
        group = [row for row in rows if row['kind'] == kind]
        counts = Counter(overall_rating({rate_hit(hit, row['query_words']) for hit in row['hits']})
                         for row in group)
        verdicts[kind] = {'queries': len(group), **counts}
        
    return {'sentences': len(rows), 
            'hits_by_label': levels,
            'verdict_by_truth': verdicts
            }


def report() -> None:
    with open(ROWS_FILE, encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    result = {'method': 'Natural mix of eligible sentences with simulated noise. Cutoffs were '
                        'chosen on dev; test was not used to choose them.',
              'dev': measure(rows[:DEV_SIZE]),
              'test': measure(rows[DEV_SIZE:])
              }
    REPORT_FILE.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['test'], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure how well search can tell a real parallel from the best weak match.')
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--sentences', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    if args.collect:
        collect(args.sentences, args.seed)
    if args.report:
        report()


if __name__ == '__main__':
    
    main()
