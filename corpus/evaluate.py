import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import cohere
import pandas as pd

from corpus import draft, embed, rerank, retrieve
from corpus.paths import BUILD_DIR, REPORTS_DIR, VISION_SIGNS

MIN_RUN = 4
MIN_SHARED_LEMMAS = 4
LOOSE_OVERLAP = 0.5
COMMON_LEMMA_DF = 2000
TRANSLIT_LETTERS = 'ꜣjꜥwbpfmnrhḥḫẖzsšqkgtṯdḏ'

# Assumed noise, not measured from Sphinx output. Sign rates follow YOLO v9 recall (0.874) and
# precision (0.914); transliteration rates are a guess at a rough draft reading.
SIGN_MISS, SIGN_WRONG, SIGN_SWAP = 0.126, 0.086, 0.05
WORD_DROP, WORD_TYPO, HYPHEN_LOST = 0.15, 0.15, 0.5

MODES = ('signs_clean', 'signs_noisy', 'translit_clean', 'translit_noisy', 'trigram_noisy',
         'embedding_noisy', 'hybrid_noisy', 'hybrid_embedding_noisy')
RERANK_MODES = ('hybrid_rerank_translit', 'hybrid_rerank_gloss')
DRAFTS = {'de': 'German', 'en': 'English', 'de_context': 'German'}
HAIKU_MODES = tuple(f'{kind}_{name}' for name in DRAFTS for kind in ('embedding', 'hybrid'))
DRAFT_WORKERS = 8


def noisy_signs(tokens: list[str],
                vocabulary: list[str],
                rng: random.Random
                ) -> list[str]:
    """Simulate detector output: missed signs, wrong labels, and local reading-order swaps."""
    kept = [rng.choice(vocabulary) if rng.random() < SIGN_WRONG else t
            for t in tokens if rng.random() >= SIGN_MISS]
    
    for i in range(len(kept) - 1):
        if rng.random() < SIGN_SWAP:
            
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
            
    return kept


def noisy_translit(text: str, rng: random.Random) -> str:
    """Simulate a rough draft reading: dropped words, typos, and lost word joins."""
    words = []
    for word in text.split():
        if rng.random() < WORD_DROP:
            continue
        if rng.random() < WORD_TYPO:
            i = rng.randrange(len(word))
            word = word[:i] + rng.choice(TRANSLIT_LETTERS) + word[i + 1:]
        if rng.random() < HYPHEN_LOST:
            word = word.replace('-', ' ')
        words.append(word)
        
    return ' '.join(words)


def longest_shared_run(a: list[str], b: list[str]) -> int:
    """Length of the longest contiguous run of items shared by two sequences."""
    best, previous = 0, [0] * (len(b) + 1)
    for item in a:
        current = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            if item == other:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
                
        previous = current
        
    return best


def gloss_query(sequence: list[str],
                glosses: dict[str, str],
                rng: random.Random
                ) -> str:
    """Simulate a rough understanding: first-sense English glosses with some words missing."""
    words = [glosses[lemma] for lemma in sequence
             if lemma in glosses and rng.random() >= WORD_DROP]
    
    return ' '.join(words)


def first_sense(gloss: str) -> str:
    return gloss.split(';')[0].split(',')[0].strip()


def required_run(length: int) -> int:
    return max(MIN_RUN, math.ceil(length / 2))


def find_parallels(query_id: str,
                   sequences: dict[str, list[str]],
                   index: dict[tuple[str, ...], set[str]],
                   sentences: dict[str, dict[str, Any]]
                   ) -> set[str]:
    """Return ids of sentences from other texts sharing a long lemma run with the query."""
    sequence = sequences[query_id]
    own = sentences[query_id]
    candidates: set[str] = set()
    for i in range(len(sequence) - MIN_RUN + 1):
        candidates |= index.get(tuple(sequence[i:i + MIN_RUN]), set())
    needed = required_run(len(sequence))
    
    return {c for c in candidates
            if sentences[c]['text_id'] != own['text_id']
            and sentences[c]['norm'] != own['norm']
            and longest_shared_run(sequence, sequences[c]) >= needed}


def find_loose_parallels(query_id: str,
                         sequences: dict[str, list[str]],
                         lemma_index: dict[str, set[str]],
                         sentences: dict[str, dict[str, Any]]
                         ) -> set[str]:
    """Return ids from other texts sharing most of the query's rarer lemmas, in any order."""
    own = sentences[query_id]
    lemmas = {lemma for lemma in sequences[query_id] if lemma in lemma_index}
    if len(lemmas) < MIN_SHARED_LEMMAS:
        return set()
    
    shared: defaultdict[str, int] = defaultdict(int)
    
    for lemma in lemmas:
        for sentence_id in lemma_index[lemma]:
            shared[sentence_id] += 1
            
    found: set[str] = set()
    for sentence_id, count in shared.items():
        if count < MIN_SHARED_LEMMAS or sentence_id == query_id:
            continue
        other = {lemma for lemma in sequences[sentence_id] if lemma in lemma_index}
        if (count / len(lemmas | other) >= LOOSE_OVERLAP
                and sentences[sentence_id]['text_id'] != own['text_id']
                and sentences[sentence_id]['norm'] != own['norm']):
            found.add(sentence_id)
            
    return found


def build_lemma_index(sequences: dict[str, list[str]]) -> dict[str, set[str]]:
    index: defaultdict[str, set[str]] = defaultdict(set)
    for sentence_id, sequence in sequences.items():
        for lemma in set(sequence):
            index[lemma].add(sentence_id)
            
    return {lemma: ids for lemma, ids in index.items() if len(ids) <= COMMON_LEMMA_DF}


def score(ranked: list[str],
          relevant: set[str],
          k_values: tuple[int, ...] = (1, 5, 10)
          ) -> dict[str, float]:
    """Hit rate at each k, recall at 10, and reciprocal rank within the top 10."""
    ranks = [i for i, sentence_id in enumerate(ranked[:max(k_values)], 1) if sentence_id in relevant]
    result = {f'hit@{k}': float(any(r <= k for r in ranks)) for k in k_values}
    result['recall@10'] = len([r for r in ranks if r <= 10]) / min(len(relevant), 10)
    result['mrr@10'] = 1 / ranks[0] if ranks else 0.0
    
    return result


def load_corpus(build_dir: Path) -> tuple[dict[str, dict[str, Any]],
                                          dict[str, list[str]],
                                          dict[tuple[str, ...], set[str]],
                                          dict[str, str]]:
    """Return (sentence table, lemma sequences, lemma-run index, English glosses)."""
    sentences = pd.read_csv(build_dir / 'sentences.csv', dtype=str,
                            usecols=['id', 'text_id', 'translit_norm', 'sign_seq'])
    tokens = pd.read_csv(build_dir / 'tokens.csv', dtype=str,
                         usecols=['sentence_id', 'position', 'alternate', 'lemma_id'])
    tokens = tokens[tokens.lemma_id.notna() & tokens.alternate.fillna('0').eq('0')].copy()
    tokens['position'] = tokens.position.astype(int)
    tokens = tokens.sort_values(['sentence_id', 'position'])
    sequences = tokens.groupby('sentence_id').lemma_id.agg(list).to_dict()
    table = {row.id: {'text_id': row.text_id,
                      'norm': row.translit_norm,
                      'signs': row.sign_seq
                      } for row in sentences.itertuples()
             }
    index: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
    
    for sentence_id, sequence in sequences.items():
        for i in range(len(sequence) - MIN_RUN + 1):
            index[tuple(sequence[i:i + MIN_RUN])].add(sentence_id)
            
    lemmas = pd.read_csv(build_dir / 'lemmas.csv', dtype=str, usecols=['id', 'gloss_en'])
    glosses = {row.id: first_sense(row.gloss_en) for row in lemmas.itertuples()
               if isinstance(row.gloss_en, str)}
    
    return table, sequences, index, glosses


def eligible_sentences(table: dict[str, dict[str, Any]],
                       sequences: dict[str, list[str]],
                       vocabulary: set[str],
                       rng: random.Random
                       ) -> list[str]:
    eligible = sorted(
        s for s, row in table.items()
        if isinstance(row['signs'], str) and isinstance(row['norm'], str)
        and len(sequences.get(s, [])) >= 6
        and sum(t.lower() in vocabulary for t in row['signs'].split()) >= 6)
    
    rng.shuffle(eligible)
    
    return eligible


def sample_queries(table: dict[str, dict[str, Any]],
                   sequences: dict[str, list[str]],
                   index: dict[tuple[str, ...], set[str]],
                   vocabulary: set[str],
                   count: int,
                   rng: random.Random
                   ) -> tuple[
                       list[tuple[str, set[str]]],
                       dict[str, float]]:
    """Draw sign-bearing sentences a vision pipeline could read; report parallel coverage."""
    eligible = eligible_sentences(table, sequences, vocabulary, rng)
    queries: list[tuple[str, set[str]]] = []
    examined = 0
    for sentence_id in eligible:
        if len(queries) == count:
            break
        examined += 1
        parallels = find_parallels(sentence_id, sequences, index, table)
        if parallels:
            queries.append((sentence_id, parallels))
            
    return queries, {'eligible': len(eligible), 'examined': examined, 'with_parallels': len(queries),
                     'coverage': len(queries) / examined if examined else 0.0}


def sample_loose_queries(table: dict[str, dict[str, Any]],
                         sequences: dict[str, list[str]],
                         index: dict[tuple[str, ...], set[str]],
                         vocabulary: set[str],
                         count: int,
                         rng: random.Random
                         ) -> tuple[list[tuple[str, set[str]]],
                                    dict[str, float]]:
    """Draw sentences with no strict parallel but at least one loose parallel."""
    lemma_index = build_lemma_index(sequences)
    queries: list[tuple[str, set[str]]] = []
    examined = 0
    for sentence_id in eligible_sentences(table, sequences, vocabulary, rng):
        if len(queries) == count:
            break
        examined += 1
        if find_parallels(sentence_id, sequences, index, table):
            continue
        parallels = find_loose_parallels(sentence_id, sequences, lemma_index, table)
        if parallels:
            queries.append((sentence_id, parallels))
            
    return queries, {'examined': examined, 
                     'with_loose_parallels': len(queries),
                     'coverage': len(queries) / examined if examined else 0.0}


def vision_tokens(signs: str, vocabulary: set[str]) -> list[str]:
    return [t.lower() for t in signs.split() if t.lower() in vocabulary]


def prepare_queries(queries: list[tuple[str, set[str]]],
                    table: dict[str, dict[str, Any]],
                    sequences: dict[str, list[str]],
                    glosses: dict[str, str],
                    vocabulary: set[str],
                    rng: random.Random,
                    gloss_rng: random.Random
                    ) -> list[dict[str, Any]]:
    """Draw every noisy query, then embed the gloss queries in batches."""
    prepared: list[dict[str, Any]] = []
    for sentence_id, parallels in queries:
        row = table[sentence_id]
        clean_signs = vision_tokens(row['signs'], vocabulary)
        prepared.append({
            'id': sentence_id, 
            'parallels': parallels,
            'excluded': {
                'exclude_text': row['text_id'], 
                'exclude_norm': row['norm']
                },
            'clean_signs': ' '.join(clean_signs), 
            'clean_translit': row['norm'],
            'signs': ' '.join(noisy_signs(clean_signs, sorted(vocabulary), rng)),
            'translit': noisy_translit(row['norm'], rng),
            'gloss': gloss_query(sequences[sentence_id], glosses, gloss_rng),
        })
    api = embed.client()
    for start in range(0, len(prepared), embed.BATCH_SIZE):
        batch = [q for q in prepared[start:start + embed.BATCH_SIZE] if q['gloss']]
        vectors, _ = embed.embed_texts(api, [q['gloss'] for q in batch])
        
        for item, vector in zip(batch, vectors):
            item['embedding'] = vector
            
    return prepared


def add_drafts(prepared: list[dict[str, Any]],
               rng: random.Random
               ) -> dict[str, int]:
    """Have Haiku read each noisy query in every draft style, then embed the drafts."""
    api = draft.client()

    def neighbours(q: dict[str, Any]) -> list[str | None]:
        rows = retrieve.context(q['id'], 1)
        found = {row['offset_from_hit']: row['transliteration'] for row in rows}
        return [noisy_translit(found.get(offset) or '', rng) or None for offset in (-1, 1)]

    around = {q['id']: neighbours(q) for q in prepared}

    def read(job: tuple[dict[str, Any], str]) -> tuple[str, int, int]:
        q, name = job
        before, after = around[q['id']] if name.endswith('context') else (None, None)
        
        return draft.translate(q['translit'], DRAFTS[name], before, after, api)

    jobs = [(q, name) for q in prepared for name in DRAFTS]
    
    with ThreadPoolExecutor(DRAFT_WORKERS) as pool:
        results = list(pool.map(read, jobs))
        
    for (q, name), (text, _, _) in zip(jobs, results):
        q.setdefault('drafts', {})[name] = text
        
    ready = [(q, name) for q, name in jobs if q['drafts'][name]]
    vectors = embed.client()
    
    for start in range(0, len(ready), embed.BATCH_SIZE):
        batch = ready[start:start + embed.BATCH_SIZE]
        found, _ = embed.embed_texts(vectors, [q['drafts'][name] for q, name in batch])
        for (q, name), vector in zip(batch, found):
            q.setdefault('draft_embeddings', {})[name] = vector
            
    return {'calls': len(jobs), 
            'blank': len(jobs) - len(ready),
            'input_tokens': sum(r[1] for r in results),
            'output_tokens': sum(r[2] for r in results)
            }


def select_modes(skip_trigram: bool = False,
                 with_rerank: bool = False,
                 only: list[str] | None = None,
                 with_haiku: bool = False
                 ) -> list[str]:
    """Return the mode names to run. An explicit list wins over the flags."""
    if only:
        unknown = set(only) - set(MODES) - set(RERANK_MODES) - set(HAIKU_MODES)
        if unknown:
            raise ValueError(f'unknown modes: {sorted(unknown)}')
        return list(only)
    
    modes = [mode for mode in MODES if not (skip_trigram and mode == 'trigram_noisy')]
    
    return modes + list(RERANK_MODES) * with_rerank + list(HAIKU_MODES) * with_haiku


def run_modes(q: dict[str, Any],
              modes: list[str],
              api: cohere.ClientV2 | None = None,
              delay: float = 0
              ) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Return ({mode: ranked ids}, {mode: seconds}) for one prepared query."""
    excluded, embedding = q['excluded'], q.get('embedding')

    def reranked(text: str) -> list[dict[str, Any]]:
        return rerank.search_reranked(text, q['signs'],
                                      q['translit'], 10, 
                                      api=api, **excluded,
                                      embedding=embedding)

    runs: dict[str, Callable[[], list[dict[str, Any]]]] = {
        'signs_clean': lambda: retrieve.search_channel('sign', q['clean_signs'], 10, **excluded),
        'signs_noisy': lambda: retrieve.search_channel('sign', q['signs'], 10, **excluded),
        'translit_clean': lambda: retrieve.search_channel('translit', q['clean_translit'], 10,
                                                          **excluded),
        'translit_noisy': lambda: retrieve.search_channel('translit', q['translit'], 10, **excluded),
        'trigram_noisy': lambda: retrieve.search_trigram(q['translit'], 10, **excluded),
        'embedding_noisy': lambda: retrieve.search_embedding(embedding, 10, **excluded),
        'hybrid_noisy': lambda: retrieve.search(q['signs'], q['translit'], 10, **excluded),
        'hybrid_embedding_noisy': lambda: retrieve.search(q['signs'], q['translit'], 10, **excluded,
                                                          embedding=embedding),
        'hybrid_rerank_translit': lambda: reranked(q['translit']),
        'hybrid_rerank_gloss': lambda: reranked(f"{q['translit']}\n{q['gloss']}".strip()),
    }
    for name in DRAFTS:
        
        vector = q.get('draft_embeddings', {}).get(name)
        runs[f'embedding_{name}'] = partial(retrieve.search_embedding,
                                            vector, 10, **excluded)
        runs[f'hybrid_{name}'] = partial(retrieve.search, q['signs'], q['translit'], 10,
                                         **excluded, embedding=vector)
        
    results: dict[str, list[str]] = {}
    seconds: dict[str, float] = {}
    
    for mode in modes:
        started = time.perf_counter()
        results[mode] = [hit['sentence_id'] for hit in runs[mode]()]
        seconds[mode] = time.perf_counter() - started
        
        if mode in RERANK_MODES:
            time.sleep(delay)
            
    return results, seconds


def evaluate(prepared: list[dict[str, Any]],
             table: dict[str, dict[str, Any]],
             modes: list[str],
             api: cohere.ClientV2 | None = None,
             delay: float = 0
             ) -> dict[str, dict[str, float]]:
    
    totals: defaultdict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    timings: defaultdict[str, list[float]] = defaultdict(list)
    counts: defaultdict[str, int] = defaultdict(int)
    
    for done, q in enumerate(prepared, 1):
        if done % 25 == 0:
            print(f'{done}/{len(prepared)} queries', file=sys.stderr, flush=True)
            
        signed = {p for p in q['parallels'] if isinstance(table[p]['signs'], str)}
        results, seconds = run_modes(q, modes, api, delay)
        
        for mode, ranked in results.items():
            timings[mode].append(seconds[mode])
            relevant = signed if mode.startswith('signs') else q['parallels']
            
            if not relevant:
                continue
            
            counts[mode] += 1
            
            for metric, value in score(ranked, relevant).items():
                totals[mode][metric] += value
                
    return {mode: {'queries': counts[mode],
                   **{m: round(v / counts[mode], 3) for m, v in metrics.items()},
                   'median_ms': round(1000 * sorted(timings[mode])[len(timings[mode]) // 2])
                   } for mode, metrics in totals.items()
            }


def main() -> None:
    
    parser = argparse.ArgumentParser(description='Measure retrieval on held-out sentences against independently defined parallels.')
    parser.add_argument('--queries', type=int, default=300)
    parser.add_argument('--loose-queries', type=int, default=200)
    parser.add_argument('--skip-trigram', action='store_true', help='drop the slow comparison')
    parser.add_argument('--rerank', action='store_true', help='add the Cohere reranked modes')
    parser.add_argument('--rerank-delay', type=float, default=0,
                        help='seconds between Cohere calls; use 6.5 on a trial key')
    parser.add_argument('--haiku', action='store_true', help='add the Haiku draft modes')
    parser.add_argument('--modes', help='comma-separated modes to run instead of the defaults')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=Path, default=None)
    
    args = parser.parse_args()
    modes = select_modes(args.skip_trigram, 
                         args.rerank, 
                         args.modes and args.modes.split(','),
                         args.haiku)
    
    api = rerank.client() if set(modes) & set(RERANK_MODES) else None
    haiku = set(modes) & set(HAIKU_MODES)
    name = ('retrieval_reranked.json' if api else
            'retrieval_haiku.json' if haiku else 'retrieval_embeddings.json')
    out = args.out or REPORTS_DIR / name
    rng = random.Random(args.seed)
    vocabulary = set(VISION_SIGNS.read_text().split())
    table, sequences, index, glosses = load_corpus(BUILD_DIR)
    queries, coverage = sample_queries(table, sequences,
                                       index, vocabulary,
                                       args.queries, rng)
    
    gloss_rng = random.Random(args.seed + 1)
    
    strict = prepare_queries(queries, table, sequences,
                             glosses, vocabulary, 
                             rng, gloss_rng)
    
    loose_queries, loose_coverage = sample_loose_queries(
                                        table, sequences,
                                        index, vocabulary,
                                        args.loose_queries,
                                        random.Random(args.seed + 2)
                                        )
    loose = prepare_queries(loose_queries, 
                            table, sequences, 
                            glosses, vocabulary,
                            rng, gloss_rng)
    usage = None
    if haiku:
        draft_rng = random.Random(args.seed + 3)  # separate, so the other queries stay unchanged
        usage = add_drafts(strict + loose, draft_rng)
        print(f'drafts: {usage}', file=sys.stderr, flush=True)
        
    report = {'coverage': coverage,
              'modes': evaluate(strict, table, modes, 
                                api, args.rerank_delay),
              'loose_coverage': loose_coverage,
              'loose_modes': evaluate(loose, table, modes,
                                      api, args.rerank_delay)}
    if haiku:
        report['haiku_usage'] = usage
        report['draft_samples'] = [
            {'clean': q['clean_translit'],
             'noisy': q['translit'],
             **q['drafts']
             } for q in strict[:5]]
        
    out.parent.mkdir(exist_ok=True)
    
    out.write_text(json.dumps(report, indent=2))
    
    print(json.dumps(report, indent=2))


# main function runs the evaluation pipeline: 
# it samples queries, prepares them with noise and embeddings, 
# optionally adds Haiku drafts, evaluates retrieval across selected modes,
# and saves a JSON report.

if __name__ == '__main__':
    
    main()
