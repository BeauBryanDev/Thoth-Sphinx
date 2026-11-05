import argparse
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI

from corpus import retrieve
from corpus.paths import ENV_FILE

MODEL = 'text-embedding-3-small'
DIMENSIONS = 1536
BATCH_SIZE = 500
PRICE_PER_MILLION_TOKENS = 0.02


def client() -> OpenAI:
    return OpenAI(api_key=dotenv_values(ENV_FILE)['OPENAI_API_KEY'], max_retries=5)


def embed_texts(api: OpenAI, texts: list[str]) -> tuple[list[list[float]], int]:
    """Return (vectors, tokens used) in input order."""
    response = api.embeddings.create(model=MODEL, input=texts, 
                                     dimensions=DIMENSIONS)
    
    vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
    
    return vectors, response.usage.total_tokens


def pending(limit: int) -> list[dict[str, Any]]:
    return retrieve.query(
        "SELECT s.id, s.translation_de FROM sentences s "
        "LEFT JOIN sentence_embeddings e ON e.sentence_id = s.id "
        "WHERE e.sentence_id IS NULL AND btrim(coalesce(s.translation_de, '')) <> '' "
        "ORDER BY s.id LIMIT %(limit)s", limit=limit)


def store(rows: list[dict[str, Any]],
          vectors: list[list[float]]) -> None:
    with retrieve.connection().cursor() as cursor:
        cursor.executemany(
            'INSERT INTO sentence_embeddings VALUES (%s, %s::vector) ON CONFLICT DO NOTHING',
            [(row['id'], retrieve.vector_literal(v)) for row, v in zip(rows, vectors)])


def main() -> None:
    parser = argparse.ArgumentParser(description='Embed German sentence translations into sentence_embeddings.')
    parser.add_argument('--limit', type=int, default=None, help='stop after this many sentences')
    parser.add_argument('--max-tokens', type=int, default=6_000_000, help='spending guard')
    args = parser.parse_args()
    api = client()
    done, tokens = 0, 0
    
    while args.limit is None or done < args.limit:
        size = BATCH_SIZE if args.limit is None else min(BATCH_SIZE, args.limit - done)
        rows = pending(size)
        if not rows:
            break
        if tokens >= args.max_tokens:
            raise SystemExit(f'stopped at the token guard after {tokens:,} tokens')
        vectors, used = embed_texts(api, [row['translation_de'] for row in rows])
        store(rows, vectors)
        done += len(rows)
        tokens += used
        if done % 10_000 < BATCH_SIZE:
            print(f'{done:>7,} sentences  {tokens:>10,} tokens', flush=True)
    cost = tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS
    print(f'embedded {done:,} sentences, {tokens:,} tokens, about ${cost:.4f}')


if __name__ == '__main__':
    
    main()
