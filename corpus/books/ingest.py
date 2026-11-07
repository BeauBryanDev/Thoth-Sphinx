import argparse
import json
from pathlib import Path
from typing import Any

from corpus import embed, retrieve
from corpus.books.book_inspector import MANIFEST
from corpus.books.chunk import MD_DIR, Chunk, chunk_book, read_book
from corpus.paths import CODE_ROOT

# Loads the plain chunks of the Markdown books into book_chunks and embeds what is not embedded yet.
# Re-running replaces changed chunks; embeddings of chunks whose id and text are unchanged are kept.

SCHEMA = CODE_ROOT / 'db' / 'books.sql'
VARIANT = 'plain'                 # the only chunking kept; semantic cuts measured no better (2026-11-04)
BATCH = 200                       # 200 chunks of at most ~670 tokens stays under the request token cap


def apply_schema() -> None:
    """Create the book tables, indexes and book_query() if missing."""
    retrieve.connection().execute(SCHEMA.read_text())


def books_meta(manifest: Path) -> dict[str, dict[str, Any]]:
    """doc_id -> topic, title and author from the books manifest."""
    return {
        b['doc_id']: {
            'topic': b.get('topic'),
            'title': b.get('title'),
            'author': b.get('author') or None,
        }
        for b in json.loads(manifest.read_text())['books']
    }


def build(md: Path) -> list[Chunk]:
    """Every chunk of every book."""
    return [c for p in sorted(md.glob('*.md')) for c in chunk_book(read_book(p))]


def store(chunks: list[Chunk], meta: dict[str, dict[str, Any]]) -> None:
    """Replace the stored chunks. Chunks whose id and text are unchanged keep their embeddings."""
    conn = retrieve.connection()
    rows = [
        (
            f'{VARIANT}:{c.doc_id}:{c.ordinal}',
            VARIANT,
            c.doc_id,
            meta.get(c.doc_id, {}).get('topic'),
            meta.get(c.doc_id, {}).get('title'),
            meta.get(c.doc_id, {}).get('author'),
            c.section,
            c.ordinal,
            c.header,
            c.text,
            c.tokens,
            c.page_start,
            c.page_end,
            c.quality,
        )
        for c in chunks
    ]
    with conn.transaction():
        old = {
            r['id']: r['body']
            for r in conn.execute('SELECT id, body FROM book_chunks WHERE variant = %s', (VARIANT,)).fetchall()
        }
        ids = {r[0] for r in rows}
        stale = [r[0] for r in rows if old.get(r[0]) not in (None, r[9])] + [i for i in old if i not in ids]
        conn.execute('DELETE FROM book_chunks WHERE id = ANY(%s)', (stale,))
        with conn.cursor() as cursor:
            cursor.executemany(
                'INSERT INTO book_chunks (id, variant, doc_id, topic, title, author, section, ordinal, header, '
                'body, tokens, page_start, page_end, quality) '
                'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) '
                'ON CONFLICT (id) DO UPDATE SET topic = EXCLUDED.topic, title = EXCLUDED.title, '
                'author = EXCLUDED.author, section = EXCLUDED.section, header = EXCLUDED.header, '
                'quality = EXCLUDED.quality',
                rows,
            )
    print(f'{len(rows):,} chunks stored ({len(stale):,} replaced or removed)')


def embed_pending(max_tokens: int) -> None:
    """Embed header + body of every chunk that has no vector yet."""
    pending = retrieve.query(
        'SELECT c.id, c.header, c.body, c.tokens FROM book_chunks c '
        'LEFT JOIN book_chunk_embeddings e ON e.chunk_id = c.id '
        'WHERE c.variant = %(variant)s AND e.chunk_id IS NULL ORDER BY c.id',
        variant=VARIANT,
    )
    need = sum(r['tokens'] + 30 for r in pending)
    print(f'{len(pending):,} chunks to embed, about {need:,} tokens '
          f'(${need / 1e6 * embed.PRICE_PER_MILLION_TOKENS:.3f})')
    if not pending:
        return
    if need > max_tokens:
        raise SystemExit(f'over the --max-tokens guard ({max_tokens:,}); raise it to proceed')
    api = embed.client()
    used = 0
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        vectors, tokens = embed.embed_texts(api, [f'{r["header"]}\n\n{r["body"]}' for r in batch])
        used += tokens
        with retrieve.connection().cursor() as cursor:
            cursor.executemany(
                'INSERT INTO book_chunk_embeddings VALUES (%s, %s::vector) ON CONFLICT DO NOTHING',
                [(r['id'], retrieve.vector_literal(v)) for r, v in zip(batch, vectors)],
            )
        print(f'  {i + len(batch):,} / {len(pending):,}', flush=True)
    print(f'embedded with {used:,} tokens, about ${used / 1e6 * embed.PRICE_PER_MILLION_TOKENS:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Load the plain chunks of the Markdown books into Postgres and embed them.')
    parser.add_argument('--md', type=Path, default=MD_DIR, help=f'default {MD_DIR}')
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help=f'default {MANIFEST}')
    parser.add_argument('--no-embed', action='store_true', help='load chunks only')
    parser.add_argument('--max-tokens', type=int, default=4_000_000, help='spending guard for embeddings')
    args = parser.parse_args()

    apply_schema()
    store(build(args.md), books_meta(args.manifest))
    if not args.no_embed:
        embed_pending(args.max_tokens)
    retrieve.connection().execute('ANALYZE book_chunks')
    retrieve.connection().execute('ANALYZE book_chunk_embeddings')


if __name__ == '__main__':
    main()
