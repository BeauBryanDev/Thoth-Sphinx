import argparse

from corpus import embed, retrieve
from corpus.paths import CODE_ROOT
from corpus.translations.extract import Passage, passages

# Loads the spell passages into english_translations and embeds what is not embedded yet.
# Re-running replaces changed passages; embeddings of passages whose id and text are unchanged are kept.

SCHEMA = CODE_ROOT / 'db' / 'translations.sql'
BATCH = 200


def apply_schema() -> None:
    """Create the translation tables and indexes if missing."""
    retrieve.connection().execute(SCHEMA.read_text())


def store(items: list[Passage]) -> None:
    """Replace the stored passages. Unchanged passages keep their embeddings."""
    conn = retrieve.connection()
    rows = [
        (p.id, p.source, p.work, p.author, p.king, p.location, p.section, p.spell, p.title,
         p.part, p.page, p.citation, p.body, p.tokens)
        for p in items
    ]
    with conn.transaction():
        old = {r['id']: r['body'] for r in conn.execute('SELECT id, body FROM english_translations').fetchall()}
        ids = {r[0] for r in rows}
        stale = [r[0] for r in rows if old.get(r[0]) not in (None, r[12])] + [i for i in old if i not in ids]
        conn.execute('DELETE FROM english_translations WHERE id = ANY(%s)', (stale,))
        with conn.cursor() as cursor:
            cursor.executemany(
                'INSERT INTO english_translations (id, source, work, author, king, location, section, spell, '
                'title, part, page, citation, body, tokens) '
                'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) '
                'ON CONFLICT (id) DO UPDATE SET location = EXCLUDED.location, section = EXCLUDED.section, '
                'title = EXCLUDED.title, page = EXCLUDED.page, citation = EXCLUDED.citation',
                rows,
            )
    print(f'{len(rows):,} passages stored ({len(stale):,} replaced or removed)')


def embed_pending(max_tokens: int) -> None:
    """Embed citation + body of every passage that has no vector yet."""
    pending = retrieve.query(
        'SELECT t.id, t.citation, t.body, t.tokens FROM english_translations t '
        'LEFT JOIN english_translation_embeddings e ON e.translation_id = t.id '
        'WHERE e.translation_id IS NULL ORDER BY t.id',
    )
    need = sum(r['tokens'] + 40 for r in pending)
    print(f'{len(pending):,} passages to embed, about {need:,} tokens '
          f'(${need / 1e6 * embed.PRICE_PER_MILLION_TOKENS:.4f})')
    if not pending:
        return
    if need > max_tokens:
        raise SystemExit(f'over the --max-tokens guard ({max_tokens:,}); raise it to proceed')
    api = embed.client()
    used = 0
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        vectors, tokens = embed.embed_texts(api, [f'{r["citation"]}\n\n{r["body"]}' for r in batch])
        used += tokens
        with retrieve.connection().cursor() as cursor:
            cursor.executemany(
                'INSERT INTO english_translation_embeddings VALUES (%s, %s::vector) ON CONFLICT DO NOTHING',
                [(r['id'], retrieve.vector_literal(v)) for r, v in zip(batch, vectors)],
            )
        print(f'  {i + len(batch):,} / {len(pending):,}', flush=True)
    print(f'embedded with {used:,} tokens, about ${used / 1e6 * embed.PRICE_PER_MILLION_TOKENS:.4f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Load the translation passages into Postgres and embed them.')
    parser.add_argument('--no-embed', action='store_true', help='load passages only')
    parser.add_argument('--max-tokens', type=int, default=500_000, help='spending guard for embeddings')
    args = parser.parse_args()

    apply_schema()
    store(passages())
    if not args.no_embed:
        embed_pending(args.max_tokens)
    retrieve.connection().execute('ANALYZE english_translations')
    retrieve.connection().execute('ANALYZE english_translation_embeddings')


if __name__ == '__main__':
    main()
