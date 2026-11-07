from typing import Any

from corpus import retrieve

# Search over book_chunks: full text, embeddings, and reciprocal rank fusion of the two.

RRF_K = 60
CANDIDATES = 50                   # hits taken from each channel before fusion
EF_SEARCH = 200
COLUMNS = 'c.id, c.doc_id, c.title, c.author, c.section, c.header, c.body, c.page_start, c.page_end, c.topic'
TOP_K = 5                         # passages handed to the agent; measured answered@5 0.92 (hybrid, plain)


def search_text(query: str, variant: str = 'plain', limit: int = TOP_K) -> list[dict[str, Any]]:
    """Full-text hits: any of the question's stemmed words, ranked by ts_rank with length normalisation."""
    return retrieve.query(
        f'SELECT {COLUMNS}, ts_rank(c.tsv, q, 1) AS score '
        'FROM book_chunks c, book_query(%(query)s) q '
        'WHERE c.variant = %(variant)s AND c.tsv @@ q '
        'ORDER BY score DESC, c.id LIMIT %(limit)s',
        query=query,
        variant=variant,
        limit=limit,
    )


def search_vector(embedding: list[float], 
variant: str = 'plain',
 limit: int = TOP_K
 )  -> list[dict[str, Any]]:
    """Nearest chunks by cosine similarity; HNSW with an iterative scan so the variant filter cannot starve it."""
    conn = retrieve.connection()
    conn.execute(f'SET hnsw.ef_search = {EF_SEARCH}')
    conn.execute('SET hnsw.iterative_scan = strict_order')
    return retrieve.query(
        f'SELECT {COLUMNS}, 1 - (e.embedding <=> %(v)s::vector) AS score '
        'FROM book_chunk_embeddings e JOIN book_chunks c ON c.id = e.chunk_id '
        'WHERE c.variant = %(variant)s '
        'ORDER BY e.embedding <=> %(v)s::vector LIMIT %(limit)s',
        v=retrieve.vector_literal(embedding),
        variant=variant,
        limit=limit,
    )


def search_hybrid(
    query: str,
    embedding: list[float],
    variant: str = 'plain',
    limit: int = TOP_K,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion of both channels; each hit keeps its rank in each channel and its cosine similarity."""
    fused: dict[str, dict[str, Any]] = {}
    for channel, hits in (
        ('text', search_text(query, variant, CANDIDATES)),
        ('vector', search_vector(embedding, variant, CANDIDATES)),
    ):
        for rank, hit in enumerate(hits, start=1):
            row = fused.setdefault(
                hit['id'],
                {**hit, 'score': 0.0, 'text_rank': None, 'vector_rank': None, 'similarity': None},
            )
            row[f'{channel}_rank'] = rank
            row['score'] += 1 / (RRF_K + rank)
            if channel == 'vector':
                row['similarity'] = hit['score']
    top = sorted(fused.values(), key=lambda r: (-r['score'], r['id']))[:limit]
    missing = [r['id'] for r in top if r['similarity'] is None]
    if missing:
        found = similarities(missing, embedding)
        for r in top:
            r['similarity'] = found.get(r['id'], r['similarity'])
    return top


def similarities(chunk_ids: list[str], embedding: list[float]) -> dict[str, float]:
    """Exact cosine similarity of the given chunks, for hits that only full text found."""
    rows = retrieve.query(
        'SELECT chunk_id, 1 - (embedding <=> %(v)s::vector) AS similarity '
        'FROM book_chunk_embeddings WHERE chunk_id = ANY(%(ids)s)',
        v=retrieve.vector_literal(embedding),
        ids=chunk_ids,
    )
    return {r['chunk_id']: r['similarity'] for r in rows}
