import re
from typing import Any

from corpus import retrieve

# Search over english_translations: full text, embeddings, and reciprocal rank fusion of the two.

RRF_K = 60
CANDIDATES = 50
EF_SEARCH = 200
COLUMNS = ('t.id, t.source, t.work, t.author, t.king, t.location, t.section, t.spell, t.title, '
           't.page, t.citation, t.body')
TOP_K = 5
RERANK_CANDIDATES = 15            # distinct spells handed to the reranker
WHOLE_SPELL_CHARS = 3000          # small-to-big cap on the spell text given to the editor
KING_NAMES = {                    # spellings of one king's name across sources and the LLM
    'UNIS': ('Unas', 'Unis', 'Wenis'),          # Piankoff and the LLM write Unas, Allen writes Unis
    'SETY I': ('Sety', 'Seti', 'Sethos'),       # Hypostyle writes Sety, royal_names.json writes Seti
    'RAMESSES II': ('Ramesses', 'Ramses', 'Rameses'),
    'RAMESSES I': ('Ramesses', 'Ramses', 'Rameses'),
    'THUTMOSE III': ('Thutmose', 'Tuthmosis', 'Thutmosis'),
}
UNITS = {                         # what one passage of each source is called; 'Spell' otherwise
    'unas_online': 'Utterance',
    'hypostyle': 'Plate',
    'medinet_habu': 'Plate',
    'birds_catalog': 'Catalog No.',
}


def _king_filter(king: str | None, sources: list[str] | None = None) -> str:
    """SQL condition for an optional pyramid owner and optional sources."""
    return ('AND t.king = %(king)s ' if king else '') + ('AND t.source = ANY(%(sources)s) ' if sources else '')


def search_text(
    query: str,
    king: str | None = None,
    limit: int = TOP_K,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Full-text hits ranked by ts_rank with length normalisation."""
    return retrieve.query(
        f'SELECT {COLUMNS}, ts_rank(t.tsv, q, 1) AS score '
        'FROM english_translations t, book_query(%(query)s) q '
        f'WHERE t.tsv @@ q {_king_filter(king, sources)}'
        'ORDER BY score DESC, t.id LIMIT %(limit)s',
        query=query,
        king=king,
        sources=sources,
        limit=limit,
    )


def search_vector(
    embedding: list[float],
    king: str | None = None,
    limit: int = TOP_K,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Nearest passages by cosine similarity; iterative scan so the king filter cannot starve HNSW."""
    conn = retrieve.connection()
    conn.execute(f'SET hnsw.ef_search = {EF_SEARCH}')
    conn.execute('SET hnsw.iterative_scan = strict_order')
    return retrieve.query(
        f'SELECT {COLUMNS}, 1 - (e.embedding <=> %(v)s::vector) AS score '
        'FROM english_translation_embeddings e JOIN english_translations t ON t.id = e.translation_id '
        f'WHERE true {_king_filter(king, sources)}'
        'ORDER BY e.embedding <=> %(v)s::vector LIMIT %(limit)s',
        v=retrieve.vector_literal(embedding),
        king=king,
        sources=sources,
        limit=limit,
    )


def search_hybrid(
    query: str,
    embedding: list[float],
    king: str | None = None,
    limit: int = TOP_K,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Reciprocal rank fusion of both channels; each hit keeps its cosine similarity."""
    fused: dict[str, dict[str, Any]] = {}
    for channel, hits in (
        ('text', search_text(query, king, CANDIDATES, sources)),
        ('vector', search_vector(embedding, king, CANDIDATES, sources)),
    ):
        for rank, hit in enumerate(hits, start=1):
            row = fused.setdefault(
                hit['id'],
                {**hit, 'score': 0.0, 
                'text_rank': None, 
                'vector_rank': None, 
                'similarity': None},
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


def similarities(ids: list[str], embedding: list[float]) -> dict[str, float]:
    """Exact cosine similarity of the given passages, for hits that only full text found."""
    rows = retrieve.query(
        'SELECT translation_id, 1 - (embedding <=> %(v)s::vector) AS similarity '
        'FROM english_translation_embeddings WHERE translation_id = ANY(%(ids)s)',
        v=retrieve.vector_literal(embedding),
        ids=ids,
    )
    return {r['translation_id']: r['similarity'] for r in rows}


def group_by_spell(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One hit per (source, king, spell): the best-ranked part stands for its spell, order kept."""
    seen: set[tuple[str, str | None, int | None]] = set()
    out = []
    for hit in hits:
        key = (hit['source'], hit['king'], hit['spell'])
        if key not in seen:
            seen.add(key)
            out.append(hit)
    return out


def whole_spell(hit: dict[str, Any], max_chars: int = WHOLE_SPELL_CHARS) -> str:
    """Small-to-big: every part of the hit's spell joined in order, cut around the matching part if too long."""
    rows = retrieve.query(
        'SELECT id, body FROM english_translations '
        'WHERE source = %(source)s AND king IS NOT DISTINCT FROM %(king)s AND spell IS NOT DISTINCT FROM %(spell)s '
        'ORDER BY part',
        source=hit['source'],
        king=hit['king'],
        spell=hit['spell'],
    )
    text = '\n'.join(r['body'] for r in rows) or hit['body']
    if len(text) <= max_chars:
        return text
    start = max(0, text.find(hit['body'][:80]) - max_chars // 4)
    return text[start:start + max_chars]


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    limit: int = TOP_K,
    king: str | None = None,
) -> list[dict[str, Any]]:
    """Cohere rerank of the candidate spells by their matching part; keeps the hybrid rank."""
    from corpus.rerank import MAX_TOKENS_PER_DOC, MODEL, client

    if not hits:
        return []
    # The reranker reads 'Unis' as a worse match for an 'Unas' query, so documents take the query's spelling.
    response = client().rerank(
        model=MODEL,
        query=query,
        top_n=min(limit, len(hits)),
        max_tokens_per_doc=MAX_TOKENS_PER_DOC,
        documents=[respell(f'{h["citation"]}\n{h["body"]}', king, query) for h in hits],
    )
    return [{**hits[r.index], 'hybrid_rank': r.index + 1, 'rerank_score': r.relevance_score}
            for r in response.results]


def name_variants(query: str, king: str | None) -> list[str]:
    """The query once per spelling of the king's name ('Unas', 'Unis', 'Wenis'): sources spell it differently."""
    names = KING_NAMES.get(king or '', ())
    used = [n for n in names if re.search(rf'\b{n}\b', query)]
    if not used:
        return [query]
    return [query] + [re.sub(rf'\b{used[0]}\b', n, query) for n in names if n != used[0]]


def respell(text: str, king: str | None, query: str) -> str:
    """The text with the king's name in the spelling the query uses."""
    names = KING_NAMES.get(king or '', ())
    target = next((n for n in names if re.search(rf'\b{n}\b', query)), None)
    if not target:
        return text
    return re.sub(r'\b(' + '|'.join(n for n in names if n != target) + r')\b', target, text)


def search_variants(
    queries: list[str],
    embeddings: list[list[float]],
    king: str | None = None,
    sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search once per query variant; each part keeps its best fused score and similarity."""
    best: dict[str, dict[str, Any]] = {}
    for query, embedding in zip(queries, embeddings):
        for hit in search_hybrid(query, embedding, king, CANDIDATES * 2, sources):
            kept = best.get(hit['id'])
            if kept is None or hit['score'] > kept['score']:
                if kept and (kept.get('similarity') or 0) > (hit.get('similarity') or 0):
                    hit = {**hit, 'similarity': kept['similarity']}
                best[hit['id']] = hit
    return sorted(best.values(), key=lambda h: (-h['score'], h['id']))


def search_spells(
    query: str,
    embeddings: list[list[float]],
    king: str | None = None,
    limit: int = TOP_K,
    sources: list[str] | None = None,
    use_rerank: bool = False,
) -> list[dict[str, Any]]:
    """Name variants, hybrid on parts, one hit per spell, optional rerank; embeddings follow name_variants(query)."""
    spells = group_by_spell(search_variants(name_variants(query, king), embeddings, king, sources))
    return rerank(query, spells[:RERANK_CANDIDATES], limit, king) if use_rerank else spells[:limit]
