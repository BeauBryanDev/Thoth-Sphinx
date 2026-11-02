from functools import cache
from typing import Any

import cohere
from dotenv import dotenv_values

from evidence_store import retrieve
from evidence_store.paths import ENV_FILE

MODEL = 'rerank-v5.0-fast'
CANDIDATES = 50
MAX_TOKENS_PER_DOC = 512


@cache
def client() -> cohere.ClientV2:
    return cohere.ClientV2(api_key=dotenv_values(ENV_FILE)['COHERE_API_KEY'])


def document_text(row: dict[str, Any]) -> str:
    """Return what the reranker reads: the transliteration, then any German translation."""
    return '\n'.join(part for part in (row['transliteration'], row['translation_de']) if part)


def search_reranked(query: str,
                    signs: str | None = None,
                    translit: str | None = None,
                    limit: int = 10,
                    api: cohere.ClientV2 | None = None,
                    **filters: Any
                    ) -> list[dict[str, Any]]:
    """
    Rerank the top hybrid candidates; 
    hits keep their channel ranks and gain hybrid_rank."""
    hits = retrieve.search(signs, translit, CANDIDATES, **filters)
    
    if not hits:
        return []
    
    rows = {row['id']: row for row in retrieve.query(
        'SELECT id, transliteration, translation_de FROM sentences WHERE id = ANY(%(ids)s)',
        ids=[hit['sentence_id'] for hit in hits])
            }
    response = (api or client()).rerank(
                                    model=MODEL,
                                    query=query,
                                    top_n=limit,
                                    max_tokens_per_doc=MAX_TOKENS_PER_DOC,
                                    documents=[document_text(rows[hit['sentence_id']]) for hit in hits]
                            )
    
    return [{**hits[result.index], 
             'hybrid_rank': result.index + 1,
             'rerank_score': result.relevance_score
             } for result in response.results ]
