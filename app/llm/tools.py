import logging
from typing import Any

from app.llm import mdc
from evidence_store import confidence, rerank, retrieve
from evidence_store.books import search as book_search

CANDIDATES = 10
MAX_LIMIT = 20
MAX_WINDOW = 5
MAX_LINKS = 30
TLA_CREDIT = 'Thesaurus Linguae Aegyptiae, BBAW. CC BY-SA 4.0.'
VERDICT_GUIDE = {
    'strong': 'A real parallel is likely. Strong hits were right about 82% of the time in testing.',
    'possible': 'Leads only. Possible hits were right about 30% of the time, so check before use.',
    'none': 'No parallel found. This is normal: most sentences have none. Do not cite these hits.',
    'unrated': 'Signs alone cannot be rated. Add a transliteration to get a verdict.',
    'too_short': 'Too few words to rate. Hits may share them by chance, so read their context.',
}
RELIABILITY_GUIDE = {
    'verified': 'Confirmed against a monument or an image.',
    'reference': 'From a cited reference work. Dates and details can vary between scholars.',
    'curated': 'Uncited background knowledge. Do not present it as evidence.',
    'derived': 'Worked out from a name or another link, not stated by a source.',
    'unverified': 'Unchecked, and possibly generated. Treat it as a hint.',
}
BOOK_SUPPORT_FLOOR = 0.5   # best vector score below this: the books likely do not cover the question
MAX_BOOK_QUERY = 300
BOOK_GUIDE = {
    True: 'The books seem to cover this. Cite the passage you use by its citation; never invent pages.',
    False: 'The books do not seem to cover this question. Say so, and answer from general knowledge only '
           'if you flag it as such.',
}
logger = logging.getLogger(__name__)


def clamp(value: int, 
          low: int, 
          high: int) -> int:
    
    return max(low, min(int(value), high))


def required(value: str | None, label: str) -> str:
    if not value or not value.strip():
        raise ValueError(f'{label} is required.')
    
    return value.strip()


def sentence_details(ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return {sentence id: citable text and source} for a list of sentences."""
    rows = retrieve.query(
        'SELECT s.id, s.transliteration, s.translation_de, s.translation_en, s.has_lacuna, '
        'd.name AS text_name, d.corpus, d.date, d.find_spot '
        'FROM sentences s JOIN documents d ON d.id = s.text_id WHERE s.id = ANY(%(ids)s)', ids=ids)
    return {row['id']: {key: value for key, value in row.items() if key != 'id'} for row in rows}


def rated(hits: list[dict[str, Any]],
          translit: str | None,
          signs: str | None
          ) -> list[str | None]:
    """Return each hit's label, or None when there is no transliteration to compare."""
    if not translit:
        return [None] * len(hits)
    
    query_words = confidence.tokens(translit)
    query_signs = confidence.sign_tokens(signs or '')
    hits = confidence.enrich(hits)
    words = set(query_words) | {word for hit in hits for word in hit['tokens']}
    weights = confidence.word_weights(words) if words else {}
    
    return [confidence.rate_hit(confidence.hit_signals(query_words, 
                                                       query_signs, 
                                                       hit, weights),
                                len(query_words)) for hit in hits]


def reranked_hits(translit: str,
                  signs: str | None,
                  limit: int,
                  filters: dict[str, Any]
                  ) -> tuple[list[dict[str, Any]], str]:
    """Return (hits, status). A Cohere failure falls back to hybrid order and says so."""
    try:
        return rerank.search_reranked(translit, signs,
                                      translit, limit, **filters), 'applied'
    except Exception as error:  # any provider failure must leave search working
        logger.warning('rerank failed: %s', error)
        
        return retrieve.search(signs, translit, limit, **filters), 'failed, using hybrid order'


def matched_by(hit: dict[str, Any]) -> list[str]:
    channels = (('signs', 'sign_rank'),
                ('transliteration', 'translit_rank'),
                ('meaning', 'embedding_rank'))
    
    return [name for name, key in channels if hit.get(key)]


def search_verdict(translit: str | None, 
                   ratings: list[str | None]
                   ) -> str:
    
    if not translit:
        return 'unrated'
    if len(confidence.tokens(translit)) < confidence.MIN_RATED_WORDS:
        return 'too_short'
    return confidence.overall_rating(set(ratings))


def search_parallels(signs: str | None = None,
                     translit: str | None = None,
                     date: str | None = None,
                     include_undated: bool = True,
                     limit: int = 5,
                     use_rerank: bool = False
                     ) -> dict[str, Any]:
    
    # Claude writes MdC ASCII (kA-nxt); the corpus is Unicode Leiden (kꜣ-nḫt). Either is accepted.
    translit = mdc.translit_query(translit) or None
    if not (signs and signs.strip()) and not translit:
        raise ValueError('Give signs, a transliteration, or both.')
    
    limit = clamp(limit, 1, MAX_LIMIT)
    
    filters: dict[str, Any] = {'date': date, 
                               'include_undated': include_undated
                               }
    size = max(limit, CANDIDATES)
    status = 'off'
    if use_rerank and translit:
        hits, status = reranked_hits(translit, signs, size, filters)
        
    else:
        hits = retrieve.search(signs, translit, size, **filters)
        
    ratings = rated(hits, translit, signs)
    order: dict[str | None, int] = {'strong': 0, 
                                    'possible': 1, 
                                    'weak': 2, None: 3
                                    }
    ranked = sorted(zip(hits, ratings),
                    key=lambda pair: order[pair[1]])[:limit]
    
    details = sentence_details([hit['sentence_id'] for hit, _ in ranked])
    verdict = search_verdict(translit, ratings)
    
    return {'verdict': verdict,
            'guide': VERDICT_GUIDE[verdict],
            'query_translit': translit,
            'rerank': status,
            'hits': [{
                'sentence_id': hit['sentence_id'], 
                'rating': rating or 'unrated',
                'matched_by': matched_by(hit), **details[hit['sentence_id']]
                } for hit, rating in ranked],
            'credit': TLA_CREDIT
            }


def lookup_lemma(text: str,
                 limit: int = 10
                 ) -> dict[str, Any]:
    
    text = required(text, 'A word')
    rows = retrieve.lookup_lemma(text, clamp(limit, 1, MAX_LIMIT))
    
    return {'found': len(rows),
            'lemmas': [{**{key: row[key] for key in ('id', 'name', 'pos', 'gloss_en', 'gloss_de',
                                                     'sign_seq')
                           },
            'obsolete': row['revision_state'] == 'published-obsolete'} for row in rows],
            'credit': TLA_CREDIT
            }


def get_context(sentence_id: str,
                window: int = 2
                ) -> dict[str, Any]:
    
    sentence_id = required(sentence_id, 'A sentence id')
    rows = retrieve.context(sentence_id, clamp(window, 0, MAX_WINDOW))
    if not rows:
        raise ValueError(f'Unknown sentence id: {sentence_id}')
    source = sentence_details([sentence_id])[sentence_id]
    
    return {'text_name': source['text_name'], 
            'date': source['date'],
            'find_spot': source['find_spot'],
            'sentences': [{key: row[key] for key in ('id', 'offset_from_hit', 'transliteration',
                                                     'translation_de', 'translation_en',
                                                     'has_lacuna')
                           } for row in rows],
            'credit': TLA_CREDIT
            }


def list_periods() -> dict[str, Any]:
    """Return the periods and a note about how they are used."""
    return {'periods': retrieve.query('SELECT period, sentences FROM date_periods()'),
            'note': 'Filters include child periods, and 1,098 undated sentences are kept unless '
                    'excluded. Century terms are not linked to dynasties.'}


def example_sentences(where: str, 
                      limit: int,
                      **params: Any
                      ) -> list[dict[str, Any]]:
    """Return citable example sentences, preferring short ones with a translation."""
    rows = retrieve.query(
        "SELECT DISTINCT s.id, s.word_count FROM sentences s "
        f"{where} AND btrim(coalesce(s.translation_de, '')) <> '' AND s.word_count >= 3 "
        'ORDER BY s.word_count, s.id LIMIT %(limit)s', limit=limit, **params)
    details = sentence_details([row['id'] for row in rows])
    
    return [{'sentence_id': row['id'],
             **details[row['id']]
             } for row in rows]


def corpus_mentions(item: dict[str, Any],
                    examples: int
                    ) -> dict[str, Any]:
    """Return how an entity reaches real texts, by lemma, find spot and date."""
    found: dict[str, Any] = {}
    if item['lemma_ids']:
        count = retrieve.query('SELECT count(DISTINCT sentence_id) AS n FROM tokens '
                               'WHERE lemma_id = ANY(%(ids)s)', ids=item['lemma_ids'])[0]['n']
        found['by_word'] = {'sentences': count, 'examples': example_sentences(
            'JOIN tokens t ON t.sentence_id = s.id WHERE t.lemma_id = ANY(%(ids)s)', examples,
            ids=item['lemma_ids'])}
        
    spots = item['attributes'].get('corpus_find_spots') or []
    if spots:
        count = retrieve.query('SELECT count(*) AS n FROM sentences s JOIN documents d '
                               'ON d.id = s.text_id WHERE d.find_spot = ANY(%(spots)s)',
                               spots=spots)[0]['n']
        found['by_find_spot'] = {
            'sentences': count,
            'covers_whole_site': item['attributes']['find_spot_covers_whole_site'],
            'examples': example_sentences('JOIN documents d ON d.id = s.text_id '
                                          'WHERE d.find_spot = ANY(%(spots)s)', examples,
                                          spots=spots)}
    path = item['attributes'].get('corpus_path')
    if path:
        prefix = path + '%'
        count = retrieve.query('SELECT count(*) AS n FROM sentences s JOIN documents d '
                               'ON d.id = s.text_id WHERE d.path LIKE %(prefix)s',
                               prefix=prefix)[0]['n']
        found['inscribed_here'] = {
            'sentences': count,
            'examples': example_sentences('JOIN documents d ON d.id = s.text_id '
                                          'WHERE d.path LIKE %(prefix)s',
                                          examples, prefix=prefix)}
    if item['date_terms']:
        found['by_date'] = {'date_terms': item['date_terms'],
                            'sentences': retrieve.query(
                            'SELECT count(*) AS n FROM sentences s WHERE s.text_id IN (SELECT document_id FROM '
                            'document_dates WHERE term = ANY(%(terms)s))',
                            terms=item['date_terms'])[0]['n']
                            }
        
    return found


def entity_links(entity_id: str) -> list[dict[str, Any]]:
    """Return an entity's links in both directions, naming the entity at the other end."""
    rows = retrieve.query(
        "SELECT CASE WHEN l.source_id = %(id)s THEN 'outgoing' ELSE 'incoming' END AS direction, "
        'l.relation, o.id, o.name, o.type, l.detail, l.source, l.reliability '
        'FROM entity_links l JOIN entities o '
        'ON o.id = CASE WHEN l.source_id = %(id)s THEN l.target_id ELSE l.source_id END '
        'WHERE l.source_id = %(id)s OR l.target_id = %(id)s '
        'ORDER BY l.relation, o.name', id=entity_id)
    
    return rows


def entity_result(item: dict[str, Any],
                  examples: int
                  ) -> dict[str, Any]:
    
    links = entity_links(item['id'])
    
    return {key: item[key] for key in ('id', 'type', 'name', 'aliases', 'start_year', 'end_year',
                                       'attributes', 'source', 'reliability')} | {
        'reliability_guide': RELIABILITY_GUIDE[item['reliability']],
        'links': links[:MAX_LINKS], 'links_total': len(links),
        'corpus': corpus_mentions(item, examples)}


def lookup_entity(name: str,
                  entity_type: str | None = None,
                  limit: int = 3,
                  examples: int = 3
                  ) -> dict[str, Any]:
                  
    name = required(name, 'A name')
    matches = retrieve.query(
        'SELECT id FROM find_entities(%(name)s, %(type)s, %(limit)s)', 
        name=name, type=entity_type,
        limit=clamp(limit, 1, 10))
    
    ids = [row['id'] for row in matches]
    rows = {row['id']: row for row in retrieve.query('SELECT * FROM entities WHERE id = ANY(%(ids)s)',
                                                     ids=ids)}
    return {'found': len(ids),
            'entities': [entity_result(rows[key], clamp(examples, 0, 10)) for key in ids],
            'note': 'Only corpus sentences are evidence. Other fields are background, rated by '
                    'reliability.',
            'credit': TLA_CREDIT
            }


def search_books(query: str, limit: int = book_search.TOP_K) -> dict[str, Any]:
    """Hybrid search (full text + meaning) over the Egyptology books; vector failure falls back to full text."""
    query = required(query, 'A question')[:MAX_BOOK_QUERY]
    limit = clamp(limit, 1, book_search.TOP_K)
    try:
        # Lazy: the OpenAI SDK (embeddings only) loads when books are first searched, not at app start.
        from evidence_store import embed

        vector = embed.embed_texts(embed.client(), [query])[0][0]
        hits = book_search.search_hybrid(query, vector, limit=limit)
        best = book_search.search_vector(vector, limit=1)
        mode = 'hybrid'
    except Exception as error:  # an embedding outage must leave book search working
        logger.warning('book search embedding failed: %s', error)
        hits = book_search.search_text(query, limit=limit)
        best = []
        mode = 'full text only (meaning search unavailable)'
    top = best[0]['score'] if best else None
    supported = bool(hits) and (top is None or top >= BOOK_SUPPORT_FLOOR)
    return {
        'found': len(hits),
        'mode': mode,
        'no_book_support': not supported,
        'best_meaning_score': round(top, 3) if top is not None else None,
        'guide': BOOK_GUIDE[supported],
        'passages': [
            {
                'chunk_id': h['id'],
                'book': h.get('title'),
                'author': h.get('author'),
                'section': h['section'],
                'pages': h['page_start'] if h['page_start'] == h['page_end'] else f"{h['page_start']}–{h['page_end']}",
                'citation': h['header'],
                'similarity': round(h['similarity'], 3) if h.get('similarity') is not None else None,
                'matched_by': [name for name, key in (('words', 'text_rank'), ('meaning', 'vector_rank')) if h.get(key)]
                or ['words'],
                'text': h['body'],
            }
            for h in hits
        ],
    }
