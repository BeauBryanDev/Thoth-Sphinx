"""Client for the retrieval functions in db/retrieval.sql."""
import os

import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.environ.get('EVIDENCE_DB',
                              'postgresql://evidence:evidence@127.0.0.1:5446/evidence')
_connection = None


def connection():
    global _connection
    if _connection is None or _connection.closed:
        _connection = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
    return _connection


def query(sql, **params):
    return connection().execute(sql, params).fetchall()


def vector_literal(embedding):
    return None if embedding is None else '[' + ','.join(map(str, embedding)) + ']'


def date_terms(date, include_undated=True):
    """Return the document_dates terms for a filter, or None when there is no filter."""
    if date is None:
        return None
    if not query('SELECT 1 FROM date_periods() WHERE period = %(date)s', date=date):
        raise ValueError(f'unknown date period: {date!r}')
    return [date, 'undated'] if include_undated else [date]


def search(signs=None, 
           translit=None, 
           limit=10, 
           exclude_text=None,
           exclude_norm=None,
           embedding=None,
           date=None,
           include_undated=True):
    
    return query(
        'SELECT * FROM search_hybrid(%(signs)s, %(translit)s, %(limit)s, %(exclude_text)s, '
        '%(exclude_norm)s, %(embedding)s::vector, %(dates)s::text[])',
        signs=signs, translit=translit, 
        limit=limit, exclude_text=exclude_text,
        exclude_norm=exclude_norm, 
        embedding=vector_literal(embedding),
        dates=date_terms(date, include_undated))


def search_channel(channel, 
                   text, 
                   limit=10,
                   exclude_text=None,
                   exclude_norm=None,
                   date=None,
                   include_undated=True):
    
    grams = 'sign_grams' if channel == 'sign' else 'translit_grams'
    
    return query(
        f'SELECT * FROM search_channel(%(channel)s, {grams}(%(text)s), %(limit)s, '
        '%(exclude_text)s, %(exclude_norm)s, %(dates)s::text[])',
        channel=channel, 
        text=text, 
        limit=limit,
        exclude_text=exclude_text, 
        exclude_norm=exclude_norm,
        dates=date_terms(date, include_undated))


def search_embedding(embedding,
                     limit=10,
                     exclude_text=None,
                     exclude_norm=None,
                     date=None,
                     include_undated=True):
    
    return query(
        'SELECT * FROM search_embedding(%(embedding)s::vector, %(limit)s, %(exclude_text)s, '
        '%(exclude_norm)s, %(dates)s::text[])',
        embedding=vector_literal(embedding),
        limit=limit,
        exclude_text=exclude_text,
        exclude_norm=exclude_norm,
        dates=date_terms(date, include_undated))


def search_trigram(translit,
                   limit=10,
                   exclude_text=None,
                   exclude_norm=None):
    
    return query(
        'SELECT * FROM search_trigram(%(translit)s, %(limit)s, %(exclude_text)s, '
        '%(exclude_norm)s)',
        translit=translit,
        limit=limit, 
        exclude_text=exclude_text, 
        exclude_norm=exclude_norm)


def context(sentence_id, window=2):
    return query('SELECT * FROM sentence_context(%(sentence_id)s, %(window)s)',
                 sentence_id=sentence_id,
                 window=window)


def lookup_lemma(text, limit=10):
    return query('SELECT * FROM lookup_lemma(%(text)s, %(limit)s)',
                 text=text, limit=limit)
