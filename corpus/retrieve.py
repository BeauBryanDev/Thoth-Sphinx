import os
import threading
from typing import Any

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from corpus.paths import ENV_FILE

# One connection per thread: a shared one serializes the parallel tool calls (app.llm.client.TOOL_POOL).
_local = threading.local()


def database_url() -> str:
    """EVIDENCE_DB from the environment, else from .env. Never hard-coded."""
    url = os.environ.get('EVIDENCE_DB') or dotenv_values(ENV_FILE).get('EVIDENCE_DB')
    if not url:
        # An OperationalError, so callers treat it like a database that isn't running.
        raise psycopg.OperationalError('EVIDENCE_DB is not set. Add it to .env.')
    return url


def connection() -> psycopg.Connection[dict[str, Any]]:
    """This thread's connection, opened on first use."""
    conn = getattr(_local, 'connection', None)
    if conn is None or conn.closed:
        conn = psycopg.connect(
            database_url(),
            autocommit=True,
            row_factory=dict_row,
        )
        _local.connection = conn

    return conn


def query(sql: str, **params: Any) -> list[dict[str, Any]]:
    
    return connection().execute(sql, params).fetchall()


def vector_literal(embedding: list[float] | None) -> str | None:
    
    return None if embedding is None else '[' + ','.join(map(str, embedding)) + ']'


def date_terms(date: str | None, 
               include_undated: bool = True
               ) -> list[str] | None:
    """Return the document_dates terms for a filter, or None when there is no filter."""
    if date is None:
        return None
    
    if not query('SELECT 1 FROM date_periods() WHERE period = %(date)s', date=date):
        raise ValueError(f'unknown date period: {date!r}')
    
    return [date, 'undated'] if include_undated else [date]


def search(signs: str | None = None,
           translit: str | None = None,
           limit: int = 10,
           exclude_text: str | None = None,
           exclude_norm: str | None = None,
           embedding: list[float] | None = None,
           date: str | None = None,
           include_undated: bool = True
           ) -> list[dict[str, Any]]:
    
    return query(
        'SELECT * FROM search_hybrid(%(signs)s, %(translit)s, %(limit)s, %(exclude_text)s, '
        '%(exclude_norm)s, %(embedding)s::vector, %(dates)s::text[])',
        signs=signs,
        translit=translit,
        limit=limit,
        exclude_text=exclude_text,
        exclude_norm=exclude_norm,
        embedding=vector_literal(embedding),
        dates=date_terms(date, include_undated))


def search_channel(channel: str,
                   text: str,
                   limit: int = 10,
                   exclude_text: str | None = None,
                   exclude_norm: str | None = None,
                   date: str | None = None,
                   include_undated: bool = True
                   ) -> list[dict[str, Any]]:
    
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


def search_embedding(embedding: list[float] | None,
                     limit: int = 10,
                     exclude_text: str | None = None,
                     exclude_norm: str | None = None,
                     date: str | None = None,
                     include_undated: bool = True
                     ) -> list[dict[str, Any]]:
    
    return query(
        'SELECT * FROM search_embedding(%(embedding)s::vector, %(limit)s, %(exclude_text)s, '
        '%(exclude_norm)s, %(dates)s::text[])',
        embedding=vector_literal(embedding),
        limit=limit,
        exclude_text=exclude_text,
        exclude_norm=exclude_norm,
        dates=date_terms(date, include_undated))


def search_trigram(translit: str,
                   limit: int = 10,
                   exclude_text: str | None = None,
                   exclude_norm: str | None = None
                   ) -> list[dict[str, Any]]:
    
    return query(
        'SELECT * FROM search_trigram(%(translit)s, %(limit)s, %(exclude_text)s, '
        '%(exclude_norm)s)',
        translit=translit,
        limit=limit,
        exclude_text=exclude_text,
        exclude_norm=exclude_norm)


def context(sentence_id: str, window: int = 2) -> list[dict[str, Any]]:
    return query('SELECT * FROM sentence_context(%(sentence_id)s, %(window)s)',
                 sentence_id=sentence_id,
                 window=window)


def lookup_lemma(text: str, limit: int = 10) -> list[dict[str, Any]]:
    return query('SELECT * FROM lookup_lemma(%(text)s, %(limit)s)',
                 text=text, limit=limit)


def lookup_signs(keys: list[str], me_only: bool = False) -> list[dict[str, Any]]:
    return query('SELECT * FROM lookup_signs(%(keys)s, %(me_only)s)',
                 keys=keys, me_only=me_only)
