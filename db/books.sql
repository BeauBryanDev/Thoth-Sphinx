-- Book chunks for the Thoth agent (evidence_store/books). Separate from the AAEW sentence corpus.
-- variant names the chunking; only 'plain' is kept (semantic cuts measured no better, 2026-11-04).
CREATE TABLE IF NOT EXISTS book_chunks (
    id         text PRIMARY KEY,                -- '<variant>:<doc_id>:<ordinal>'
    variant    text NOT NULL,
    doc_id     text NOT NULL,
    topic      text,
    section    text NOT NULL,
    ordinal    integer NOT NULL,
    header     text NOT NULL,                   -- 'Book › Section › p. 12–13', embedded with the body
    body       text NOT NULL,
    tokens     integer NOT NULL,
    page_start text,
    page_end   text,
    quality    real,
    tsv        tsvector GENERATED ALWAYS AS (
                   setweight(to_tsvector('english', section), 'B')
                   || setweight(to_tsvector('english', body), 'A')
               ) STORED
);
-- Book title and author from the manifest, as written there (not yet cleaned for citations).
ALTER TABLE book_chunks ADD COLUMN IF NOT EXISTS title text;
ALTER TABLE book_chunks ADD COLUMN IF NOT EXISTS author text;
CREATE INDEX IF NOT EXISTS book_chunks_tsv_idx ON book_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS book_chunks_variant_idx ON book_chunks (variant, doc_id);

-- text-embedding-3-small over header + body. See evidence_store/books/ingest.py.
CREATE TABLE IF NOT EXISTS book_chunk_embeddings (
    chunk_id  text PRIMARY KEY REFERENCES book_chunks (id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS book_chunk_embeddings_hnsw_idx
    ON book_chunk_embeddings USING hnsw (embedding vector_cosine_ops);

-- A question as an OR of its stemmed words: plainto_tsquery would demand every word.
CREATE OR REPLACE FUNCTION book_query(p_text text) RETURNS tsquery
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(
        to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')),
        ''::tsquery
    )
    FROM unnest(tsvector_to_array(to_tsvector('english', p_text))) AS lexeme
$$;
