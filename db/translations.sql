-- English translations of whole spells (Allen's Pyramid Texts, Faulkner's Coffin Texts), one row per passage.
-- Searched by the agent's search_translations tool. Uses book_query() from db/books.sql.
CREATE TABLE IF NOT EXISTS english_translations (
    id        text PRIMARY KEY,                 -- '<source>:<king>:<spell>:<part>'
    source    text NOT NULL,                    -- 'pyramid_texts' | 'coffin_texts'
    work      text NOT NULL,
    author    text NOT NULL,
    king      text,                             -- the pyramid's owner as Allen spells it (UNIS, TETI, ...)
    location  text,                             -- the wall, e.g. 'ANTECHAMBER, WEST GABLE'
    section   text,
    spell     integer,
    title     text,
    part      integer NOT NULL,
    page      text,
    citation  text NOT NULL,                    -- embedded with the body
    body      text NOT NULL,
    tokens    integer NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (
                  setweight(to_tsvector('english', coalesce(title, '')), 'B')
                  || setweight(to_tsvector('english', body), 'A')
              ) STORED
);
CREATE INDEX IF NOT EXISTS english_translations_tsv_idx ON english_translations USING gin (tsv);
CREATE INDEX IF NOT EXISTS english_translations_king_idx ON english_translations (source, king);

-- text-embedding-3-small over citation + body. See corpus/translations/ingest.py.
CREATE TABLE IF NOT EXISTS english_translation_embeddings (
    translation_id text PRIMARY KEY REFERENCES english_translations (id) ON DELETE CASCADE,
    embedding      vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS english_translation_embeddings_hnsw_idx
    ON english_translation_embeddings USING hnsw (embedding vector_cosine_ops);
