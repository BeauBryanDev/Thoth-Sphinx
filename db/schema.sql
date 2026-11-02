CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS terms (
    id        text PRIMARY KEY,
    type      text,
    name      text,
    parent_id text
);

CREATE TABLE IF NOT EXISTS lemmas (
    id             text PRIMARY KEY,
    name           text,
    pos            text,
    subtype        text,
    gloss_de       text,
    gloss_en       text,
    sign_seq       text,
    revision_state text
);

CREATE TABLE IF NOT EXISTS lemma_relations (
    lemma_id  text NOT NULL REFERENCES lemmas (id),
    relation  text NOT NULL,
    target_id text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id               text PRIMARY KEY,
    corpus           text NOT NULL,
    kind             text NOT NULL,
    type             text,
    name             text,
    parent_ids       jsonb NOT NULL,
    path             text,
    owner            text,
    revision_state   text,
    date_ids         jsonb NOT NULL,
    date             text,
    find_spot        text,
    script           text,
    text_type        text,
    object_type      text,
    material         text,
    present_location text,
    egy_text_name    text,
    bibliography     text
);

CREATE TABLE IF NOT EXISTS sentences (
    id              text PRIMARY KEY,
    text_id         text NOT NULL REFERENCES documents (id),
    position        integer NOT NULL,
    transliteration text,  -- NULL for marker-only sentences, such as destroyed lines
    translit_norm   text,
    sign_seq        text,
    word_count      integer NOT NULL,
    has_lacuna      boolean NOT NULL,
    translation_de  text,
    translation_en  text,
    UNIQUE (text_id, position)
);

-- lemma_id is not a foreign key: 4,140 tokens point at lemmas missing from the export.
CREATE TABLE IF NOT EXISTS tokens (
    sentence_id text NOT NULL REFERENCES sentences (id),
    position    integer NOT NULL,
    alternate   integer,
    word_id     text,
    w_char      text,
    w_norm      text,
    lemma_id    text,
    flex_code   text,
    gloss_de    text,
    sign_seq    text,
    PRIMARY KEY (sentence_id, position)
);

CREATE INDEX IF NOT EXISTS tokens_lemma_idx ON tokens (lemma_id);
CREATE INDEX IF NOT EXISTS documents_corpus_idx ON documents (corpus);
CREATE INDEX IF NOT EXISTS lemma_relations_lemma_idx ON lemma_relations (lemma_id);
CREATE INDEX IF NOT EXISTS sentences_translit_trgm_idx ON sentences USING gin (translit_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS sentences_signs_trgm_idx ON sentences USING gin (sign_seq gin_trgm_ops);
CREATE INDEX IF NOT EXISTS lemmas_name_trgm_idx ON lemmas USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS lemmas_gloss_en_trgm_idx ON lemmas USING gin (gloss_en gin_trgm_ops);
