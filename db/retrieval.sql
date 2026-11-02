-- Retrieval tables and functions. Safe to rerun; db/index.sql fills the tables.

CREATE TABLE IF NOT EXISTS vision_signs (code text PRIMARY KEY);

CREATE TABLE IF NOT EXISTS sentence_grams (
    channel     text NOT NULL,
    gram        text NOT NULL,
    sentence_id text NOT NULL REFERENCES sentences (id),
    PRIMARY KEY (channel, gram, sentence_id)
);

CREATE TABLE IF NOT EXISTS gram_stats (
    channel text NOT NULL,
    gram    text NOT NULL,
    df      integer NOT NULL,
    PRIMARY KEY (channel, gram)
);

CREATE TABLE IF NOT EXISTS sentence_stats (
    channel     text NOT NULL,
    sentence_id text NOT NULL,
    length      integer NOT NULL,
    PRIMARY KEY (channel, sentence_id)
);

CREATE TABLE IF NOT EXISTS channel_stats (
    channel text PRIMARY KEY,
    n       integer NOT NULL,
    avg_len double precision NOT NULL
);

-- text-embedding-3-small over German translations. See evidence_store/embed.py.
CREATE TABLE IF NOT EXISTS sentence_embeddings (
    sentence_id text PRIMARY KEY REFERENCES sentences (id),
    embedding   vector(1536) NOT NULL
);

-- Each document's dates by name, with their parent periods, so '6. Dyn.' also finds 'Pepi II.'.
-- Undated documents hold the single term 'undated'. Filled by db/date_index.sql.
CREATE TABLE IF NOT EXISTS document_dates (
    term        text NOT NULL,
    document_id text NOT NULL,
    PRIMARY KEY (term, document_id)
);

-- The old signatures, so that adding p_dates doesn't leave overloads behind.
DROP FUNCTION IF EXISTS search_hybrid(text, text, integer, text, text);
DROP FUNCTION IF EXISTS search_hybrid(text, text, integer, text, text, vector);
DROP FUNCTION IF EXISTS search_channel(text, text[], integer, text, text);
DROP FUNCTION IF EXISTS search_embedding(vector, integer, text, text);

-- Keeps only signs the YOLO detector can output, in reading order.
CREATE OR REPLACE FUNCTION vision_tokens(seq text) RETURNS text[]
LANGUAGE sql STABLE AS $$
    SELECT coalesce(array_agg(t.tok ORDER BY t.ord), '{}')
    FROM unnest(string_to_array(lower(seq), ' ')) WITH ORDINALITY AS t(tok, ord)
    JOIN vision_signs v ON v.code = t.tok
$$;

-- Adjacent pairs plus pairs skipping one sign, so a missed detection still matches.
CREATE OR REPLACE FUNCTION sign_grams(seq text) RETURNS text[]
LANGUAGE sql STABLE AS $$
    WITH t AS (SELECT vision_tokens(seq) AS tokens)
    SELECT CASE WHEN cardinality(tokens) < 2 THEN tokens ELSE ARRAY(
        SELECT tokens[i] || ' ' || tokens[i + gap]
        FROM generate_subscripts(tokens, 1) AS i, (VALUES (1), (2)) AS g(gap)
        WHERE i + gap <= cardinality(tokens)
    ) END
    FROM t
$$;

-- Folds spelling conventions: compounds split, plural marks and punctuation dropped.
CREATE OR REPLACE FUNCTION translit_tokens(q text) RETURNS text[]
LANGUAGE sql IMMUTABLE AS $$
    SELECT coalesce(array_agg(tok), '{}')
    FROM regexp_split_to_table(
        translate(regexp_replace(lower(q), '[.,](pl|du)\.?', '', 'g'), E'.,=̯̱', ''),
        '[\s-]+') AS tok
    WHERE tok <> ''
$$;

CREATE OR REPLACE FUNCTION translit_grams(q text) RETURNS text[]
LANGUAGE sql IMMUTABLE AS $$
    WITH t AS (SELECT translit_tokens(q) AS tokens)
    SELECT tokens || ARRAY(
        SELECT tokens[i] || ' ' || tokens[i + 1]
        FROM generate_subscripts(tokens, 1) AS i
        WHERE i < cardinality(tokens)
    )
    FROM t
$$;

-- BM25 over distinct grams. The exclusions keep evaluation honest. p_dates matches document_dates
-- terms, and it runs inside the channel because each channel keeps only its top hits.
CREATE OR REPLACE FUNCTION search_channel(
    p_channel text, p_grams text[], p_limit integer,
    p_exclude_text text DEFAULT NULL, p_exclude_norm text DEFAULT NULL,
    p_dates text[] DEFAULT NULL
) RETURNS TABLE (sentence_id text, score double precision)
LANGUAGE sql STABLE AS $$
    WITH stats AS (SELECT n, avg_len FROM channel_stats WHERE channel = p_channel),
    query AS (SELECT DISTINCT unnest(p_grams) AS gram),
    weights AS (
        SELECT q.gram, ln(1 + (s.n - g.df + 0.5) / (g.df + 0.5)) AS idf
        FROM query q
        JOIN gram_stats g ON g.channel = p_channel AND g.gram = q.gram
        CROSS JOIN stats s
        ORDER BY idf DESC, q.gram
        LIMIT 40  -- the rarest grams carry the ranking; common ones only add cost
    ),
    matches AS (
        SELECT sg.sentence_id, sum(w.idf) AS idf_sum
        FROM weights w
        JOIN sentence_grams sg ON sg.channel = p_channel AND sg.gram = w.gram
        GROUP BY sg.sentence_id
    )
    SELECT m.sentence_id,
           m.idf_sum * 2.2 / (1 + 1.2 * (0.25 + 0.75 * st.length / s.avg_len))
    FROM matches m
    JOIN sentence_stats st ON st.channel = p_channel AND st.sentence_id = m.sentence_id
    JOIN sentences x ON x.id = m.sentence_id
    CROSS JOIN stats s
    WHERE (p_exclude_text IS NULL OR x.text_id <> p_exclude_text)
      AND (p_exclude_norm IS NULL OR x.translit_norm <> p_exclude_norm)
      AND (p_dates IS NULL OR x.text_id IN (SELECT d.document_id FROM document_dates d
                                            WHERE d.term = ANY (p_dates)))
    ORDER BY 2 DESC, 1
    LIMIT p_limit
$$;

-- Nearest translations by cosine distance. The query is usually English; documents are German.
-- PL/pgSQL with custom plans: a SQL function hides LIMIT from the planner, which then skips HNSW.
-- Iterative scan keeps going when a date filter rejects the first candidates. A narrow filter can
-- still exhaust that scan and return nothing, so up to 10,000 sentences are compared exactly.
CREATE OR REPLACE FUNCTION search_embedding(
    p_embedding vector, p_limit integer,
    p_exclude_text text DEFAULT NULL, p_exclude_norm text DEFAULT NULL,
    p_dates text[] DEFAULT NULL
) RETURNS TABLE (sentence_id text, score double precision)
LANGUAGE plpgsql STABLE
SET hnsw.ef_search = 200
SET hnsw.iterative_scan = strict_order
SET plan_cache_mode = force_custom_plan
AS $$
DECLARE
    filtered bigint;
BEGIN
    IF p_embedding IS NULL THEN
        RETURN;
    END IF;
    IF p_dates IS NOT NULL THEN
        SELECT count(*) INTO filtered
        FROM sentences x
        WHERE x.text_id IN (SELECT d.document_id FROM document_dates d WHERE d.term = ANY (p_dates));
    END IF;
    IF filtered <= 10000 THEN
        RETURN QUERY
        WITH candidates AS MATERIALIZED (
            SELECT e.sentence_id, e.embedding
            FROM sentence_embeddings e
            JOIN sentences x ON x.id = e.sentence_id
            WHERE x.text_id IN (SELECT d.document_id FROM document_dates d
                                WHERE d.term = ANY (p_dates))
              AND (p_exclude_text IS NULL OR x.text_id <> p_exclude_text)
              AND (p_exclude_norm IS NULL OR x.translit_norm <> p_exclude_norm)
        )
        SELECT c.sentence_id, 1 - (c.embedding <=> p_embedding)
        FROM candidates c
        ORDER BY c.embedding <=> p_embedding
        LIMIT p_limit;
        RETURN;
    END IF;
    RETURN QUERY
    SELECT e.sentence_id, 1 - (e.embedding <=> p_embedding)
    FROM sentence_embeddings e
    JOIN sentences x ON x.id = e.sentence_id
    WHERE (p_exclude_text IS NULL OR x.text_id <> p_exclude_text)
      AND (p_exclude_norm IS NULL OR x.translit_norm <> p_exclude_norm)
      AND (p_dates IS NULL OR x.text_id IN (SELECT d.document_id FROM document_dates d
                                            WHERE d.term = ANY (p_dates)))
    ORDER BY e.embedding <=> p_embedding
    LIMIT p_limit;
END
$$;

-- Reciprocal rank fusion of every channel given a query. The ranks show which channel found a hit.
CREATE OR REPLACE FUNCTION search_hybrid(
    p_signs text, p_translit text, p_limit integer DEFAULT 10,
    p_exclude_text text DEFAULT NULL, p_exclude_norm text DEFAULT NULL,
    p_embedding vector DEFAULT NULL, p_dates text[] DEFAULT NULL
) RETURNS TABLE (sentence_id text, score double precision, sign_rank bigint,
                 translit_rank bigint, embedding_rank bigint, date text)
LANGUAGE sql STABLE AS $$
    WITH hits AS (
        SELECT 'sign' AS channel, h.sentence_id,
               row_number() OVER (ORDER BY h.score DESC, h.sentence_id) AS r
        FROM search_channel('sign', sign_grams(coalesce(p_signs, '')), 100,
                            p_exclude_text, p_exclude_norm, p_dates) h
        UNION ALL
        SELECT 'translit', h.sentence_id,
               row_number() OVER (ORDER BY h.score DESC, h.sentence_id)
        FROM search_channel('translit', translit_grams(coalesce(p_translit, '')), 100,
                            p_exclude_text, p_exclude_norm, p_dates) h
        UNION ALL
        SELECT 'embedding', h.sentence_id,
               row_number() OVER (ORDER BY h.score DESC, h.sentence_id)
        FROM search_embedding(p_embedding, 100, p_exclude_text, p_exclude_norm, p_dates) h
    ),
    fused AS (
        SELECT h.sentence_id AS id, sum(1.0 / (60 + h.r))::double precision AS total,
               min(h.r) FILTER (WHERE h.channel = 'sign') AS by_sign,
               min(h.r) FILTER (WHERE h.channel = 'translit') AS by_translit,
               min(h.r) FILTER (WHERE h.channel = 'embedding') AS by_embedding
        FROM hits h
        GROUP BY h.sentence_id
        ORDER BY 2 DESC, 1
        LIMIT p_limit
    )
    SELECT f.id, f.total, f.by_sign, f.by_translit, f.by_embedding, doc.date
    FROM fused f
    JOIN sentences x ON x.id = f.id
    JOIN documents doc ON doc.id = x.text_id
    ORDER BY f.total DESC, f.id
$$;

-- Valid date filter names, with how many sentences each covers.
CREATE OR REPLACE FUNCTION date_periods()
RETURNS TABLE (period text, sentences bigint)
LANGUAGE sql STABLE AS $$
    SELECT d.term, count(*)
    FROM document_dates d
    JOIN sentences s ON s.text_id = d.document_id
    WHERE d.term <> 'undated'
    GROUP BY d.term
    ORDER BY 2 DESC, 1
$$;

-- Plain trigram search, kept as the comparison baseline in evaluation.
CREATE OR REPLACE FUNCTION search_trigram(
    p_translit text, p_limit integer DEFAULT 10,
    p_exclude_text text DEFAULT NULL, p_exclude_norm text DEFAULT NULL
) RETURNS TABLE (sentence_id text, score double precision)
LANGUAGE sql STABLE AS $$
    SELECT x.id, similarity(x.translit_norm, p_translit)::double precision
    FROM sentences x
    WHERE x.translit_norm IS NOT NULL
      AND (p_exclude_text IS NULL OR x.text_id <> p_exclude_text)
      AND (p_exclude_norm IS NULL OR x.translit_norm <> p_exclude_norm)
    ORDER BY x.translit_norm <-> p_translit, x.id
    LIMIT p_limit
$$;

-- The hit plus neighbouring sentences, since many sentences are fragments.
CREATE OR REPLACE FUNCTION sentence_context(p_sentence_id text, p_window integer DEFAULT 2)
RETURNS TABLE (id text, offset_from_hit integer, transliteration text, translation_de text,
               translation_en text, has_lacuna boolean)
LANGUAGE sql STABLE AS $$
    SELECT n.id, n.position - s.position, n.transliteration, n.translation_de, n.translation_en,
           n.has_lacuna
    FROM sentences s
    JOIN sentences n ON n.text_id = s.text_id
        AND n.position BETWEEN s.position - p_window AND s.position + p_window
    WHERE s.id = p_sentence_id
    ORDER BY n.position
$$;

-- Dictionary lookup by transliteration or English gloss; obsolete entries rank last.
CREATE OR REPLACE FUNCTION lookup_lemma(p_query text, p_limit integer DEFAULT 10)
RETURNS TABLE (id text, name text, pos text, gloss_en text, gloss_de text, sign_seq text,
               revision_state text, score real)
LANGUAGE sql STABLE AS $$
    SELECT l.id, l.name, l.pos, l.gloss_en, l.gloss_de, l.sign_seq, l.revision_state,
           greatest(similarity(l.name, p_query), word_similarity(p_query, coalesce(l.gloss_en, '')))
    FROM lemmas l
    WHERE l.name % p_query OR p_query <% l.gloss_en
    ORDER BY l.revision_state = 'published-obsolete', 8 DESC,
             l.pos IN ('undefined', 'root', ''), l.gloss_en IS NULL, l.id
    LIMIT p_limit
$$;
