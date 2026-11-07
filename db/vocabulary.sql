-- Run from the repo root after db/date_index.sql. Takes seconds; safe to rerun.
-- Middle Egyptian here means attested in a text dated to the First Intermediate
-- Period, the Middle Kingdom, or the Second Intermediate Period. Lemmas carry no
-- language stage, so the corpus decides. Undated texts never count.
BEGIN;
DROP MATERIALIZED VIEW IF EXISTS me_vocabulary;

CREATE MATERIALIZED VIEW me_vocabulary AS
WITH epochs (epoch, ord) AS (
    VALUES ('prädynastische Zeit', 1), ('Thinitenzeit', 2), ('Altes Reich', 3),
           ('Erste Zwischenzeit', 4), ('Mittleres Reich', 5), ('Zweite Zwischenzeit', 6),
           ('Neues Reich', 7), ('Dritte Zwischenzeit', 8), ('Spätzeit', 9),
           ('Makedonen, Ptolemäer', 10), ('römische Zeit', 11), ('byzantinische Zeit', 12)
),
me_docs AS (
    SELECT DISTINCT document_id
    FROM document_dates
    WHERE term IN ('Erste Zwischenzeit', 'Mittleres Reich', 'Zweite Zwischenzeit')
),
-- First reading of an ambivalence only, as sentence text does.
readings AS (
    SELECT t.lemma_id, t.sentence_id, t.w_char, t.sign_seq, s.text_id
    FROM tokens t
    JOIN sentences s ON s.id = t.sentence_id
    WHERE t.lemma_id IS NOT NULL
      AND coalesce(t.alternate, 0) = 0
),
me_readings AS (
    SELECT r.*
    FROM readings r
    JOIN me_docs m ON m.document_id = r.text_id
),
stats AS (
    SELECT r.lemma_id,
           count(*) AS me_tokens,
           count(DISTINCT r.sentence_id) AS me_sentences,
           count(DISTINCT r.text_id) AS me_texts,
           array_agg(DISTINCT d.corpus ORDER BY d.corpus) AS me_corpora
    FROM me_readings r
    JOIN documents d ON d.id = r.text_id
    GROUP BY r.lemma_id
),
-- The five commonest spellings in Middle Egyptian texts.
spellings AS (
    SELECT lemma_id,
           jsonb_agg(jsonb_build_object('w_char', w_char, 'sign_seq', sign_seq, 'count', n)
                     ORDER BY n DESC, w_char) AS spellings
    FROM (
        SELECT lemma_id, w_char, sign_seq, count(*) AS n,
               row_number() OVER (PARTITION BY lemma_id ORDER BY count(*) DESC, w_char) AS rn
        FROM me_readings
        WHERE w_char IS NOT NULL AND w_char <> ''
        GROUP BY lemma_id, w_char, sign_seq
    ) s
    WHERE rn <= 5
    GROUP BY lemma_id
),
-- Every epoch the lemma is attested in, across the whole corpus.
spans AS (
    SELECT lemma_id,
           array_agg(epoch ORDER BY ord) AS epochs,
           (array_agg(epoch ORDER BY ord))[1] AS first_epoch,
           (array_agg(epoch ORDER BY ord DESC))[1] AS last_epoch
    FROM (
        SELECT DISTINCT r.lemma_id, e.epoch, e.ord
        FROM readings r
        JOIN document_dates dd ON dd.document_id = r.text_id
        JOIN epochs e ON e.epoch = dd.term
        WHERE r.lemma_id IN (SELECT lemma_id FROM stats)
    ) x
    GROUP BY lemma_id
),
-- Three attestations with a German translation, preferring about eight words.
examples AS (
    SELECT lemma_id, array_agg(sentence_id ORDER BY rn) AS example_sentence_ids
    FROM (
        SELECT r.lemma_id, r.sentence_id,
               row_number() OVER (PARTITION BY r.lemma_id
                                  ORDER BY abs(s.word_count - 8), s.has_lacuna, r.sentence_id) AS rn
        FROM (SELECT DISTINCT lemma_id, sentence_id FROM me_readings) r
        JOIN sentences s ON s.id = r.sentence_id
        WHERE btrim(coalesce(s.translation_de, '')) <> ''
    ) x
    WHERE rn <= 3
    GROUP BY lemma_id
)
-- Inner join: about 480 attested lemma ids are missing from the word list.
SELECT l.id AS lemma_id, l.name, l.pos, l.subtype, l.gloss_en, l.gloss_de,
       l.sign_seq, l.revision_state,
       st.me_tokens, st.me_sentences, st.me_texts, st.me_corpora,
       coalesce(sp.spellings, '[]'::jsonb) AS spellings,
       sn.epochs, sn.first_epoch, sn.last_epoch,
       coalesce(ex.example_sentence_ids, '{}') AS example_sentence_ids
FROM stats st
JOIN lemmas l ON l.id = st.lemma_id
LEFT JOIN spellings sp ON sp.lemma_id = st.lemma_id
LEFT JOIN spans sn ON sn.lemma_id = st.lemma_id
LEFT JOIN examples ex ON ex.lemma_id = st.lemma_id;

CREATE UNIQUE INDEX me_vocabulary_pkey ON me_vocabulary (lemma_id);
CREATE INDEX me_vocabulary_name_trgm_idx ON me_vocabulary USING gin (name gin_trgm_ops);
CREATE INDEX me_vocabulary_gloss_en_trgm_idx ON me_vocabulary USING gin (gloss_en gin_trgm_ops);
CREATE INDEX me_vocabulary_pos_idx ON me_vocabulary (pos);

-- Every attested sign spelling in the whole corpus, for lookup_signs.
-- sign_key is the uppercased sign_seq, so Aa1 and AA1 meet.
DROP MATERIALIZED VIEW IF EXISTS sign_spellings CASCADE;
CREATE MATERIALIZED VIEW sign_spellings AS
WITH me_docs AS (
    SELECT DISTINCT document_id
    FROM document_dates
    WHERE term IN ('Erste Zwischenzeit', 'Mittleres Reich', 'Zweite Zwischenzeit')
)
SELECT upper(t.sign_seq) AS sign_key, t.lemma_id,
       cardinality(string_to_array(t.sign_seq, ' ')) AS n_signs,
       count(*) AS tokens,
       count(m.document_id) AS me_tokens
FROM tokens t
JOIN sentences s ON s.id = t.sentence_id
JOIN lemmas l ON l.id = t.lemma_id
LEFT JOIN me_docs m ON m.document_id = s.text_id
WHERE coalesce(t.alternate, 0) = 0
  AND btrim(coalesce(t.sign_seq, '')) <> ''
GROUP BY 1, 2, 3;

CREATE INDEX sign_spellings_key_idx ON sign_spellings (sign_key);

-- Each spelling of 3+ signs with one sign deleted, so a near miss is an index lookup
-- (the symmetric-delete trick). Shorter spellings only match exactly or swapped.
CREATE MATERIALIZED VIEW sign_spelling_deletes AS
SELECT DISTINCT array_to_string(k.signs[:i - 1] || k.signs[i + 1:], ' ') AS del_key, k.sign_key
FROM (SELECT DISTINCT sign_key, string_to_array(sign_key, ' ') AS signs
      FROM sign_spellings WHERE n_signs >= 3) k,
     generate_series(1, cardinality(k.signs)) AS i;

CREATE INDEX sign_spelling_deletes_key_idx ON sign_spelling_deletes (del_key);

-- Words whose spelling is one of p_keys, or whose spelling minus one sign is.
-- The caller sends each span, its one-sign deletions and its adjacent swaps, then
-- labels each hit by comparing spellings. Top p_per_key words per spelling.
CREATE OR REPLACE FUNCTION lookup_signs(p_keys text[], p_me_only boolean DEFAULT false,
                                        p_per_key integer DEFAULT 3)
RETURNS TABLE (sign_key text, lemma_id text, name text, pos text, gloss_en text,
               gloss_de text, tokens bigint, me_tokens bigint, obsolete boolean)
LANGUAGE sql STABLE AS $$
    WITH keys AS (
        SELECT sp.sign_key FROM sign_spellings sp WHERE sp.sign_key = ANY(p_keys)
        UNION
        SELECT d.sign_key FROM sign_spelling_deletes d WHERE d.del_key = ANY(p_keys)
    ),
    ranked AS (
        SELECT sp.*, row_number() OVER (
                   PARTITION BY sp.sign_key
                   ORDER BY CASE WHEN p_me_only THEN sp.me_tokens ELSE sp.tokens END DESC,
                            sp.lemma_id) AS rn
        FROM sign_spellings sp
        JOIN keys k ON k.sign_key = sp.sign_key
        WHERE NOT p_me_only OR sp.me_tokens > 0
    )
    SELECT r.sign_key, r.lemma_id, l.name, l.pos, l.gloss_en, l.gloss_de, r.tokens,
           r.me_tokens, l.revision_state = 'published-obsolete'
    FROM ranked r
    JOIN lemmas l ON l.id = r.lemma_id
    WHERE r.rn <= p_per_key
$$;
COMMIT;
ANALYZE me_vocabulary;
ANALYZE sign_spellings;
ANALYZE sign_spelling_deletes;
