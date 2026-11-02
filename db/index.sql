-- Run from the repo root after db/load.sql and db/retrieval.sql.
BEGIN;
TRUNCATE sentence_grams, gram_stats, sentence_stats, channel_stats, vision_signs;
\copy vision_signs FROM 'evidence_store/vision_signs.txt'

INSERT INTO sentence_grams (channel, gram, sentence_id)
SELECT DISTINCT 'sign', g, s.id
FROM sentences s, LATERAL unnest(sign_grams(s.sign_seq)) AS g
WHERE s.sign_seq IS NOT NULL;

INSERT INTO sentence_grams (channel, gram, sentence_id)
SELECT DISTINCT 'translit', g, s.id
FROM sentences s, LATERAL unnest(translit_grams(s.translit_norm)) AS g
WHERE s.translit_norm IS NOT NULL;

INSERT INTO gram_stats
SELECT channel, gram, count(*) FROM sentence_grams GROUP BY channel, gram;

INSERT INTO sentence_stats
SELECT channel, sentence_id, count(*) FROM sentence_grams GROUP BY channel, sentence_id;

INSERT INTO channel_stats
SELECT channel, count(*), avg(length) FROM sentence_stats GROUP BY channel;
COMMIT;
ANALYZE sentence_grams;
ANALYZE gram_stats;
ANALYZE sentence_stats;
ANALYZE channel_stats;
