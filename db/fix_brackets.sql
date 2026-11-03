-- One-off fix for the angle-bracket bug (U+2329/U+232A left in translit_norm and w_norm).
-- Updates only the two normalized columns from a fixed extract; nothing else changes.
-- Dry run by default (ROLLBACK). Apply with -v apply=1. Then rerun db/index.sql.
--   psql "$EVIDENCE_DB" -v ON_ERROR_STOP=1 -v build_dir=<fixed build> -f db/fix_brackets.sql
\if :{?build_dir}
\else
  \echo 'build_dir is not set'
  \quit
\endif
\cd :build_dir

BEGIN;

CREATE TEMP TABLE fix_sentences (LIKE sentences) ON COMMIT DROP;
CREATE TEMP TABLE fix_tokens (LIKE tokens) ON COMMIT DROP;
\copy fix_sentences FROM 'sentences.csv' WITH (FORMAT csv, HEADER true)
\copy fix_tokens FROM 'tokens.csv' WITH (FORMAT csv, HEADER true)

-- The fixed extract must cover exactly the rows already loaded.
DO $$
BEGIN
  IF (SELECT count(*) FROM fix_sentences) <> (SELECT count(*) FROM sentences)
     OR (SELECT count(*) FROM fix_tokens) <> (SELECT count(*) FROM tokens) THEN
    RAISE EXCEPTION 'row counts differ from the loaded tables';
  END IF;
END $$;

UPDATE sentences s SET translit_norm = f.translit_norm
FROM fix_sentences f
WHERE s.id = f.id AND s.translit_norm IS DISTINCT FROM f.translit_norm;

UPDATE tokens t SET w_norm = f.w_norm
FROM fix_tokens f
WHERE t.sentence_id = f.sentence_id AND t.position = f.position
  AND t.w_norm IS DISTINCT FROM f.w_norm;

-- Expected from the CSV diff: 9,590 sentences and 13,727 tokens, and no brackets left.
DO $$
DECLARE
  left_s int := (SELECT count(*) FROM sentences WHERE translit_norm ~ U&'[\3008\3009\2329\232A]');
  left_t int := (SELECT count(*) FROM tokens WHERE w_norm ~ U&'[\3008\3009\2329\232A]');
  diff_s int := (SELECT count(*) FROM sentences s JOIN fix_sentences f USING (id)
                 WHERE s.translit_norm IS DISTINCT FROM f.translit_norm);
BEGIN
  RAISE NOTICE 'brackets left: sentences %, tokens %; rows still differing: %', left_s, left_t, diff_s;
  IF left_s + left_t + diff_s > 0 THEN
    RAISE EXCEPTION 'fix incomplete';
  END IF;
END $$;

\if :{?apply}
  COMMIT;
  \echo 'APPLIED. Next: psql ... -f db/index.sql'
\else
  ROLLBACK;
  \echo 'DRY RUN, rolled back. Rerun with -v apply=1 to commit.'
\endif
