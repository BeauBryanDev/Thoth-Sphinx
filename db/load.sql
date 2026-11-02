-- Run after evidence_store.extract, passing the folder that holds its CSVs:
--   psql "$DB" -v build_dir="$HOME/Downloads/AncientLanguage/build" -f db/load.sql
-- psql does not expand variables inside \copy, so this changes into that folder first.
\if :{?build_dir}
\else
  DO $$ BEGIN RAISE EXCEPTION 'set build_dir, e.g. -v build_dir=$HOME/Downloads/AncientLanguage/build'; END $$;
  \quit
\endif
\cd :build_dir
BEGIN;
TRUNCATE tokens, sentences, documents, lemma_relations, lemmas, terms;
\copy terms FROM 'terms.csv' WITH (FORMAT csv, HEADER true)
\copy lemmas FROM 'lemmas.csv' WITH (FORMAT csv, HEADER true)
\copy lemma_relations FROM 'lemma_relations.csv' WITH (FORMAT csv, HEADER true)
\copy documents FROM 'documents.csv' WITH (FORMAT csv, HEADER true)
\copy sentences FROM 'sentences.csv' WITH (FORMAT csv, HEADER true)
\copy tokens FROM 'tokens.csv' WITH (FORMAT csv, HEADER true)
COMMIT;
ANALYZE;
