-- Run from the repo root after db/load.sql and db/retrieval.sql. Takes seconds.
BEGIN;
TRUNCATE document_dates;

INSERT INTO document_dates (term, document_id)
WITH RECURSIVE walk AS (
    SELECT d.id AS document_id, t.name, t.parent_id
    FROM documents d, jsonb_array_elements_text(d.date_ids) AS i(term_id)
    JOIN terms t ON t.id = i.term_id
    UNION
    SELECT w.document_id, p.name, p.parent_id
    FROM walk w
    JOIN terms p ON p.id = w.parent_id
)
SELECT DISTINCT name, document_id
FROM walk
WHERE name NOT LIKE '9 =%'    -- the root of the date tree
  AND name !~ '^\(\w';        -- grouping labels such as '(Epochen und Dynastien)', not '( ) / Itj'

INSERT INTO document_dates (term, document_id)
SELECT 'undated', d.id
FROM documents d
WHERE NOT EXISTS (SELECT 1 FROM document_dates x WHERE x.document_id = d.id);
COMMIT;
ANALYZE document_dates;
