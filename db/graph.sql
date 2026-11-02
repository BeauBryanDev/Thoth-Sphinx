-- Knowledge graph of gods, rulers, royal names, dynasties and temples. Safe to rerun;
-- python -m evidence_store.graph replaces the rows.
CREATE TABLE IF NOT EXISTS entities (
    id          text PRIMARY KEY,
    type        text NOT NULL,
    name        text NOT NULL,
    aliases     text[] NOT NULL DEFAULT '{}',
    start_year  integer,  -- negative is BC
    end_year    integer,
    lemma_ids   text[] NOT NULL DEFAULT '{}',  -- AAEW lemmas, so tokens lead to real sentences
    date_terms  text[] NOT NULL DEFAULT '{}',  -- document_dates terms, so periods lead to texts
    attributes  jsonb NOT NULL DEFAULT '{}',
    source      text NOT NULL,
    reliability text NOT NULL CHECK (reliability IN ('verified', 'reference', 'curated',
                                                     'derived', 'unverified'))
);

-- Symmetric relations are stored once, with the lower id first.
CREATE TABLE IF NOT EXISTS entity_links (
    source_id   text NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    relation    text NOT NULL,
    target_id   text NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    detail      text,  -- the source's own wording
    source      text NOT NULL,
    reliability text NOT NULL CHECK (reliability IN ('verified', 'reference', 'curated',
                                                     'derived', 'unverified')),
    PRIMARY KEY (source_id, relation, target_id)
);

ALTER TABLE entities DROP CONSTRAINT IF EXISTS entities_type_check;
ALTER TABLE entities ADD CONSTRAINT entities_type_check
    CHECK (type IN ('god', 'pharaoh', 'royal_name', 'dynasty', 'temple', 'pyramid'));

CREATE INDEX IF NOT EXISTS entity_links_target_idx ON entity_links (target_id);
CREATE INDEX IF NOT EXISTS entities_type_idx ON entities (type);
CREATE INDEX IF NOT EXISTS entities_lemma_idx ON entities USING gin (lemma_ids);
CREATE INDEX IF NOT EXISTS entities_name_trgm_idx ON entities USING gin (name gin_trgm_ops);

-- Finds entities by name or alias. Exact matches rank first, then close spellings.
CREATE OR REPLACE FUNCTION find_entities(p_query text, p_type text DEFAULT NULL, p_limit integer DEFAULT 3)
RETURNS TABLE (id text, match_score real)
LANGUAGE sql STABLE AS $$
    SELECT e.id, max(greatest(
               CASE WHEN lower(n.name) = lower(p_query) THEN 2.0 ELSE 0 END,
               word_similarity(lower(p_query), lower(n.name)) * similarity(lower(p_query), lower(n.name)) + 0.5 * similarity(lower(p_query), lower(n.name))
           ))::real
    FROM entities e, unnest(e.name || e.aliases) AS n(name)
    WHERE (p_type IS NULL OR e.type = p_type)
      AND (lower(n.name) = lower(p_query) OR lower(n.name) % lower(p_query)
           OR lower(p_query) <% lower(n.name))
    GROUP BY e.id
    ORDER BY 2 DESC, 1
    LIMIT p_limit
$$;
