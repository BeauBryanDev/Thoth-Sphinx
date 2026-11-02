-- Run after evidence_store.embed. Workers stay at 0 because parallel builds outgrow Docker's /dev/shm.
SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 0;
CREATE INDEX IF NOT EXISTS sentence_embeddings_hnsw_idx
    ON sentence_embeddings USING hnsw (embedding vector_cosine_ops);
ANALYZE sentence_embeddings;
