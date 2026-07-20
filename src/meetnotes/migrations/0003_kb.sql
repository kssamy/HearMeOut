CREATE TABLE IF NOT EXISTS kb_docs (
    id TEXT PRIMARY KEY,
    path TEXT,
    title TEXT,
    sha256 TEXT,
    ingested_at INTEGER
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES kb_docs(id),
    heading TEXT,
    page INTEGER,
    start_char INTEGER,
    end_char INTEGER,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks (doc_id);

-- Keyword search over chunk text; kept in sync by the store layer.
CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(text, chunk_id UNINDEXED);

-- The sqlite-vec virtual table (kb_vec) is created at connect time in db.py,
-- not here: if the loadable extension is unavailable this migration must still
-- succeed, with retrieval degrading to FTS-only.
