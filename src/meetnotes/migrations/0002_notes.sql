CREATE TABLE IF NOT EXISTS rough_notes (
    meeting_id TEXT PRIMARY KEY REFERENCES meetings(id),
    lines TEXT NOT NULL,          -- JSON: [{"ts_ms": int, "text": str}, ...]
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS generated_notes (
    meeting_id TEXT PRIMARY KEY REFERENCES meetings(id),
    template TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('generating', 'done', 'error')),
    markdown TEXT,
    action_items TEXT,            -- JSON: [{"task", "owner", "due", "source_ts"}, ...]
    decisions TEXT,               -- JSON: [str, ...]
    model TEXT,
    error TEXT,
    generated_at INTEGER
);
