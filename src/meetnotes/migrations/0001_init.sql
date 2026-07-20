CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT,
    started_at INTEGER,
    ended_at INTEGER,
    status TEXT CHECK (status IN ('recording', 'paused', 'ended'))
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    meeting_id TEXT REFERENCES meetings(id),
    speaker INTEGER,
    channel INTEGER NOT NULL,
    start_ms INTEGER,
    end_ms INTEGER,
    text TEXT NOT NULL,
    confidence REAL
);

CREATE INDEX IF NOT EXISTS idx_segments_meeting_start ON segments (meeting_id, start_ms);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
