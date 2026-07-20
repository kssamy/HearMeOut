"""SQLite persistence via aiosqlite, with numbered SQL-file migrations."""

from __future__ import annotations

import logging
import re
import struct
import time
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
EMBED_DIM = 384  # all-MiniLM-L6-v2; changing models means a new kb_vec table


def now_ms() -> int:
    return int(time.time() * 1000)


def _serialize_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class Database:
    """Async SQLite wrapper. Finalized segments are committed the moment they
    arrive so a hard kill loses at most the in-flight interim segment."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conn: aiosqlite.Connection | None = None
        self.vec_available = False

    @property
    def path(self) -> Path:
        return self._path

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._load_vec_extension()
        await self._migrate()

    async def _load_vec_extension(self) -> None:
        """Load sqlite-vec for vector search. On failure the KB degrades to
        keyword-only retrieval instead of breaking the app."""
        assert self._conn is not None
        try:
            import sqlite_vec

            await self._conn.enable_load_extension(True)
            await self._conn.load_extension(sqlite_vec.loadable_path())
            await self._conn.enable_load_extension(False)
            await self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS kb_vec USING vec0("
                f"chunk_id TEXT PRIMARY KEY, embedding FLOAT[{EMBED_DIM}] "
                "distance_metric=cosine)"
            )
            self.vec_available = True
        except Exception as exc:
            logger.warning("sqlite-vec unavailable (%s) — KB retrieval will be keyword-only", exc)
            self.vec_available = False

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected")
        return self._conn

    async def _migrate(self) -> None:
        db = self._db()
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)"
        )
        async with db.execute("SELECT version FROM schema_migrations") as cur:
            applied = {row["version"] for row in await cur.fetchall()}
        for sql_file in sorted(MIGRATIONS_DIR.glob("[0-9]*.sql")):
            version = int(sql_file.name.split("_", 1)[0])
            if version in applied:
                continue
            await db.executescript(sql_file.read_text(encoding="utf-8"))
            await db.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, now_ms()),
            )
            logger.info("Applied migration %s", sql_file.name)
        await db.commit()

    # -- meetings ----------------------------------------------------------

    async def create_meeting(self, meeting_id: str, title: str, started_at: int) -> None:
        db = self._db()
        await db.execute(
            "INSERT INTO meetings (id, title, started_at, status) VALUES (?, ?, ?, 'recording')",
            (meeting_id, title, started_at),
        )
        await db.commit()

    async def rename_meeting(self, meeting_id: str, title: str) -> bool:
        db = self._db()
        cur = await db.execute("UPDATE meetings SET title = ? WHERE id = ?", (title, meeting_id))
        await db.commit()
        return bool(cur.rowcount)

    async def set_meeting_status(self, meeting_id: str, status: str) -> None:
        db = self._db()
        await db.execute("UPDATE meetings SET status = ? WHERE id = ?", (status, meeting_id))
        await db.commit()

    async def end_meeting(self, meeting_id: str, ended_at: int) -> None:
        db = self._db()
        await db.execute(
            "UPDATE meetings SET status = 'ended', ended_at = ? WHERE id = ?",
            (ended_at, meeting_id),
        )
        await db.commit()

    async def get_meeting(self, meeting_id: str) -> dict[str, Any] | None:
        async with self._db().execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_meetings(self) -> list[dict[str, Any]]:
        async with self._db().execute(
            "SELECT m.*, "
            "  (SELECT COUNT(*) FROM segments s WHERE s.meeting_id = m.id) AS segment_count "
            "FROM meetings m ORDER BY m.started_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def delete_meeting(self, meeting_id: str) -> bool:
        """Hard-delete a meeting and everything attached to it. Returns False
        if no such meeting existed."""
        db = self._db()
        await db.execute("DELETE FROM segments WHERE meeting_id = ?", (meeting_id,))
        await db.execute("DELETE FROM rough_notes WHERE meeting_id = ?", (meeting_id,))
        await db.execute("DELETE FROM generated_notes WHERE meeting_id = ?", (meeting_id,))
        cur = await db.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        await db.commit()
        return bool(cur.rowcount)

    async def recover_orphaned_meetings(self) -> int:
        """Close meetings left 'recording'/'paused' by a crash, keeping all
        finalized segments intact. Returns the number of meetings recovered."""
        db = self._db()
        cur = await db.execute(
            "UPDATE meetings SET status = 'ended', ended_at = COALESCE("
            "  (SELECT meetings.started_at + MAX(s.end_ms) FROM segments s"
            "   WHERE s.meeting_id = meetings.id),"
            "  started_at)"
            " WHERE status IN ('recording', 'paused')"
        )
        await db.commit()
        return cur.rowcount or 0

    # -- settings ----------------------------------------------------------

    async def get_setting(self, key: str) -> str | None:
        async with self._db().execute("SELECT value FROM settings WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return row["value"] if row is not None else None

    async def set_setting(self, key: str, value: str | None) -> None:
        db = self._db()
        if value is None:
            await db.execute("DELETE FROM settings WHERE key = ?", (key,))
        else:
            await db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        await db.commit()

    # -- segments ----------------------------------------------------------

    async def add_segment(
        self,
        segment_id: str,
        meeting_id: str,
        *,
        speaker: int | None,
        channel: int,
        start_ms: int,
        end_ms: int,
        text: str,
        confidence: float,
    ) -> None:
        db = self._db()
        await db.execute(
            "INSERT INTO segments (id, meeting_id, speaker, channel, start_ms, end_ms,"
            " text, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (segment_id, meeting_id, speaker, channel, start_ms, end_ms, text, confidence),
        )
        await db.commit()

    async def get_segments(self, meeting_id: str) -> list[dict[str, Any]]:
        async with self._db().execute(
            "SELECT * FROM segments WHERE meeting_id = ? ORDER BY start_ms",
            (meeting_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    # -- rough notes (in-meeting notepad) ----------------------------------

    async def save_rough_notes(self, meeting_id: str, lines_json: str) -> None:
        db = self._db()
        await db.execute(
            "INSERT INTO rough_notes (meeting_id, lines, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(meeting_id) DO UPDATE SET lines = excluded.lines,"
            " updated_at = excluded.updated_at",
            (meeting_id, lines_json, now_ms()),
        )
        await db.commit()

    async def get_rough_notes(self, meeting_id: str) -> str | None:
        async with self._db().execute(
            "SELECT lines FROM rough_notes WHERE meeting_id = ?", (meeting_id,)
        ) as cur:
            row = await cur.fetchone()
        return row["lines"] if row is not None else None

    # -- generated notes ---------------------------------------------------

    async def start_generated_note(self, meeting_id: str, template: str) -> None:
        db = self._db()
        await db.execute(
            "INSERT INTO generated_notes (meeting_id, template, status)"
            " VALUES (?, ?, 'generating')"
            " ON CONFLICT(meeting_id) DO UPDATE SET template = excluded.template,"
            " status = 'generating', error = NULL",
            (meeting_id, template),
        )
        await db.commit()

    async def finish_generated_note(
        self,
        meeting_id: str,
        *,
        markdown: str,
        action_items_json: str,
        decisions_json: str,
        model: str,
    ) -> None:
        db = self._db()
        await db.execute(
            "UPDATE generated_notes SET status = 'done', markdown = ?, action_items = ?,"
            " decisions = ?, model = ?, error = NULL, generated_at = ? WHERE meeting_id = ?",
            (markdown, action_items_json, decisions_json, model, now_ms(), meeting_id),
        )
        await db.commit()

    async def fail_generated_note(self, meeting_id: str, error: str) -> None:
        db = self._db()
        await db.execute(
            "UPDATE generated_notes SET status = 'error', error = ? WHERE meeting_id = ?",
            (error, meeting_id),
        )
        await db.commit()

    async def update_note_markdown(self, meeting_id: str, markdown: str) -> bool:
        db = self._db()
        cur = await db.execute(
            "UPDATE generated_notes SET markdown = ? WHERE meeting_id = ? AND status = 'done'",
            (markdown, meeting_id),
        )
        await db.commit()
        return bool(cur.rowcount)

    async def get_generated_note(self, meeting_id: str) -> dict[str, Any] | None:
        async with self._db().execute(
            "SELECT * FROM generated_notes WHERE meeting_id = ?", (meeting_id,)
        ) as cur:
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    # -- knowledge base ----------------------------------------------------

    async def get_kb_doc_by_path(self, path: str) -> dict[str, Any] | None:
        async with self._db().execute("SELECT * FROM kb_docs WHERE path = ?", (path,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def list_kb_docs(self) -> list[dict[str, Any]]:
        async with self._db().execute(
            "SELECT d.*, (SELECT COUNT(*) FROM kb_chunks c WHERE c.doc_id = d.id) AS chunks"
            " FROM kb_docs d ORDER BY d.ingested_at DESC"
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]

    async def kb_doc_count(self) -> int:
        async with self._db().execute("SELECT COUNT(*) AS n FROM kb_docs") as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row is not None else 0

    async def delete_kb_doc(self, doc_id: str) -> bool:
        db = self._db()
        async with db.execute("SELECT id FROM kb_chunks WHERE doc_id = ?", (doc_id,)) as cur:
            chunk_ids = [row["id"] for row in await cur.fetchall()]
        for chunk_id in chunk_ids:
            await db.execute("DELETE FROM kb_fts WHERE chunk_id = ?", (chunk_id,))
            if self.vec_available:
                await db.execute("DELETE FROM kb_vec WHERE chunk_id = ?", (chunk_id,))
        await db.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
        cur2 = await db.execute("DELETE FROM kb_docs WHERE id = ?", (doc_id,))
        await db.commit()
        return bool(cur2.rowcount)

    async def replace_kb_doc(
        self,
        *,
        doc_id: str,
        path: str,
        title: str,
        sha256: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]] | None,
    ) -> None:
        """Insert or replace a document and all of its chunks atomically-ish:
        any previous doc at the same path is removed first (re-ingest)."""
        existing = await self.get_kb_doc_by_path(path)
        if existing is not None:
            await self.delete_kb_doc(existing["id"])
        db = self._db()
        await db.execute(
            "INSERT INTO kb_docs (id, path, title, sha256, ingested_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, path, title, sha256, now_ms()),
        )
        for index, chunk in enumerate(chunks):
            await db.execute(
                "INSERT INTO kb_chunks (id, doc_id, heading, page, start_char, end_char, text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk["id"],
                    doc_id,
                    chunk["heading"],
                    chunk["page"],
                    chunk["start_char"],
                    chunk["end_char"],
                    chunk["text"],
                ),
            )
            await db.execute(
                "INSERT INTO kb_fts (text, chunk_id) VALUES (?, ?)",
                (chunk["text"], chunk["id"]),
            )
            if self.vec_available and embeddings is not None:
                await db.execute(
                    "INSERT INTO kb_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk["id"], _serialize_vector(embeddings[index])),
                )
        await db.commit()

    async def kb_vector_search(self, embedding: list[float], k: int = 8) -> list[tuple[str, float]]:
        """Top-k (chunk_id, cosine similarity), best first."""
        if not self.vec_available:
            return []
        async with self._db().execute(
            "SELECT chunk_id, distance FROM kb_vec WHERE embedding MATCH ? AND k = ?",
            (_serialize_vector(embedding), k),
        ) as cur:
            rows = await cur.fetchall()
        return [(str(row["chunk_id"]), 1.0 - float(row["distance"])) for row in rows]

    async def kb_keyword_search(self, query: str, k: int = 8) -> list[str]:
        """Top-k chunk ids by BM25, best first. The query is sanitized to bare
        OR-ed terms so user text can't break FTS5 syntax."""
        terms = re.findall(r"[A-Za-z0-9]{2,}", query)
        if not terms:
            return []
        match = " OR ".join(f'"{t}"' for t in terms[:20])
        async with self._db().execute(
            "SELECT chunk_id FROM kb_fts WHERE kb_fts MATCH ? ORDER BY bm25(kb_fts) LIMIT ?",
            (match, k),
        ) as cur:
            rows = await cur.fetchall()
        return [str(row["chunk_id"]) for row in rows]

    async def get_kb_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        """Chunk rows joined with their doc title/path, in the given order."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        async with self._db().execute(
            f"SELECT c.*, d.title AS doc_title, d.path AS doc_path"
            f" FROM kb_chunks c JOIN kb_docs d ON d.id = c.doc_id"
            f" WHERE c.id IN ({placeholders})",
            chunk_ids,
        ) as cur:
            rows = await cur.fetchall()
        by_id = {row["id"]: dict(row) for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def kb_entity_terms(self) -> list[str]:
        """Doc titles and headings — the copilot's entity-trigger vocabulary."""
        terms: list[str] = []
        async with self._db().execute("SELECT title FROM kb_docs") as cur:
            terms.extend(str(row["title"]) for row in await cur.fetchall() if row["title"])
        async with self._db().execute(
            "SELECT DISTINCT heading FROM kb_chunks WHERE heading != ''"
        ) as cur:
            for row in await cur.fetchall():
                terms.extend(part.strip() for part in str(row["heading"]).split(">"))
        return [t for t in terms if t]
