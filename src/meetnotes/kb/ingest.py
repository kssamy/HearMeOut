"""Ingestion pipeline: extract → chunk → embed → store (with change detection)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..db import Database
from .chunker import chunk_blocks
from .embedder import Embedder
from .extract import SUPPORTED_SUFFIXES, ExtractedDoc, ExtractionError, extract_file
from .extract import extract_markdown_text as _extract_md

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    ingested: list[str]  # doc titles newly (re)ingested
    skipped: list[str]  # unchanged since last ingest
    errors: list[str]  # "path: reason"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file():
            files.append(path)
        else:
            logger.warning("Skipping missing path %s", path)
    return files


async def ingest_paths(db: Database, embedder: Embedder | None, paths: list[Path]) -> IngestResult:
    ingested: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    for file_path in collect_files(paths):
        try:
            outcome = await _ingest_file(db, embedder, file_path)
        except ExtractionError as exc:
            errors.append(f"{file_path}: {exc}")
            continue
        except Exception as exc:
            logger.exception("Ingestion failed for %s", file_path)
            errors.append(f"{file_path}: {exc}")
            continue
        (ingested if outcome is not None else skipped).append(outcome or file_path.name)
    return IngestResult(ingested=ingested, skipped=skipped, errors=errors)


async def _ingest_file(db: Database, embedder: Embedder | None, path: Path) -> str | None:
    """Returns the doc title if (re)ingested, None if unchanged."""
    resolved = str(await asyncio.to_thread(path.resolve))
    digest = _sha256_bytes(await asyncio.to_thread(path.read_bytes))
    existing = await db.get_kb_doc_by_path(resolved)
    if existing is not None and existing["sha256"] == digest:
        return None
    extracted = await asyncio.to_thread(extract_file, path)
    await _store_doc(db, embedder, path=resolved, digest=digest, extracted=extracted)
    return extracted.title


async def ingest_meetings(db: Database, embedder: Embedder | None) -> IngestResult:
    """Ingest ended meetings (transcript + generated note) as KB docs."""
    ingested: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    for meeting in await db.list_meetings():
        if meeting["status"] != "ended":
            continue
        try:
            text = await _meeting_text(db, meeting)
            if not text.strip():
                continue
            pseudo_path = f"meeting://{meeting['id']}"
            digest = _sha256_bytes(text.encode())
            existing = await db.get_kb_doc_by_path(pseudo_path)
            if existing is not None and existing["sha256"] == digest:
                skipped.append(meeting["title"] or meeting["id"])
                continue
            title = meeting["title"] or f"Meeting {meeting['id'][:8]}"
            extracted = _extract_md(text, title)
            await _store_doc(db, embedder, path=pseudo_path, digest=digest, extracted=extracted)
            ingested.append(title)
        except Exception as exc:
            logger.exception("Meeting ingestion failed for %s", meeting["id"])
            errors.append(f"{meeting['id']}: {exc}")
    return IngestResult(ingested=ingested, skipped=skipped, errors=errors)


async def _meeting_text(db: Database, meeting: dict[str, Any]) -> str:
    parts = [f"# {meeting['title'] or 'Meeting'}"]
    note = await db.get_generated_note(meeting["id"])
    if note is not None and note["status"] == "done" and note["markdown"]:
        parts.append(str(note["markdown"]))
    segments = await db.get_segments(meeting["id"])
    if segments:
        parts.append("## Transcript")
        parts.append(
            "\n\n".join(f"{'Them' if s['channel'] == 1 else 'Me'}: {s['text']}" for s in segments)
        )
    if len(parts) == 1:
        return ""  # title-only meeting: nothing worth indexing
    return "\n\n".join(parts)


async def _store_doc(
    db: Database,
    embedder: Embedder | None,
    *,
    path: str,
    digest: str,
    extracted: ExtractedDoc,
) -> None:
    chunks = chunk_blocks(extracted.blocks)
    chunk_rows = [
        {
            "id": uuid4().hex,
            "heading": chunk.heading,
            "page": chunk.page,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    embeddings: list[list[float]] | None = None
    if embedder is not None and db.vec_available and chunk_rows:
        embeddings = await asyncio.to_thread(
            embedder.embed, [str(row["text"]) for row in chunk_rows]
        )
    await db.replace_kb_doc(
        doc_id=uuid4().hex,
        path=path,
        title=extracted.title,
        sha256=digest,
        chunks=chunk_rows,
        embeddings=embeddings,
    )
