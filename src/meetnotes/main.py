"""`uv run meetnotes` entry point: serve the app, or ingest into the KB."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

from .app import HOST, PORT, default_db_path


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:])
    if args.command == "ingest":
        asyncio.run(_ingest(args))
        return
    print(f"MeetNotes → http://{HOST}:{PORT}")
    uvicorn.run(
        "meetnotes.app:build",
        factory=True,
        host=HOST,
        port=PORT,
        log_level="warning",
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meetnotes", description="MeetNotes")
    sub = parser.add_subparsers(dest="command")
    ingest = sub.add_parser("ingest", help="Ingest documents into the knowledge base")
    ingest.add_argument("paths", nargs="*", type=Path, help="Files or directories")
    ingest.add_argument(
        "--meetings",
        action="store_true",
        help="Also ingest past meeting transcripts/notes from the app database",
    )
    return parser.parse_args(argv)


async def _ingest(args: argparse.Namespace) -> None:
    from .db import Database
    from .kb.embedder import get_local_embedder
    from .kb.ingest import ingest_meetings, ingest_paths

    if not args.paths and not args.meetings:
        print("Nothing to ingest: pass file/directory paths and/or --meetings")
        raise SystemExit(2)

    db = Database(default_db_path())
    await db.connect()
    try:
        embedder = get_local_embedder() if db.vec_available else None
        if not db.vec_available:
            print("warning: sqlite-vec unavailable — ingesting for keyword search only")
        totals = {"ingested": 0, "skipped": 0, "errors": 0}
        results = []
        if args.paths:
            results.append(await ingest_paths(db, embedder, args.paths))
        if args.meetings:
            results.append(await ingest_meetings(db, embedder))
        for result in results:
            for title in result.ingested:
                print(f"  + {title}")
            for title in result.skipped:
                print(f"  = {title} (unchanged)")
            for error in result.errors:
                print(f"  ! {error}")
            totals["ingested"] += len(result.ingested)
            totals["skipped"] += len(result.skipped)
            totals["errors"] += len(result.errors)
        print(
            f"Done: {totals['ingested']} ingested, {totals['skipped']} unchanged, "
            f"{totals['errors']} errors"
        )
    finally:
        await db.close()
