from __future__ import annotations

from pathlib import Path

from meetnotes.db import Database


async def test_migrations_are_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "m.db")
    await db.connect()
    await db.close()
    # Re-connecting re-runs the migration pass; already-applied versions skip.
    db2 = Database(tmp_path / "m.db")
    await db2.connect()
    async with db2._db().execute("SELECT COUNT(*) AS n FROM schema_migrations") as cur:
        row = await cur.fetchone()
    assert row["n"] == 3  # 0001_init + 0002_notes + 0003_kb, each applied once
    await db2.close()


async def test_segment_persistence_roundtrip(db: Database) -> None:
    await db.create_meeting("m1", "Standup", 1_000)
    await db.add_segment(
        "s1",
        "m1",
        speaker=0,
        channel=0,
        start_ms=0,
        end_ms=1500,
        text="hello world",
        confidence=0.97,
    )
    await db.add_segment(
        "s2",
        "m1",
        speaker=1,
        channel=0,
        start_ms=1500,
        end_ms=2500,
        text="hi there",
        confidence=0.92,
    )
    segments = await db.get_segments("m1")
    assert [s["text"] for s in segments] == ["hello world", "hi there"]
    assert segments[0]["speaker"] == 0
    assert segments[0]["channel"] == 0


async def test_segments_index_exists(db: Database) -> None:
    async with db._db().execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='segments'"
    ) as cur:
        names = {row["name"] for row in await cur.fetchall()}
    assert "idx_segments_meeting_start" in names


async def test_recover_orphaned_meetings(db: Database) -> None:
    await db.create_meeting("crashed", "Crashed meeting", 10_000)
    await db.add_segment(
        "s1",
        "crashed",
        speaker=None,
        channel=0,
        start_ms=0,
        end_ms=4000,
        text="still here",
        confidence=0.9,
    )
    recovered = await db.recover_orphaned_meetings()
    assert recovered == 1
    meeting = await db.get_meeting("crashed")
    assert meeting is not None
    assert meeting["status"] == "ended"
    assert meeting["ended_at"] == 14_000  # started_at + last segment end_ms
    # Segments survive recovery untouched.
    assert (await db.get_segments("crashed"))[0]["text"] == "still here"


async def test_rough_notes_roundtrip(db: Database) -> None:
    await db.create_meeting("m1", "T", 0)
    assert await db.get_rough_notes("m1") is None
    await db.save_rough_notes("m1", '[{"ts_ms": 5000, "text": "hello"}]')
    await db.save_rough_notes("m1", '[{"ts_ms": 5000, "text": "hello edited"}]')
    assert await db.get_rough_notes("m1") == '[{"ts_ms": 5000, "text": "hello edited"}]'


async def test_generated_note_lifecycle(db: Database) -> None:
    await db.create_meeting("m1", "T", 0)
    assert await db.get_generated_note("m1") is None

    await db.start_generated_note("m1", "general")
    note = await db.get_generated_note("m1")
    assert note is not None
    assert note["status"] == "generating"

    await db.fail_generated_note("m1", "boom")
    note = await db.get_generated_note("m1")
    assert (note["status"], note["error"]) == ("error", "boom")

    # Regenerate resets error state, then finishes cleanly.
    await db.start_generated_note("m1", "standup")
    await db.finish_generated_note(
        "m1",
        markdown="# Note",
        action_items_json="[]",
        decisions_json='["ship"]',
        model="claude-sonnet-4-6",
    )
    note = await db.get_generated_note("m1")
    assert note["status"] == "done"
    assert note["template"] == "standup"
    assert note["error"] is None
    assert note["markdown"] == "# Note"

    # Manual edits only apply to finished notes.
    assert await db.update_note_markdown("m1", "# Edited") is True
    assert (await db.get_generated_note("m1"))["markdown"] == "# Edited"
    assert await db.update_note_markdown("missing", "x") is False


async def test_delete_meeting_cascades(db: Database) -> None:
    await db.create_meeting("m1", "T", 0)
    await db.add_segment(
        "s1", "m1", speaker=0, channel=0, start_ms=0, end_ms=1000, text="hi", confidence=0.9
    )
    await db.save_rough_notes("m1", "[]")
    await db.start_generated_note("m1", "general")

    assert await db.delete_meeting("m1") is True
    assert await db.get_meeting("m1") is None
    assert await db.get_segments("m1") == []
    assert await db.get_rough_notes("m1") is None
    assert await db.get_generated_note("m1") is None
    assert await db.delete_meeting("m1") is False  # already gone


async def test_meeting_lifecycle(db: Database) -> None:
    await db.create_meeting("m1", "T", 5)
    await db.set_meeting_status("m1", "paused")
    assert (await db.get_meeting("m1"))["status"] == "paused"
    await db.end_meeting("m1", 99)
    meeting = await db.get_meeting("m1")
    assert meeting["status"] == "ended"
    assert meeting["ended_at"] == 99
    listing = await db.list_meetings()
    assert listing[0]["id"] == "m1"
    assert listing[0]["segment_count"] == 0
