"""REST surface tests over the real ASGI app (keyring monkeypatched)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from meetnotes import secrets
from meetnotes.app import create_app
from meetnotes.notegen import GeneratedNote, NoteGenerationError


class FakeNoteGenerator:
    """Scripted stand-in for the Claude-backed generator."""

    fail = False

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def generate(
        self,
        *,
        title: str,
        segments: list[dict[str, Any]],
        rough_lines: list[dict[str, Any]],
        template: str,
    ) -> GeneratedNote:
        if FakeNoteGenerator.fail:
            raise NoteGenerationError("scripted failure")
        return GeneratedNote(
            markdown=f"# {title}\n\nExpanded from {len(rough_lines)} note line(s).",
            action_items=[{"task": "Follow up", "owner": None, "due": None, "source_ts": "00:05"}],
            decisions=["Proceed"],
            model="fake-model",
        )


@pytest.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncClient:
    store: dict[str, str] = {}
    monkeypatch.setattr(secrets, "get_deepgram_key", lambda: store.get("dg"))
    monkeypatch.setattr(secrets, "set_deepgram_key", lambda v: store.__setitem__("dg", v))
    monkeypatch.setattr(secrets, "get_anthropic_key", lambda: store.get("an"))
    monkeypatch.setattr(secrets, "set_anthropic_key", lambda v: store.__setitem__("an", v))
    FakeNoteGenerator.fail = False
    app = create_app(db_path=tmp_path / "api.db", notegen_factory=FakeNoteGenerator)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            http.app_ctx = app.state.ctx  # type: ignore[attr-defined]
            yield http


async def test_settings_status_never_leaks_key(client: AsyncClient) -> None:
    res = await client.get("/settings/status")
    assert res.json() == {
        "deepgram_configured": False,
        "anthropic_configured": False,
        "active_meeting_id": None,
        "system_device": None,
    }

    res = await client.put("/settings/deepgram-key", json={"key": "dg-secret-123"})
    assert res.status_code == 200

    res = await client.get("/settings/status")
    body = res.json()
    assert body["deepgram_configured"] is True
    assert "dg-secret-123" not in res.text


async def test_empty_key_rejected(client: AsyncClient) -> None:
    res = await client.put("/settings/deepgram-key", json={"key": "   "})
    assert res.status_code == 400


async def test_meeting_requires_key(client: AsyncClient) -> None:
    res = await client.post("/meetings", json={})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "no_key"


async def test_meetings_listing_and_transcript_404(client: AsyncClient) -> None:
    assert (await client.get("/meetings")).json() == []
    assert (await client.get("/meetings/nope/transcript")).status_code == 404


async def test_pause_without_active_meeting(client: AsyncClient) -> None:
    res = await client.post("/meetings/x/pause")
    assert res.status_code == 409


FAKE_DEVICES = [
    {
        "index": 0,
        "name": "MacBook Pro Microphone",
        "input_channels": 1,
        "is_blackhole": False,
        "is_default": True,
    },
    {
        "index": 2,
        "name": "BlackHole 2ch",
        "input_channels": 2,
        "is_blackhole": True,
        "is_default": False,
    },
]


@pytest.fixture
def fake_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    from meetnotes import api as api_module

    monkeypatch.setattr(api_module, "list_input_devices", lambda: FAKE_DEVICES)
    monkeypatch.setattr(
        api_module,
        "find_device_index",
        lambda name: next((d["index"] for d in FAKE_DEVICES if d["name"] == name), None),
    )


async def test_device_listing_flags_blackhole(client: AsyncClient, fake_devices: None) -> None:
    res = await client.get("/settings/devices")
    body = res.json()
    assert body["blackhole_detected"] is True
    assert [d["name"] for d in body["devices"]] == ["MacBook Pro Microphone", "BlackHole 2ch"]


async def test_system_device_roundtrip(client: AsyncClient, fake_devices: None) -> None:
    res = await client.put("/settings/system-device", json={"device_name": "BlackHole 2ch"})
    assert res.status_code == 200
    assert (await client.get("/settings/status")).json()["system_device"] == "BlackHole 2ch"

    res = await client.put("/settings/system-device", json={"device_name": "Ghost Device"})
    assert res.status_code == 404

    res = await client.put("/settings/system-device", json={"device_name": None})
    assert res.status_code == 200
    assert (await client.get("/settings/status")).json()["system_device"] is None


# -- notes --------------------------------------------------------------------


async def _seed_ended_meeting(client: AsyncClient, meeting_id: str = "m1") -> None:
    ctx = client.app_ctx  # type: ignore[attr-defined]
    await ctx.db.create_meeting(meeting_id, "Planning", 0)
    await ctx.db.add_segment(
        "s1",
        meeting_id,
        speaker=0,
        channel=0,
        start_ms=0,
        end_ms=5000,
        text="let's proceed with the plan",
        confidence=0.9,
    )
    await ctx.db.end_meeting(meeting_id, 60_000)


async def test_rough_notes_roundtrip_api(client: AsyncClient) -> None:
    await _seed_ended_meeting(client)
    res = await client.put(
        "/meetings/m1/rough-notes",
        json={"lines": [{"ts_ms": 4000, "text": "plan agreed"}]},
    )
    assert res.status_code == 200
    res = await client.get("/meetings/m1/rough-notes")
    assert res.json() == {"lines": [{"ts_ms": 4000, "text": "plan agreed"}]}
    assert (await client.get("/meetings/nope/rough-notes")).status_code == 404


async def _wait_note_status(client: AsyncClient, meeting_id: str, status: str) -> dict:
    for _ in range(100):
        note = (await client.get(f"/meetings/{meeting_id}/note")).json()
        if note["status"] == status:
            return note
        await asyncio.sleep(0.01)
    raise AssertionError(f"note never reached status {status!r}: {note}")


async def test_note_generation_flow(client: AsyncClient) -> None:
    await _seed_ended_meeting(client)
    await client.put("/settings/anthropic-key", json={"key": "an-secret"})
    await client.put("/meetings/m1/rough-notes", json={"lines": [{"ts_ms": 1000, "text": "note"}]})

    res = await client.post("/meetings/m1/note/regenerate", json={})
    assert res.status_code == 200

    note = await _wait_note_status(client, "m1", "done")
    assert note["markdown"].startswith("# Planning")
    assert "1 note line(s)" in note["markdown"]
    assert note["action_items"][0]["task"] == "Follow up"
    assert note["decisions"] == ["Proceed"]
    assert note["model"] == "fake-model"

    # Editing and export
    res = await client.put("/meetings/m1/note", json={"markdown": "# Edited note"})
    assert res.status_code == 200
    export = await client.get("/meetings/m1/note.md")
    assert export.status_code == 200
    assert export.text == "# Edited note"
    assert "attachment" in export.headers["content-disposition"]


async def test_note_generation_failure_recorded(client: AsyncClient) -> None:
    await _seed_ended_meeting(client)
    await client.put("/settings/anthropic-key", json={"key": "an-secret"})
    FakeNoteGenerator.fail = True
    await client.post("/meetings/m1/note/regenerate", json={})
    note = await _wait_note_status(client, "m1", "error")
    assert note["error"] == "scripted failure"
    assert (await client.get("/meetings/m1/note.md")).status_code == 404


async def test_rename_meeting(client: AsyncClient) -> None:
    await _seed_ended_meeting(client)
    res = await client.put("/meetings/m1/title", json={"title": "  Q3 Pricing Sync  "})
    assert res.status_code == 200
    assert res.json() == {"id": "m1", "title": "Q3 Pricing Sync"}
    transcript = (await client.get("/meetings/m1/transcript")).json()
    assert transcript["meeting"]["title"] == "Q3 Pricing Sync"

    assert (await client.put("/meetings/m1/title", json={"title": "   "})).status_code == 400
    assert (await client.put("/meetings/nope/title", json={"title": "X"})).status_code == 404


async def test_delete_meeting(client: AsyncClient) -> None:
    await _seed_ended_meeting(client)
    await client.put("/meetings/m1/rough-notes", json={"lines": [{"ts_ms": 1000, "text": "note"}]})

    res = await client.delete("/meetings/m1")
    assert res.status_code == 200
    assert res.json() == {"deleted": "m1"}
    assert (await client.get("/meetings")).json() == []
    assert (await client.get("/meetings/m1/transcript")).status_code == 404
    assert (await client.delete("/meetings/m1")).status_code == 404


async def test_delete_active_meeting_rejected(client: AsyncClient) -> None:
    from types import SimpleNamespace

    ctx = client.app_ctx  # type: ignore[attr-defined]
    await ctx.db.create_meeting("live1", "Live", 0)
    ctx.session = SimpleNamespace(meeting_id="live1")
    try:
        res = await client.delete("/meetings/live1")
        assert res.status_code == 409
    finally:
        ctx.session = None
    assert (await client.get("/meetings/live1/transcript")).status_code == 200


async def test_regenerate_guards(client: AsyncClient) -> None:
    await _seed_ended_meeting(client)
    # No Anthropic key
    res = await client.post("/meetings/m1/note/regenerate", json={})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "no_anthropic_key"
    # Unknown template
    await client.put("/settings/anthropic-key", json={"key": "an-secret"})
    res = await client.post("/meetings/m1/note/regenerate", json={"template": "haiku"})
    assert res.status_code == 400
    # Meeting still recording
    ctx = client.app_ctx  # type: ignore[attr-defined]
    await ctx.db.create_meeting("live1", "Live", 0)
    res = await client.post("/meetings/live1/note/regenerate", json={})
    assert res.status_code == 409
