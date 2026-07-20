"""REST + WebSocket API surface."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from . import secrets
from .capture import Capture, CaptureError
from .copilot.engine import CopilotEngine, first_sentence
from .copilot.extract import QueryExtractor
from .copilot.suppression import RELEVANCE_FLOOR_DEFAULT
from .copilot.triggers import TriggerKind
from .db import now_ms
from .devices import find_device_index, list_input_devices
from .kb.ingest import ingest_meetings, ingest_paths
from .notegen import DEFAULT_TEMPLATE, TEMPLATES, NoteGenerationError
from .ring import ChannelFrame, RingQueue
from .session import MeetingSession

if TYPE_CHECKING:
    from .app import AppContext

logger = logging.getLogger(__name__)

router = APIRouter()

RING_MAX_FRAMES = 200  # 20s of 100ms frames between callbacks and sender
SYSTEM_DEVICE_KEY = "system_audio_device"
LEVEL_TEST_TIMEOUT_S = 120.0
COPILOT_ENABLED_KEY = "copilot_enabled"
COPILOT_FLOOR_KEY = "copilot_relevance_floor"


def _ctx(request: Request) -> AppContext:
    return cast("AppContext", request.app.state.ctx)


class CreateMeetingBody(BaseModel):
    title: str | None = None


class DeepgramKeyBody(BaseModel):
    key: str


class SystemDeviceBody(BaseModel):
    device_name: str | None


class LevelTestBody(BaseModel):
    device: int


class NoteLine(BaseModel):
    ts_ms: int
    text: str


class RoughNotesBody(BaseModel):
    lines: list[NoteLine]


class NoteEditBody(BaseModel):
    markdown: str


class RegenerateBody(BaseModel):
    template: str | None = None


class RenameBody(BaseModel):
    title: str


class KbIngestBody(BaseModel):
    path: str | None = None
    include_meetings: bool = False


class CopilotSettingsBody(BaseModel):
    enabled: bool | None = None
    relevance_floor: float | None = None


@router.post("/meetings")
async def create_meeting(request: Request, body: CreateMeetingBody) -> dict[str, Any]:
    ctx = _ctx(request)
    if ctx.session is not None:
        raise HTTPException(status_code=409, detail="A meeting is already in progress.")
    api_key = await asyncio.to_thread(secrets.get_deepgram_key)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail={"message": "Deepgram API key is not configured.", "code": "no_key"},
        )
    await _stop_probe(ctx)

    # Channel plan: 0 = mic (always); 1 = system loopback when configured.
    capture_plan: list[tuple[int, int | None]] = [(0, None)]
    device_warning: str | None = None
    system_device_name = await ctx.db.get_setting(SYSTEM_DEVICE_KEY)
    if system_device_name:
        device_index = await asyncio.to_thread(find_device_index, system_device_name)
        if device_index is None:
            device_warning = (
                f'System-audio device "{system_device_name}" was not found — '
                "recording the microphone only. Check Settings."
            )
        else:
            capture_plan.append((1, device_index))

    meeting_id = uuid4().hex
    title = body.title or datetime.now().strftime("Meeting %b %d, %H:%M")
    started_at = now_ms()
    await ctx.db.create_meeting(meeting_id, title, started_at)

    ring: RingQueue[ChannelFrame] = RingQueue(RING_MAX_FRAMES)
    captures: list[tuple[int, Capture]] = [
        (channel, ctx.capture_factory(ring, channel, device)) for channel, device in capture_plan
    ]
    session = MeetingSession(
        meeting_id,
        db=ctx.db,
        bus=ctx.bus,
        provider=ctx.provider_factory(api_key),
        captures=captures,
        ring=ring,
    )
    try:
        await session.start()
    except CaptureError as exc:
        await ctx.db.end_meeting(meeting_id, now_ms())
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Could not open the microphone: {exc}",
                "code": "mic_error",
            },
        ) from exc
    ctx.session = session
    await _start_copilot(ctx, session)
    if device_warning:
        await ctx.bus.broadcast(
            "capture.status",
            {
                "meeting_id": meeting_id,
                "state": "recording",
                "message": device_warning,
                "code": "device_missing",
            },
        )
    return {
        "id": meeting_id,
        "title": title,
        "started_at": started_at,
        "status": "recording",
        "channels": session.channels,
    }


def _require_active(ctx: AppContext, meeting_id: str) -> MeetingSession:
    if ctx.session is None:
        raise HTTPException(status_code=409, detail="No meeting is in progress.")
    if ctx.session.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="That meeting is not the active one.")
    return ctx.session


@router.post("/meetings/{meeting_id}/pause")
async def pause_meeting(request: Request, meeting_id: str) -> dict[str, Any]:
    session = _require_active(_ctx(request), meeting_id)
    await session.pause()
    return {"id": meeting_id, "status": "paused"}


@router.post("/meetings/{meeting_id}/resume")
async def resume_meeting(request: Request, meeting_id: str) -> dict[str, Any]:
    session = _require_active(_ctx(request), meeting_id)
    await session.resume()
    return {"id": meeting_id, "status": "recording"}


async def _copilot_effective_enabled(ctx: AppContext) -> bool:
    """Setting wins when explicitly set; otherwise ON iff the KB has docs."""
    setting = await ctx.db.get_setting(COPILOT_ENABLED_KEY)
    if setting is not None:
        return setting == "1"
    return await ctx.db.kb_doc_count() > 0


async def _copilot_floor(ctx: AppContext) -> float:
    raw = await ctx.db.get_setting(COPILOT_FLOOR_KEY)
    try:
        return float(raw) if raw is not None else RELEVANCE_FLOOR_DEFAULT
    except ValueError:
        return RELEVANCE_FLOOR_DEFAULT


async def _start_copilot(ctx: AppContext, session: MeetingSession) -> None:
    """Attach a CopilotEngine to the session when enabled and the KB is
    non-empty. When OFF (or KB empty) nothing is constructed — zero API calls."""
    if not await _copilot_effective_enabled(ctx) or await ctx.db.kb_doc_count() == 0:
        return
    anthropic_key = await asyncio.to_thread(secrets.get_anthropic_key)
    extractor = QueryExtractor(anthropic_key) if anthropic_key else None
    embedder = ctx.embedder_factory() if ctx.db.vec_available else None
    engine = CopilotEngine(
        session.meeting_id,
        db=ctx.db,
        bus=ctx.bus,
        session=session,
        embedder=embedder,
        extractor=extractor,
        relevance_floor=await _copilot_floor(ctx),
    )
    session.on_final_segment = engine.on_final_segment
    await engine.start()
    ctx.copilot = engine


@router.post("/meetings/{meeting_id}/end")
async def end_meeting(request: Request, meeting_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    session = _require_active(ctx, meeting_id)
    if ctx.copilot is not None:
        await ctx.copilot.stop()
        ctx.copilot = None
    await session.end()
    ctx.session = None
    # Phase-4 hook: kick off Claude note generation in the background.
    anthropic_key = await asyncio.to_thread(secrets.get_anthropic_key)
    if anthropic_key:
        _start_note_generation(ctx, meeting_id, DEFAULT_TEMPLATE, anthropic_key)
    meeting = await ctx.db.get_meeting(meeting_id)
    return meeting or {"id": meeting_id, "status": "ended"}


def _start_note_generation(ctx: AppContext, meeting_id: str, template: str, api_key: str) -> None:
    existing = ctx.note_tasks.get(meeting_id)
    if existing is not None and not existing.done():
        return
    task = asyncio.create_task(
        _generate_note(ctx, meeting_id, template, api_key),
        name=f"notegen-{meeting_id}",
    )
    ctx.note_tasks[meeting_id] = task


async def _generate_note(ctx: AppContext, meeting_id: str, template: str, api_key: str) -> None:
    await ctx.db.start_generated_note(meeting_id, template)
    await ctx.bus.broadcast("note.status", {"meeting_id": meeting_id, "status": "generating"})
    try:
        meeting = await ctx.db.get_meeting(meeting_id)
        segments = await ctx.db.get_segments(meeting_id)
        rough_json = await ctx.db.get_rough_notes(meeting_id)
        rough_lines = json.loads(rough_json) if rough_json else []
        generator = ctx.notegen_factory(api_key)
        note = await generator.generate(
            title=(meeting or {}).get("title") or "Untitled meeting",
            segments=segments,
            rough_lines=rough_lines,
            template=template,
        )
        await ctx.db.finish_generated_note(
            meeting_id,
            markdown=note.markdown,
            action_items_json=json.dumps(note.action_items),
            decisions_json=json.dumps(note.decisions),
            model=note.model,
        )
        await ctx.bus.broadcast("note.status", {"meeting_id": meeting_id, "status": "done"})
    except NoteGenerationError as exc:
        await ctx.db.fail_generated_note(meeting_id, str(exc))
        await ctx.bus.broadcast(
            "note.status", {"meeting_id": meeting_id, "status": "error", "error": str(exc)}
        )
    except Exception as exc:
        logger.exception("Note generation failed for %s", meeting_id)
        message = f"Note generation failed: {exc}"
        await ctx.db.fail_generated_note(meeting_id, message)
        await ctx.bus.broadcast(
            "note.status", {"meeting_id": meeting_id, "status": "error", "error": message}
        )


@router.get("/meetings")
async def list_meetings(request: Request) -> list[dict[str, Any]]:
    return await _ctx(request).db.list_meetings()


@router.put("/meetings/{meeting_id}/title")
async def rename_meeting(request: Request, meeting_id: str, body: RenameBody) -> dict[str, Any]:
    ctx = _ctx(request)
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="The title must not be empty.")
    if not await ctx.db.rename_meeting(meeting_id, title):
        raise HTTPException(status_code=404, detail="No such meeting.")
    return {"id": meeting_id, "title": title}


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(request: Request, meeting_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    if ctx.session is not None and ctx.session.meeting_id == meeting_id:
        raise HTTPException(status_code=409, detail="End the meeting before deleting it.")
    note_task = ctx.note_tasks.pop(meeting_id, None)
    if note_task is not None and not note_task.done():
        note_task.cancel()
    if not await ctx.db.delete_meeting(meeting_id):
        raise HTTPException(status_code=404, detail="No such meeting.")
    return {"deleted": meeting_id}


@router.get("/meetings/{meeting_id}/transcript")
async def get_transcript(request: Request, meeting_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    meeting = await ctx.db.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="No such meeting.")
    segments = await ctx.db.get_segments(meeting_id)
    return {"meeting": meeting, "segments": segments}


# -- notes ---------------------------------------------------------------


async def _require_meeting(ctx: AppContext, meeting_id: str) -> dict[str, Any]:
    meeting = await ctx.db.get_meeting(meeting_id)
    if meeting is None:
        raise HTTPException(status_code=404, detail="No such meeting.")
    return meeting


@router.put("/meetings/{meeting_id}/rough-notes")
async def save_rough_notes(
    request: Request, meeting_id: str, body: RoughNotesBody
) -> dict[str, Any]:
    ctx = _ctx(request)
    await _require_meeting(ctx, meeting_id)
    lines = [{"ts_ms": line.ts_ms, "text": line.text} for line in body.lines]
    await ctx.db.save_rough_notes(meeting_id, json.dumps(lines))
    return {"saved": True, "lines": len(lines)}


@router.get("/meetings/{meeting_id}/rough-notes")
async def get_rough_notes(request: Request, meeting_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    await _require_meeting(ctx, meeting_id)
    raw = await ctx.db.get_rough_notes(meeting_id)
    return {"lines": json.loads(raw) if raw else []}


def _note_payload(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"status": None}
    return {
        "status": row["status"],
        "template": row["template"],
        "markdown": row["markdown"],
        "action_items": json.loads(row["action_items"]) if row["action_items"] else [],
        "decisions": json.loads(row["decisions"]) if row["decisions"] else [],
        "model": row["model"],
        "error": row["error"],
        "generated_at": row["generated_at"],
    }


@router.get("/meetings/{meeting_id}/note")
async def get_note(request: Request, meeting_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    await _require_meeting(ctx, meeting_id)
    payload = _note_payload(await ctx.db.get_generated_note(meeting_id))
    payload["templates"] = list(TEMPLATES)
    return payload


@router.put("/meetings/{meeting_id}/note")
async def edit_note(request: Request, meeting_id: str, body: NoteEditBody) -> dict[str, Any]:
    ctx = _ctx(request)
    await _require_meeting(ctx, meeting_id)
    if not await ctx.db.update_note_markdown(meeting_id, body.markdown):
        raise HTTPException(status_code=409, detail="No finished note to edit yet.")
    return {"saved": True}


@router.post("/meetings/{meeting_id}/note/regenerate")
async def regenerate_note(
    request: Request, meeting_id: str, body: RegenerateBody
) -> dict[str, Any]:
    ctx = _ctx(request)
    meeting = await _require_meeting(ctx, meeting_id)
    if meeting["status"] != "ended":
        raise HTTPException(status_code=409, detail="End the meeting first.")
    existing = ctx.note_tasks.get(meeting_id)
    if existing is not None and not existing.done():
        raise HTTPException(status_code=409, detail="A note is already being generated.")
    api_key = await asyncio.to_thread(secrets.get_anthropic_key)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail={"message": "Anthropic API key is not configured.", "code": "no_anthropic_key"},
        )
    template = body.template or DEFAULT_TEMPLATE
    if template not in TEMPLATES:
        raise HTTPException(status_code=400, detail=f"Unknown template {template!r}.")
    _start_note_generation(ctx, meeting_id, template, api_key)
    return {"status": "generating", "template": template}


@router.get("/meetings/{meeting_id}/note.md")
async def export_note(request: Request, meeting_id: str) -> PlainTextResponse:
    ctx = _ctx(request)
    meeting = await _require_meeting(ctx, meeting_id)
    note = await ctx.db.get_generated_note(meeting_id)
    if note is None or note["status"] != "done" or not note["markdown"]:
        raise HTTPException(status_code=404, detail="No finished note for this meeting.")
    filename = (
        "".join(
            c if c.isalnum() or c in " -_" else "_" for c in (meeting["title"] or "meeting")
        ).strip()
        or "meeting"
    )
    return PlainTextResponse(
        note["markdown"],
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}.md"'},
    )


# -- knowledge base ------------------------------------------------------


@router.get("/kb/docs")
async def kb_docs(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    return {
        "docs": await ctx.db.list_kb_docs(),
        "vec_available": ctx.db.vec_available,
        "copilot_enabled": await _copilot_effective_enabled(ctx),
        "relevance_floor": await _copilot_floor(ctx),
    }


@router.post("/kb/ingest")
async def kb_ingest(request: Request, body: KbIngestBody) -> dict[str, Any]:
    ctx = _ctx(request)
    if not body.path and not body.include_meetings:
        raise HTTPException(status_code=400, detail="Provide a path and/or include_meetings.")
    embedder = ctx.embedder_factory() if ctx.db.vec_available else None
    ingested: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    if body.path:
        path = Path(body.path).expanduser()  # noqa: ASYNC240 — env lookup, no filesystem I/O
        if not await asyncio.to_thread(path.exists):
            raise HTTPException(status_code=404, detail=f"No such path: {path}")
        result = await ingest_paths(ctx.db, embedder, [path])
        ingested += result.ingested
        skipped += result.skipped
        errors += result.errors
    if body.include_meetings:
        result = await ingest_meetings(ctx.db, embedder)
        ingested += result.ingested
        skipped += result.skipped
        errors += result.errors
    return {"ingested": ingested, "skipped": skipped, "errors": errors}


@router.delete("/kb/docs/{doc_id}")
async def kb_delete_doc(request: Request, doc_id: str) -> dict[str, Any]:
    if not await _ctx(request).db.delete_kb_doc(doc_id):
        raise HTTPException(status_code=404, detail="No such document.")
    return {"deleted": doc_id}


@router.post("/kb/docs/{doc_id}/reingest")
async def kb_reingest_doc(request: Request, doc_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    docs = {d["id"]: d for d in await ctx.db.list_kb_docs()}
    doc = docs.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="No such document.")
    embedder = ctx.embedder_factory() if ctx.db.vec_available else None
    if str(doc["path"]).startswith("meeting://"):
        result = await ingest_meetings(ctx.db, embedder)
    else:
        path = Path(str(doc["path"]))
        if not await asyncio.to_thread(path.exists):
            raise HTTPException(status_code=404, detail=f"Source file is gone: {path}")
        await ctx.db.delete_kb_doc(doc_id)  # force re-ingest even if unchanged
        result = await ingest_paths(ctx.db, embedder, [path])
    if result.errors:
        raise HTTPException(status_code=500, detail="; ".join(result.errors))
    return {"reingested": result.ingested or result.skipped}


@router.put("/settings/copilot")
async def set_copilot_settings(request: Request, body: CopilotSettingsBody) -> dict[str, Any]:
    ctx = _ctx(request)
    if body.enabled is not None:
        await ctx.db.set_setting(COPILOT_ENABLED_KEY, "1" if body.enabled else "0")
    if body.relevance_floor is not None:
        if not 0.0 <= body.relevance_floor <= 1.0:
            raise HTTPException(status_code=400, detail="relevance_floor must be in [0, 1].")
        await ctx.db.set_setting(COPILOT_FLOOR_KEY, str(body.relevance_floor))
    return {
        "copilot_enabled": await _copilot_effective_enabled(ctx),
        "relevance_floor": await _copilot_floor(ctx),
    }


# -- copilot (live) ------------------------------------------------------


def _require_copilot(ctx: AppContext) -> CopilotEngine:
    if ctx.copilot is None:
        raise HTTPException(
            status_code=409,
            detail="The copilot is not running (no meeting, disabled, or empty KB).",
        )
    return ctx.copilot


@router.post("/copilot/trigger")
async def copilot_trigger(request: Request) -> dict[str, Any]:
    engine = _require_copilot(_ctx(request))
    await engine.trigger(TriggerKind.HOTKEY)
    return {"triggered": True}


@router.post("/copilot/cards/{card_id}/dismiss")
async def copilot_dismiss(request: Request, card_id: str) -> dict[str, Any]:
    engine = _require_copilot(_ctx(request))
    card = engine.dismiss(card_id)
    if card is None:
        raise HTTPException(status_code=404, detail="No such card.")
    return {"dismissed": card_id, "doc_silenced": card.doc_title}


@router.post("/copilot/cards/{card_id}/insert")
async def copilot_insert(request: Request, card_id: str) -> dict[str, Any]:
    ctx = _ctx(request)
    engine = _require_copilot(ctx)
    card = engine.cards.get(card_id)
    if card is None:
        raise HTTPException(status_code=404, detail="No such card.")
    meeting = await ctx.db.get_meeting(engine.meeting_id)
    started_at = int(meeting["started_at"]) if meeting and meeting["started_at"] else now_ms()
    ts_ms = max(0, now_ms() - started_at)
    citation = f"> [KB] {first_sentence(card.snippet)} — {card.doc_title}"
    if card.page is not None:
        citation += f", p.{card.page}"
    raw = await ctx.db.get_rough_notes(engine.meeting_id)
    lines: list[dict[str, Any]] = json.loads(raw) if raw else []
    lines.append({"ts_ms": ts_ms, "text": citation})
    await ctx.db.save_rough_notes(engine.meeting_id, json.dumps(lines))
    return {"inserted": True, "line": citation, "ts_ms": ts_ms}


@router.put("/settings/deepgram-key")
async def set_deepgram_key(body: DeepgramKeyBody) -> dict[str, Any]:
    key = body.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="The key must not be empty.")
    await asyncio.to_thread(secrets.set_deepgram_key, key)
    return {"configured": True}


@router.put("/settings/anthropic-key")
async def set_anthropic_key(body: DeepgramKeyBody) -> dict[str, Any]:
    key = body.key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="The key must not be empty.")
    await asyncio.to_thread(secrets.set_anthropic_key, key)
    return {"configured": True}


@router.get("/settings/status")
async def settings_status(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    deepgram = await asyncio.to_thread(secrets.get_deepgram_key) is not None
    anthropic_configured = await asyncio.to_thread(secrets.get_anthropic_key) is not None
    return {
        "deepgram_configured": deepgram,  # never the keys themselves
        "anthropic_configured": anthropic_configured,
        "active_meeting_id": ctx.session.meeting_id if ctx.session else None,
        "system_device": await ctx.db.get_setting(SYSTEM_DEVICE_KEY),
    }


@router.get("/settings/devices")
async def get_devices(request: Request) -> dict[str, Any]:
    try:
        devices = await asyncio.to_thread(list_input_devices)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not enumerate audio devices: {exc}"
        ) from exc
    return {
        "devices": devices,
        "blackhole_detected": any(d["is_blackhole"] for d in devices),
    }


@router.put("/settings/system-device")
async def set_system_device(request: Request, body: SystemDeviceBody) -> dict[str, Any]:
    ctx = _ctx(request)
    name = body.device_name.strip() if body.device_name else None
    if name is not None:
        index = await asyncio.to_thread(find_device_index, name)
        if index is None:
            raise HTTPException(status_code=404, detail=f'No input device named "{name}".')
    await ctx.db.set_setting(SYSTEM_DEVICE_KEY, name)
    return {"system_device": name}


async def _stop_probe(ctx: AppContext) -> None:
    if ctx.probe_timeout is not None:
        ctx.probe_timeout.cancel()
        ctx.probe_timeout = None
    if ctx.probe is not None:
        ctx.probe.stop()
        ctx.probe = None


@router.post("/settings/level-test")
async def start_level_test(request: Request, body: LevelTestBody) -> dict[str, Any]:
    ctx = _ctx(request)
    if ctx.session is not None:
        raise HTTPException(status_code=409, detail="Stop the running meeting first.")
    await _stop_probe(ctx)

    loop = asyncio.get_running_loop()

    def on_level(level: float) -> None:
        task = loop.create_task(
            ctx.bus.broadcast("audio.level", {"device": body.device, "level": level})
        )
        task.add_done_callback(lambda t: t.exception())  # never let it warn unretrieved

    probe = ctx.probe_factory(body.device, on_level)
    try:
        probe.start()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not open that device: {exc}") from exc
    ctx.probe = probe

    async def auto_stop() -> None:
        await asyncio.sleep(LEVEL_TEST_TIMEOUT_S)
        await _stop_probe(ctx)

    ctx.probe_timeout = asyncio.create_task(auto_stop())
    return {"running": True}


@router.delete("/settings/level-test")
async def stop_level_test(request: Request) -> dict[str, Any]:
    await _stop_probe(_ctx(request))
    return {"running": False}


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    ctx = cast("AppContext", websocket.app.state.ctx)
    await websocket.accept()
    ctx.bus.register(websocket)
    # Snapshot so a late-joining tab knows the current state immediately.
    snapshot: dict[str, object] = (
        ctx.session.status_payload() if ctx.session else {"state": "idle", "message": ""}
    )
    try:
        await websocket.send_text(json.dumps({"type": "capture.status", "data": snapshot}))
        while True:
            await websocket.receive_text()  # only serves to detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        ctx.bus.unregister(websocket)
