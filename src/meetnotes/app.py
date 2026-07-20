"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import webbrowser
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from .api import router
from .asr.base import AsrProvider
from .asr.deepgram import DeepgramProvider
from .bus import EventBus
from .capture import Capture, MicCapture
from .copilot.engine import CopilotEngine
from .db import Database
from .devices import LevelProbe
from .kb.embedder import Embedder, get_local_embedder
from .notegen import NoteGenerator, NoteGeneratorLike
from .ring import ChannelFrame, RingQueue
from .session import MeetingSession

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8756
STATIC_DIR = Path(__file__).parent / "static"

ProviderFactory = Callable[[str], AsrProvider]
# (ring, channel, device_index) -> capture. Channel 0 is the mic (device None
# = system default); channel 1 is the configured system-audio loopback.
CaptureFactory = Callable[[RingQueue[ChannelFrame], int, "int | None"], Capture]
# (device_index, on_level) -> probe for the setup screen's level meter.
ProbeFactory = Callable[[int, Callable[[float], None]], LevelProbe]
# (anthropic_api_key) -> note generator; injectable so tests skip the real API.
NotegenFactory = Callable[[str], NoteGeneratorLike]
# () -> embedder; injectable so tests never load sentence-transformers.
EmbedderFactory = Callable[[], Embedder]


def _default_capture_factory(
    ring: RingQueue[ChannelFrame], channel: int, device: int | None
) -> Capture:
    return MicCapture(ring, channel=channel, device=device)


class NoCacheStaticFiles(StaticFiles):
    """Static files with revalidation: browsers otherwise cache the frontend
    heuristically and keep running stale JS after the app is updated."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def default_db_path() -> Path:
    if override := os.environ.get("MEETNOTES_DB"):
        return Path(override)
    return Path.home() / ".meetnotes" / "meetnotes.db"


@dataclass
class AppContext:
    db: Database
    bus: EventBus = field(default_factory=EventBus)
    provider_factory: ProviderFactory = DeepgramProvider
    capture_factory: CaptureFactory = _default_capture_factory
    probe_factory: ProbeFactory = LevelProbe
    notegen_factory: NotegenFactory = NoteGenerator
    embedder_factory: EmbedderFactory = get_local_embedder
    session: MeetingSession | None = None
    copilot: CopilotEngine | None = None
    probe: LevelProbe | None = None
    probe_timeout: asyncio.Task[None] | None = None
    note_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    open_browser: bool = False


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    ctx: AppContext = app.state.ctx
    await ctx.db.connect()
    recovered = await ctx.db.recover_orphaned_meetings()
    if recovered:
        logger.info("Recovered %d interrupted meeting(s); transcripts are intact.", recovered)
    browser_task: asyncio.Task[None] | None = None
    if ctx.open_browser:
        browser_task = asyncio.create_task(_open_browser_soon())
    try:
        yield
    finally:
        if browser_task is not None:
            browser_task.cancel()
        if ctx.probe_timeout is not None:
            ctx.probe_timeout.cancel()
        if ctx.probe is not None:
            ctx.probe.stop()
        if ctx.copilot is not None:
            with contextlib.suppress(Exception):
                await ctx.copilot.stop()
            ctx.copilot = None
        if ctx.session is not None:
            with contextlib.suppress(Exception):
                await ctx.session.end()
            ctx.session = None
        # Give in-flight note generation a short grace period, then cancel.
        pending = [t for t in ctx.note_tasks.values() if not t.done()]
        if pending:
            _, still_pending = await asyncio.wait(pending, timeout=5.0)
            for task in still_pending:
                task.cancel()
        await ctx.db.close()


async def _open_browser_soon() -> None:
    await asyncio.sleep(0.4)  # let uvicorn bind first
    webbrowser.open(f"http://{HOST}:{PORT}")


def create_app(
    *,
    db_path: Path | None = None,
    provider_factory: ProviderFactory | None = None,
    capture_factory: CaptureFactory | None = None,
    probe_factory: ProbeFactory | None = None,
    notegen_factory: NotegenFactory | None = None,
    embedder_factory: EmbedderFactory | None = None,
    open_browser: bool = False,
) -> FastAPI:
    app = FastAPI(title="MeetNotes", lifespan=_lifespan)
    app.state.ctx = AppContext(
        db=Database(db_path or default_db_path()),
        provider_factory=provider_factory or DeepgramProvider,
        capture_factory=capture_factory or _default_capture_factory,
        probe_factory=probe_factory or LevelProbe,
        notegen_factory=notegen_factory or NoteGenerator,
        embedder_factory=embedder_factory or get_local_embedder,
        open_browser=open_browser,
    )
    app.include_router(router)
    app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def build() -> FastAPI:
    """uvicorn factory target for `uv run meetnotes`."""
    return create_app(open_browser=os.environ.get("MEETNOTES_NO_BROWSER") != "1")
