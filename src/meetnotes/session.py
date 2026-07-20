"""MeetingSession: wires capture → ring → ASR provider → SQLite + browser events."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import uuid4

from .asr.base import AsrProvider, AsrSessionConfig, AsrStatusEvent, TranscriptEvent
from .bus import EventBus
from .capture import SILENCE_PEAK, Capture, peak
from .db import Database, now_ms
from .ring import ChannelFrame, RingQueue

logger = logging.getLogger(__name__)

MIC_CHECK_FRAMES = 20  # ~2s of 100ms frames used for the silent-mic check

SessionState = Literal["recording", "paused", "ended"]


class MeetingSession:
    def __init__(
        self,
        meeting_id: str,
        *,
        db: Database,
        bus: EventBus,
        provider: AsrProvider,
        captures: list[tuple[int, Capture]],
        ring: RingQueue[ChannelFrame],
        sample_rate: int = 16000,
    ) -> None:
        self.meeting_id = meeting_id
        self.state: SessionState = "recording"
        self._db = db
        self._bus = bus
        self._provider = provider
        self._captures = captures  # (channel, capture); 0 = mic, 1 = system loopback
        self._ring = ring
        self._sample_rate = sample_rate
        self._pump_task: asyncio.Task[None] | None = None
        self._events_task: asyncio.Task[None] | None = None
        self._buffered_ms = 0
        # Optional observer (the copilot engine) called with every final segment.
        self.on_final_segment: Callable[[dict[str, object]], Awaitable[None]] | None = None

    @property
    def channels(self) -> int:
        return max(channel for channel, _ in self._captures) + 1

    async def start(self) -> None:
        await self._provider.start(
            AsrSessionConfig(sample_rate=self._sample_rate, channels=self.channels)
        )
        started: list[Capture] = []
        try:
            for _, capture in self._captures:
                capture.start()  # raises CaptureError if the device can't open
                started.append(capture)
        except Exception:
            for capture in started:
                capture.stop()
            raise
        self._pump_task = asyncio.create_task(self._pump(), name="session-pump")
        self._events_task = asyncio.create_task(self._consume_events(), name="session-events")
        await self._broadcast_status("recording", "Recording.")

    async def pause(self) -> None:
        for _, capture in self._captures:
            capture.stop()
        self.state = "paused"
        await self._db.set_meeting_status(self.meeting_id, "paused")
        await self._broadcast_status("paused", "Paused.")

    async def resume(self) -> None:
        for _, capture in self._captures:
            capture.start()
        self.state = "recording"
        await self._db.set_meeting_status(self.meeting_id, "recording")
        await self._broadcast_status("recording", "Recording.")

    async def end(self) -> None:
        self.state = "ended"
        for _, capture in self._captures:
            capture.stop()
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
        # Flush frames the callbacks queued before the streams stopped.
        while (item := self._ring.get_nowait()) is not None:
            channel, frame = item
            await self._provider.send_audio(frame, channel=channel)
        await self._provider.finalize()
        if self._events_task is not None:
            await self._events_task
        await self._db.end_meeting(self.meeting_id, now_ms())
        # Note generation on end is kicked off by the API layer (api.end_meeting),
        # which owns the background-task registry.
        await self._bus.broadcast("meeting.ended", {"meeting_id": self.meeting_id})

    def status_payload(self) -> dict[str, object]:
        return {
            "meeting_id": self.meeting_id,
            "state": self.state,
            "message": "Recording." if self.state == "recording" else "Paused.",
        }

    # -- internals ---------------------------------------------------------

    async def _broadcast_status(
        self,
        state: str,
        message: str,
        *,
        code: str | None = None,
        buffered_s: int | None = None,
    ) -> None:
        data: dict[str, object] = {
            "meeting_id": self.meeting_id,
            "state": state,
            "message": message,
        }
        if code is not None:
            data["code"] = code
        if buffered_s is not None:
            data["buffered_s"] = buffered_s
        await self._bus.broadcast("capture.status", data)

    async def _pump(self) -> None:
        checked = 0
        max_peak = 0
        try:
            while True:
                channel, frame = await self._ring.get()
                if channel == 0 and checked < MIC_CHECK_FRAMES:
                    max_peak = max(max_peak, peak(frame))
                    checked += 1
                    if checked == MIC_CHECK_FRAMES and max_peak <= SILENCE_PEAK:
                        await self._broadcast_status(
                            "error",
                            "No microphone signal detected — macOS may be blocking "
                            "microphone access for this app.",
                            code="mic_silent",
                        )
                await self._provider.send_audio(frame, channel=channel)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Audio pump failed")
            await self._broadcast_status("error", f"Audio pipeline failure: {exc}")

    async def _consume_events(self) -> None:
        try:
            async for event in self._provider.events():
                if isinstance(event, TranscriptEvent):
                    await self._handle_transcript(event)
                else:
                    await self._handle_status(event)
        except Exception as exc:
            logger.exception("ASR event consumer failed")
            await self._broadcast_status("error", f"Transcription failure: {exc}")

    async def _handle_transcript(self, event: TranscriptEvent) -> None:
        payload: dict[str, object] = {
            "meeting_id": self.meeting_id,
            "speaker": event.speaker,
            "channel": event.channel,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
            "text": event.text,
            "confidence": event.confidence,
        }
        if event.kind == "final":
            # Persist immediately: a kill -9 loses at most the interim segment.
            await self._db.add_segment(
                uuid4().hex,
                self.meeting_id,
                speaker=event.speaker,
                channel=event.channel,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                text=event.text,
                confidence=event.confidence,
            )
            await self._bus.broadcast("transcript.final", payload)
            if self.on_final_segment is not None:
                try:
                    await self.on_final_segment(payload)
                except Exception:
                    logger.exception("Final-segment listener failed")
        else:
            await self._bus.broadcast("transcript.interim", payload)

    async def _handle_status(self, event: AsrStatusEvent) -> None:
        self._buffered_ms = event.buffered_ms
        if event.state == "connected":
            message = "Recording." if self.state == "recording" else "Paused."
            await self._broadcast_status(self.state, message)
        elif event.state == "reconnecting":
            await self._broadcast_status(
                "reconnecting",
                event.message,
                buffered_s=event.buffered_ms // 1000,
            )
        elif event.state == "buffer_warning":
            await self._broadcast_status(
                self.state,
                event.message,
                code="buffer_warning",
                buffered_s=event.buffered_ms // 1000,
            )
        elif event.state == "error":
            await self._broadcast_status("error", event.message)
        # "closed" is expected during end(); meeting.ended covers the UI side.
