"""End-to-end: WAV PCM → ring → session → fake ASR → SQLite + broadcasts."""

from __future__ import annotations

import asyncio
import wave
from collections.abc import AsyncIterator
from pathlib import Path

from meetnotes.asr.base import (
    AsrEvent,
    AsrProvider,
    AsrSessionConfig,
    AsrStatusEvent,
    TranscriptEvent,
)
from meetnotes.bus import EventBus
from meetnotes.db import Database
from meetnotes.ring import RingQueue
from meetnotes.session import MeetingSession

FRAME_BYTES = 3200  # 100 ms at 16 kHz mono int16


class FakeAsrProvider(AsrProvider):
    """Scripted provider: interim after 1 s of audio, final after 2 s."""

    def __init__(self) -> None:
        self.received = bytearray()
        self.channel_bytes: dict[int, int] = {}
        self.config: AsrSessionConfig | None = None
        self._events: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self._interim_sent = False
        self._final_sent = False

    async def start(self, session: AsrSessionConfig) -> None:
        self.config = session
        self._events.put_nowait(AsrStatusEvent(state="connected", message="fake connected"))

    async def send_audio(self, frames: bytes, channel: int) -> None:
        self.channel_bytes[channel] = self.channel_bytes.get(channel, 0) + len(frames)
        if channel != 0:
            return
        self.received.extend(frames)
        if len(self.received) >= 16000 and not self._interim_sent:
            self._interim_sent = True
            self._events.put_nowait(
                TranscriptEvent(
                    kind="interim",
                    speaker=None,
                    channel=0,
                    start_ms=0,
                    end_ms=500,
                    text="hello",
                    confidence=0.5,
                )
            )
        if len(self.received) >= 32000 and not self._final_sent:
            self._final_sent = True
            self._events.put_nowait(
                TranscriptEvent(
                    kind="final",
                    speaker=0,
                    channel=0,
                    start_ms=0,
                    end_ms=1000,
                    text="hello world",
                    confidence=0.97,
                )
            )

    async def finalize(self) -> None:
        self._events.put_nowait(
            TranscriptEvent(
                kind="final",
                speaker=1,
                channel=0,
                start_ms=1000,
                end_ms=2000,
                text="closing words",
                confidence=0.9,
            )
        )
        self._events.put_nowait(None)

    async def events(self) -> AsyncIterator[AsrEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event


class FakeCapture:
    def start(self) -> None: ...

    def stop(self) -> None: ...


class RecordingBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[str, dict[str, object]]] = []
        self.final_seen = asyncio.Event()

    async def broadcast(self, event_type: str, data: dict[str, object]) -> None:
        self.messages.append((event_type, data))
        if event_type == "transcript.final":
            self.final_seen.set()


async def test_wav_through_pipeline_to_sqlite(db: Database, wav_fixture: Path) -> None:
    await db.create_meeting("m1", "E2E", 0)
    ring = RingQueue(maxsize=64)
    bus = RecordingBus()
    provider = FakeAsrProvider()
    session = MeetingSession(
        "m1", db=db, bus=bus, provider=provider, captures=[(0, FakeCapture())], ring=ring
    )
    await session.start()

    with wave.open(str(wav_fixture), "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
    # Simulate the sounddevice callback: 100 ms frames into the ring queue.
    for i in range(0, len(pcm), FRAME_BYTES):
        ring.put_nowait((0, pcm[i : i + FRAME_BYTES]))

    await asyncio.wait_for(bus.final_seen.wait(), timeout=5)

    # Crash-safety property: the final segment is already committed while the
    # meeting is still live, before end() runs.
    live_segments = await db.get_segments("m1")
    assert [s["text"] for s in live_segments] == ["hello world"]

    await session.end()

    assert bytes(provider.received) == pcm, "no audio lost between WAV and provider"
    segments = await db.get_segments("m1")
    assert [s["text"] for s in segments] == ["hello world", "closing words"]
    assert segments[0]["speaker"] == 0
    assert segments[1]["speaker"] == 1
    assert all(s["channel"] == 0 for s in segments)

    meeting = await db.get_meeting("m1")
    assert meeting is not None
    assert meeting["status"] == "ended"

    types = [t for t, _ in bus.messages]
    assert "transcript.interim" in types
    assert "transcript.final" in types
    assert "meeting.ended" in types
    interim = next(d for t, d in bus.messages if t == "transcript.interim")
    assert interim["text"] == "hello"


async def test_two_channel_session_attributes_channels(db: Database) -> None:
    """Phase-2 shape: mic + system loopback both flow, tagged by channel."""
    await db.create_meeting("m2", "Two-channel", 0)
    ring = RingQueue(maxsize=64)
    bus = RecordingBus()
    provider = FakeAsrProvider()
    session = MeetingSession(
        "m2",
        db=db,
        bus=bus,
        provider=provider,
        captures=[(0, FakeCapture()), (1, FakeCapture())],
        ring=ring,
    )
    await session.start()
    assert provider.config is not None
    assert provider.config.channels == 2

    mic = bytes(FRAME_BYTES)
    system = b"\x01\x00" * (FRAME_BYTES // 2)
    for _ in range(3):
        ring.put_nowait((0, mic))
        ring.put_nowait((1, system))
    provider._events.put_nowait(  # a "them" final, as multichannel Deepgram would emit
        TranscriptEvent(
            kind="final",
            speaker=0,
            channel=1,
            start_ms=0,
            end_ms=300,
            text="from the other side",
            confidence=0.9,
        )
    )
    await asyncio.wait_for(bus.final_seen.wait(), timeout=5)
    await session.end()

    assert provider.channel_bytes[0] == 3 * FRAME_BYTES
    assert provider.channel_bytes[1] == 3 * FRAME_BYTES
    segments = await db.get_segments("m2")
    them = [s for s in segments if s["text"] == "from the other side"]
    assert them and them[0]["channel"] == 1
