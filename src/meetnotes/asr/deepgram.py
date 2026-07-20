"""Deepgram streaming ASR over a raw WebSocket (no SDK), with reconnect + replay.

Timestamps from Deepgram are relative to the audio start of the *current*
connection. `conn_base_ms` (the buffer cursor at connect time) rebases them
onto the meeting-wide audio timeline so events stay correct across reconnects.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, InvalidHandshake

from ..mixer import ChannelMixer
from .base import AsrEvent, AsrProvider, AsrSessionConfig, AsrStatusEvent, TranscriptEvent
from .replay import EndOfStream, ReplayBuffer

logger = logging.getLogger(__name__)

DEFAULT_URL = "wss://api.deepgram.com/v1/listen"
BUFFER_CAP_MS = 600_000  # keep at most 10 minutes of unacknowledged audio
BUFFER_WARN_FRACTION = 0.8
KEEPALIVE_INTERVAL_S = 5.0
KEEPALIVE_MSG = json.dumps({"type": "KeepAlive"})
CLOSE_STREAM_MSG = json.dumps({"type": "CloseStream"})


def backoff_delays(initial: float = 0.25, cap: float = 8.0) -> Iterator[float]:
    """Yield exponential backoff delays: initial, 2x, 4x, ... capped at `cap`."""
    delay = initial
    while True:
        yield min(delay, cap)
        delay = min(delay * 2.0, cap)


def parse_results(
    msg: dict[str, Any], conn_base_ms: int
) -> tuple[list[TranscriptEvent], int | None]:
    """Convert a Deepgram `Results` message into transcript events.

    Returns the events plus, for finals, the absolute end-ms now covered by
    finalized transcript (used to acknowledge replay-buffered audio). Final
    results are split into contiguous same-speaker word runs so each event
    carries a single diarized speaker.
    """
    channel_index = msg.get("channel_index") or [0]
    channel = int(channel_index[0])
    start_s = float(msg.get("start", 0.0))
    duration_s = float(msg.get("duration", 0.0))
    is_final = bool(msg.get("is_final", False))
    abs_start = conn_base_ms + round(start_s * 1000)
    abs_end = conn_base_ms + round((start_s + duration_s) * 1000)
    ack_ms = abs_end if is_final else None

    alternatives = (msg.get("channel") or {}).get("alternatives") or []
    if not alternatives:
        return [], ack_ms
    alt = alternatives[0]
    transcript = str(alt.get("transcript") or "").strip()
    if not transcript:
        return [], ack_ms
    confidence = float(alt.get("confidence") or 0.0)
    words: list[dict[str, Any]] = alt.get("words") or []

    def _speaker(word: dict[str, Any]) -> int | None:
        value = word.get("speaker")
        return int(value) if value is not None else None

    if not is_final or not words:
        speaker = _speaker(words[-1]) if words else None
        kind: str = "final" if is_final else "interim"
        return [
            TranscriptEvent(
                kind=kind,  # type: ignore[arg-type]
                speaker=speaker,
                channel=channel,
                start_ms=abs_start,
                end_ms=abs_end,
                text=transcript,
                confidence=confidence,
            )
        ], ack_ms

    events: list[TranscriptEvent] = []
    run: list[dict[str, Any]] = []

    def _flush() -> None:
        if not run:
            return
        text = " ".join(str(w.get("punctuated_word") or w.get("word") or "") for w in run).strip()
        if not text:
            return
        events.append(
            TranscriptEvent(
                kind="final",
                speaker=_speaker(run[0]),
                channel=channel,
                start_ms=conn_base_ms + round(float(run[0].get("start", start_s)) * 1000),
                end_ms=conn_base_ms + round(float(run[-1].get("end", start_s)) * 1000),
                text=text,
                confidence=confidence,
            )
        )

    for word in words:
        if run and _speaker(word) != _speaker(run[-1]):
            _flush()
            run = []
        run.append(word)
    _flush()
    return events, ack_ms


class DeepgramProvider(AsrProvider):
    """nova-3 streaming with diarization, lossless across network drops."""

    def __init__(
        self,
        api_key: str,
        *,
        url: str = DEFAULT_URL,
        buffer_cap_ms: int = BUFFER_CAP_MS,
        backoff_initial: float = 0.25,
        backoff_cap: float = 8.0,
    ) -> None:
        self._api_key = api_key
        self._url = url
        self._buffer_cap_ms = buffer_cap_ms
        self._backoff_initial = backoff_initial
        self._backoff_cap = backoff_cap
        self._config = AsrSessionConfig()
        self._events: asyncio.Queue[AsrEvent | None] = asyncio.Queue()
        self._buffer: ReplayBuffer | None = None
        self._mixer: ChannelMixer | None = None
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._warned_buffer = False

    async def start(self, session: AsrSessionConfig) -> None:
        self._config = session
        bytes_per_ms = session.sample_rate * 2 * session.channels // 1000
        self._buffer = ReplayBuffer(cap_ms=self._buffer_cap_ms, bytes_per_ms=bytes_per_ms)
        if session.channels > 1:
            self._mixer = ChannelMixer(session.channels, sample_rate=session.sample_rate)
        self._task = asyncio.create_task(self._run(), name="deepgram-asr")

    async def send_audio(self, frames: bytes, channel: int) -> None:
        if channel >= self._config.channels:
            raise ValueError(f"channel {channel} not registered (channels={self._config.channels})")
        buffer = self._require_buffer()
        if self._mixer is not None:
            # Multichannel: pair mono streams into interleaved PCM first.
            frames = self._mixer.add(channel, frames)
            if not frames:
                return
        buffer.append(frames)
        fraction = buffer.buffered_ms / buffer.cap_ms
        if fraction >= BUFFER_WARN_FRACTION and not self._warned_buffer:
            self._warned_buffer = True
            self._emit(
                AsrStatusEvent(
                    state="buffer_warning",
                    message=(
                        f"Offline audio buffer is {fraction:.0%} full "
                        f"({buffer.buffered_ms // 1000}s); oldest audio will be "
                        "dropped when it fills."
                    ),
                    buffered_ms=buffer.buffered_ms,
                )
            )
        elif fraction < BUFFER_WARN_FRACTION:
            self._warned_buffer = False

    async def finalize(self) -> None:
        self._closing = True
        buffer = self._buffer
        if buffer is not None:
            if self._mixer is not None:
                leftover = self._mixer.flush()
                if leftover:
                    buffer.append(leftover)
            buffer.close()
        if self._task is None:
            self._events.put_nowait(None)
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=10.0)
        except TimeoutError:
            remaining_s = (buffer.buffered_ms if buffer else 0) // 1000
            self._emit(
                AsrStatusEvent(
                    state="error",
                    message=f"Gave up flushing {remaining_s}s of audio to Deepgram.",
                )
            )
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def events(self) -> AsyncIterator[AsrEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    # -- internals ---------------------------------------------------------

    def _require_buffer(self) -> ReplayBuffer:
        if self._buffer is None:
            raise RuntimeError("provider not started")
        return self._buffer

    def _emit(self, event: AsrEvent) -> None:
        self._events.put_nowait(event)

    def _build_url(self) -> str:
        params = {
            "model": "nova-3",
            "encoding": "linear16",
            "sample_rate": str(self._config.sample_rate),
            "channels": str(self._config.channels),
            "diarize": "true",
            "interim_results": "true",
            "smart_format": "true",
            "language": self._config.language,
        }
        if self._config.channels > 1:
            params["multichannel"] = "true"
        return f"{self._url}?{urlencode(params)}"

    async def _run(self) -> None:
        buffer = self._require_buffer()
        delays = backoff_delays(self._backoff_initial, self._backoff_cap)
        attempt = 0
        try:
            while True:
                try:
                    ws = await connect(
                        self._build_url(),
                        additional_headers={"Authorization": f"Token {self._api_key}"},
                        open_timeout=10,
                        close_timeout=3,
                    )
                except Exception as exc:
                    if _is_auth_failure(exc):
                        self._emit(
                            AsrStatusEvent(
                                state="error",
                                message="Deepgram rejected the API key — check Settings.",
                            )
                        )
                        return
                    attempt += 1
                    delay = next(delays)
                    self._emit(
                        AsrStatusEvent(
                            state="reconnecting",
                            message=(
                                f"Can't reach Deepgram ({type(exc).__name__}); "
                                f"retrying in {delay:.2g}s. Audio is being buffered."
                            ),
                            buffered_ms=buffer.buffered_ms,
                            attempt=attempt,
                        )
                    )
                    if self._closing and buffer.is_drained:
                        return
                    await asyncio.sleep(delay)
                    continue

                delays = backoff_delays(self._backoff_initial, self._backoff_cap)
                attempt = 0
                buffer.rewind()
                conn_base_ms = buffer.cursor_offset_ms
                self._emit(AsrStatusEvent(state="connected", message="Connected to Deepgram."))
                sender = asyncio.create_task(self._sender(ws), name="deepgram-sender")
                clean_close = False
                try:
                    async for raw in ws:
                        if isinstance(raw, str):
                            self._handle_text(raw, conn_base_ms)
                    clean_close = True
                except ConnectionClosedOK:
                    clean_close = True
                except ConnectionClosed:
                    clean_close = False
                finally:
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender
                    with contextlib.suppress(Exception):
                        await ws.close()
                if self._closing and clean_close:
                    return
                attempt += 1
                delay = next(delays)
                self._emit(
                    AsrStatusEvent(
                        state="reconnecting",
                        message=(
                            f"Lost the Deepgram connection; retrying in {delay:.2g}s. "
                            "Audio is being buffered."
                        ),
                        buffered_ms=buffer.buffered_ms,
                        attempt=attempt,
                    )
                )
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Deepgram provider crashed")
            self._emit(AsrStatusEvent(state="error", message=f"ASR failure: {exc}"))
        finally:
            self._emit(AsrStatusEvent(state="closed", message="Transcription stream closed."))
            self._events.put_nowait(None)

    async def _sender(self, ws: ClientConnection) -> None:
        buffer = self._require_buffer()
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(
                        buffer.next_to_send(), timeout=KEEPALIVE_INTERVAL_S
                    )
                except TimeoutError:
                    # Nothing to send (e.g. paused) — keep Deepgram from timing out.
                    await ws.send(KEEPALIVE_MSG)
                    continue
                except EndOfStream:
                    await ws.send(CLOSE_STREAM_MSG)
                    return
                await ws.send(frame.data)
        except ConnectionClosed:
            return  # the receive loop handles reconnect

    def _handle_text(self, raw: str, conn_base_ms: int) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Unparseable Deepgram message: %.200s", raw)
            return
        if msg.get("type") != "Results":
            return
        events, ack_ms = parse_results(msg, conn_base_ms)
        if ack_ms is not None:
            self._require_buffer().ack_upto(ack_ms)
        for event in events:
            self._emit(event)


def _is_auth_failure(exc: Exception) -> bool:
    if isinstance(exc, InvalidHandshake):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        return status in (401, 403)
    return False
