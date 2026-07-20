"""Microphone capture: sounddevice (PortAudio) → thread-safe handoff → RingQueue.

The PortAudio callback runs on a non-asyncio thread; it does nothing but copy
the frame and hand it to the event loop via `loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import array
import asyncio
import logging
from typing import Any, Protocol

from .ring import ChannelFrame, RingQueue

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_MS = 100
SILENCE_PEAK = 1  # int16 amplitude at/below which a frame counts as silent


class CaptureError(Exception):
    """The input stream could not be opened or started."""


class Capture(Protocol):
    """What MeetingSession needs from an audio source (real mic or test fake)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...


def peak(frame: bytes) -> int:
    """Max absolute int16 amplitude in a PCM frame (mic-permission detection)."""
    samples = array.array("h")
    samples.frombytes(frame)
    return max((abs(s) for s in samples), default=0)


class MicCapture:
    """One mono input stream, tagged with its channel.

    Channel 0 reads the default microphone ("me"); channel 1 reads a BlackHole
    loopback device carrying system audio ("them"). MeetingSession owns one
    instance per registered channel.
    """

    def __init__(
        self,
        ring: RingQueue[ChannelFrame],
        *,
        channel: int = 0,
        sample_rate: int = SAMPLE_RATE,
        block_ms: int = BLOCK_MS,
        device: int | None = None,
    ) -> None:
        self._ring = ring
        self._channel = channel
        self._sample_rate = sample_rate
        self._blocksize = sample_rate * block_ms // 1000
        self._device = device
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        try:
            self._stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._blocksize,
                channels=1,
                dtype="int16",
                device=self._device,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            raise CaptureError(str(exc)) from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.warning("Error closing input stream", exc_info=True)
            self._stream = None

    # Runs on PortAudio's thread — sounddevice callback boundary, kept minimal
    # and untyped-args by design (mypy strictness is relaxed here only).
    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            logger.warning("PortAudio status: %s", status)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._ring.put_nowait, (self._channel, bytes(indata)))
