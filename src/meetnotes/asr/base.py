"""Provider-agnostic ASR interface.

A new provider (Whisper, Azure, ...) implements `AsrProvider`; callers only
ever see `TranscriptEvent` / `AsrStatusEvent` and never provider wire formats.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    kind: Literal["interim", "final"]
    speaker: int | None
    channel: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class AsrStatusEvent:
    """Connection/buffer health, surfaced to the UI as capture.status events."""

    state: Literal["connected", "reconnecting", "buffer_warning", "error", "closed"]
    message: str
    buffered_ms: int = 0
    attempt: int = 0


AsrEvent = TranscriptEvent | AsrStatusEvent


@dataclass(frozen=True, slots=True)
class AsrSessionConfig:
    sample_rate: int = 16000
    channels: int = 1
    language: str = "en"


class AsrProvider(abc.ABC):
    """Streaming speech-to-text session.

    Lifecycle: `start()` once, `send_audio()` per frame, `finalize()` to flush
    and close. `events()` yields transcript and status events until the
    provider shuts down.
    """

    @abc.abstractmethod
    async def start(self, session: AsrSessionConfig) -> None: ...

    @abc.abstractmethod
    async def send_audio(self, frames: bytes, channel: int) -> None: ...

    @abc.abstractmethod
    async def finalize(self) -> None: ...

    @abc.abstractmethod
    def events(self) -> AsyncIterator[AsrEvent]: ...
