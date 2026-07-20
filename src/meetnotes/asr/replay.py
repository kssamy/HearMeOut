"""Replay buffer: audio is retained until acknowledged by a final transcript.

Deepgram never explicitly acks audio, so a finalized result covering a time
range acts as the ack for the frames underneath it. On reconnect the sender
rewinds and replays every unacknowledged frame, which is what makes network
drops lossless. The buffer is capped (default 10 minutes); past the cap the
oldest audio is dropped and counted.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass


class EndOfStream(Exception):
    """Raised by next_to_send() once the buffer is closed and fully sent."""


@dataclass(frozen=True, slots=True)
class Frame:
    offset_ms: int
    data: bytes


class ReplayBuffer:
    def __init__(self, *, cap_ms: int = 600_000, bytes_per_ms: int = 32) -> None:
        self._frames: deque[Frame] = deque()
        self._cap_ms = cap_ms
        self._bytes_per_ms = bytes_per_ms
        # frames[:_send_idx] have been sent on the *current* connection
        self._send_idx = 0
        self._next_offset_ms = 0
        self._closed = False
        self._wakeup = asyncio.Event()
        self.dropped_ms = 0

    def _duration_ms(self, frame: Frame) -> int:
        return len(frame.data) // self._bytes_per_ms

    @property
    def cap_ms(self) -> int:
        return self._cap_ms

    @property
    def buffered_ms(self) -> int:
        if not self._frames:
            return 0
        return self._next_offset_ms - self._frames[0].offset_ms

    @property
    def is_drained(self) -> bool:
        return not self._frames

    @property
    def cursor_offset_ms(self) -> int:
        """Timeline position of the next byte the sender will transmit —
        i.e. the absolute offset that a fresh connection's t=0 maps to."""
        if self._send_idx < len(self._frames):
            return self._frames[self._send_idx].offset_ms
        return self._next_offset_ms

    def append(self, data: bytes) -> None:
        if self._closed:
            raise RuntimeError("ReplayBuffer is closed")
        frame = Frame(self._next_offset_ms, data)
        self._next_offset_ms += self._duration_ms(frame)
        self._frames.append(frame)
        while self.buffered_ms > self._cap_ms:
            dropped = self._frames.popleft()
            self.dropped_ms += self._duration_ms(dropped)
            if self._send_idx > 0:
                self._send_idx -= 1
        self._wakeup.set()

    def ack_upto(self, end_ms: int) -> None:
        """Release frames fully covered by a finalized transcript ending at end_ms."""
        while self._frames:
            head = self._frames[0]
            if head.offset_ms + self._duration_ms(head) > end_ms:
                break
            self._frames.popleft()
            if self._send_idx > 0:
                self._send_idx -= 1

    def rewind(self) -> None:
        """Mark all buffered frames unsent — called when a new connection opens."""
        self._send_idx = 0

    def close(self) -> None:
        self._closed = True
        self._wakeup.set()

    async def next_to_send(self) -> Frame:
        while self._send_idx >= len(self._frames):
            if self._closed:
                raise EndOfStream
            self._wakeup.clear()
            await self._wakeup.wait()
        frame = self._frames[self._send_idx]
        self._send_idx += 1
        return frame
