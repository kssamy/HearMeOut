"""Bounded frame queue with ring-buffer overflow semantics."""

from __future__ import annotations

import asyncio
from collections import deque

# A captured audio frame tagged with its channel (0 = mic/"me", 1 = system/"them").
ChannelFrame = tuple[int, bytes]


class RingQueue[T]:
    """Single-consumer queue of audio frames.

    On overflow the *oldest* item is dropped and counted, so a stalled
    consumer loses the most stale audio instead of ever blocking the
    capture callback's thread-safe handoff.
    """

    def __init__(self, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = maxsize
        self._items: deque[T] = deque()
        self._ready = asyncio.Event()
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._items)

    def put_nowait(self, item: T) -> None:
        if len(self._items) >= self._maxsize:
            self._items.popleft()
            self.dropped += 1
        self._items.append(item)
        self._ready.set()

    def get_nowait(self) -> T | None:
        return self._items.popleft() if self._items else None

    async def get(self) -> T:
        while not self._items:
            self._ready.clear()
            await self._ready.wait()
        return self._items.popleft()
