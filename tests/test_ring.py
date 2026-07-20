from __future__ import annotations

import asyncio

import pytest

from meetnotes.ring import RingQueue


async def test_overflow_drops_oldest_and_counts() -> None:
    q = RingQueue(maxsize=3)
    for i in range(5):
        q.put_nowait(bytes([i]))
    assert q.dropped == 2
    assert len(q) == 3
    # Oldest two (0, 1) were dropped; 2, 3, 4 remain in order.
    assert [await q.get() for _ in range(3)] == [b"\x02", b"\x03", b"\x04"]


async def test_get_waits_for_data() -> None:
    q = RingQueue(maxsize=2)
    getter = asyncio.create_task(q.get())
    await asyncio.sleep(0.01)
    assert not getter.done()
    q.put_nowait(b"x")
    assert await asyncio.wait_for(getter, timeout=1) == b"x"


async def test_get_nowait_empty_returns_none() -> None:
    q = RingQueue(maxsize=2)
    assert q.get_nowait() is None
    q.put_nowait(b"a")
    assert q.get_nowait() == b"a"


def test_rejects_nonpositive_maxsize() -> None:
    with pytest.raises(ValueError):
        RingQueue(maxsize=0)
