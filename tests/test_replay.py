from __future__ import annotations

import asyncio

import pytest

from meetnotes.asr.replay import EndOfStream, ReplayBuffer

# bytes_per_ms=32 → a 3200-byte frame is 100 ms
FRAME = bytes(3200)


def make_buffer(cap_ms: int = 1000) -> ReplayBuffer:
    return ReplayBuffer(cap_ms=cap_ms, bytes_per_ms=32)


async def test_ack_releases_covered_frames() -> None:
    buf = make_buffer()
    for _ in range(5):
        buf.append(FRAME)
    assert buf.buffered_ms == 500
    buf.ack_upto(250)  # covers frames [0,100) and [100,200) fully
    assert buf.buffered_ms == 300


async def test_rewind_resends_unacked() -> None:
    buf = make_buffer()
    buf.append(FRAME)
    buf.append(FRAME)
    first = await buf.next_to_send()
    second = await buf.next_to_send()
    assert (first.offset_ms, second.offset_ms) == (0, 100)
    buf.ack_upto(100)  # first frame acknowledged
    buf.rewind()  # simulated reconnect
    replayed = await buf.next_to_send()
    assert replayed.offset_ms == 100  # unacked frame is replayed, acked one is not
    assert buf.cursor_offset_ms == 200


async def test_cap_drops_oldest_and_counts() -> None:
    buf = make_buffer(cap_ms=300)
    for _ in range(5):
        buf.append(FRAME)
    assert buf.buffered_ms == 300
    assert buf.dropped_ms == 200
    head = await buf.next_to_send()
    assert head.offset_ms == 200  # frames at 0 and 100 ms were dropped


async def test_close_drains_then_raises_end_of_stream() -> None:
    buf = make_buffer()
    buf.append(FRAME)
    buf.close()
    frame = await buf.next_to_send()
    assert frame.offset_ms == 0
    with pytest.raises(EndOfStream):
        await buf.next_to_send()


async def test_next_to_send_wakes_on_append() -> None:
    buf = make_buffer()
    waiter = asyncio.create_task(buf.next_to_send())
    await asyncio.sleep(0.01)
    assert not waiter.done()
    buf.append(FRAME)
    frame = await asyncio.wait_for(waiter, timeout=1)
    assert frame.offset_ms == 0
