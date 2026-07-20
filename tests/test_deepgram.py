"""DeepgramProvider: message parsing plus reconnect/replay against a mock WS server."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from websockets.asyncio.server import Server, ServerConnection, serve

from meetnotes.asr.base import AsrSessionConfig, AsrStatusEvent, TranscriptEvent
from meetnotes.asr.deepgram import DeepgramProvider, parse_results

FRAME = bytes(3200)  # 100 ms at 16 kHz mono int16


def results_msg(
    *,
    start: float,
    duration: float,
    is_final: bool,
    words: list[dict[str, Any]],
    transcript: str,
) -> str:
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "start": start,
            "duration": duration,
            "is_final": is_final,
            "channel": {
                "alternatives": [{"transcript": transcript, "confidence": 0.95, "words": words}]
            },
        }
    )


# -- parse_results ------------------------------------------------------------


def test_parse_interim_event() -> None:
    msg = json.loads(
        results_msg(start=1.0, duration=0.5, is_final=False, transcript="hello", words=[])
    )
    events, ack = parse_results(msg, conn_base_ms=2000)
    assert ack is None
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "interim"
    assert ev.start_ms == 3000  # rebased onto the meeting timeline
    assert ev.end_ms == 3500
    assert ev.text == "hello"


def test_parse_final_splits_speaker_runs() -> None:
    words = [
        {"word": "hi", "punctuated_word": "Hi,", "start": 0.0, "end": 0.3, "speaker": 0},
        {"word": "sam", "punctuated_word": "Sam.", "start": 0.3, "end": 0.6, "speaker": 0},
        {"word": "hey", "punctuated_word": "Hey!", "start": 0.9, "end": 1.2, "speaker": 1},
    ]
    msg = json.loads(
        results_msg(start=0.0, duration=1.5, is_final=True, transcript="hi sam hey", words=words)
    )
    events, ack = parse_results(msg, conn_base_ms=0)
    assert ack == 1500  # final ack covers the full result window
    assert [(e.speaker, e.text) for e in events] == [(0, "Hi, Sam."), (1, "Hey!")]
    assert events[0].start_ms == 0
    assert events[0].end_ms == 600
    assert events[1].start_ms == 900
    assert all(e.kind == "final" for e in events)


def test_parse_empty_transcript_still_acks() -> None:
    msg = json.loads(results_msg(start=0.0, duration=2.0, is_final=True, transcript="", words=[]))
    events, ack = parse_results(msg, conn_base_ms=1000)
    assert events == []
    assert ack == 3000


# -- reconnect / replay -------------------------------------------------------


class FlakyServer:
    """Drops the first connection after 2 frames; records the second fully."""

    def __init__(self) -> None:
        self.connections: list[list[bytes]] = []
        self.close_stream_seen = asyncio.Event()
        self.second_conn_frames = asyncio.Event()

    async def handler(self, ws: ServerConnection) -> None:
        received: list[bytes] = []
        self.connections.append(received)
        conn_no = len(self.connections)
        async for message in ws:
            if isinstance(message, bytes):
                received.append(message)
                if conn_no == 1 and len(received) == 2:
                    await ws.close(code=1011)  # simulate a mid-stream drop
                    return
                if conn_no >= 2 and len(received) >= 4:
                    self.second_conn_frames.set()
            else:
                data = json.loads(message)
                if data.get("type") == "CloseStream":
                    # Ack everything received with one final result.
                    total_s = len(received) * 0.1
                    await ws.send(
                        results_msg(
                            start=0.0,
                            duration=total_s,
                            is_final=True,
                            transcript="replayed audio",
                            words=[],
                        )
                    )
                    self.close_stream_seen.set()
                    await ws.close()
                    return


async def start_server(handler: Any) -> tuple[Server, int]:
    server = await serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def test_reconnect_replays_unacknowledged_audio() -> None:
    flaky = FlakyServer()
    server, port = await start_server(flaky.handler)
    provider = DeepgramProvider(
        "test-key",
        url=f"ws://127.0.0.1:{port}",
        backoff_initial=0.05,
        backoff_cap=0.1,
    )
    await provider.start(AsrSessionConfig())

    events = []

    async def collect() -> None:
        async for ev in provider.events():
            events.append(ev)

    collector = asyncio.create_task(collect())
    try:
        frames = [bytes([i]) * 3200 for i in range(4)]
        for frame in frames:
            await provider.send_audio(frame, channel=0)
            await asyncio.sleep(0.05)

        # The second connection must eventually hold ALL four frames: the two
        # sent-but-unacked frames from connection 1 are replayed, none lost.
        await asyncio.wait_for(flaky.second_conn_frames.wait(), timeout=5)
        await provider.finalize()
        await asyncio.wait_for(collector, timeout=5)
    finally:
        collector.cancel()
        server.close()
        await server.wait_closed()

    assert len(flaky.connections) >= 2
    assert flaky.connections[1] == frames, "replay must resend unacked audio in order"
    states = [e.state for e in events if isinstance(e, AsrStatusEvent)]
    assert "reconnecting" in states
    assert states.count("connected") >= 2
    finals = [e for e in events if isinstance(e, TranscriptEvent) and e.kind == "final"]
    assert len(finals) == 1
    # Connection 2 replayed from offset 0 (nothing was acked), so timestamps
    # stay on the meeting timeline: 4 frames = 400 ms.
    assert finals[0].start_ms == 0
    assert finals[0].end_ms == 400
    assert flaky.close_stream_seen.is_set()


async def test_multichannel_interleaves_before_send() -> None:
    received: list[bytes] = []
    done = asyncio.Event()

    async def handler(ws: ServerConnection) -> None:
        async for message in ws:
            if isinstance(message, bytes):
                received.append(message)
            elif json.loads(message).get("type") == "CloseStream":
                done.set()
                await ws.close()
                return

    server, port = await start_server(handler)
    provider = DeepgramProvider("test-key", url=f"ws://127.0.0.1:{port}")
    await provider.start(AsrSessionConfig(channels=2))
    try:
        mic = b"\x01\x00" * 1600  # 100 ms of value-1 samples
        system = b"\x02\x00" * 1600
        await provider.send_audio(mic, channel=0)
        await provider.send_audio(system, channel=1)
        assert "multichannel=true" in provider._build_url()
        assert "channels=2" in provider._build_url()
        await provider.finalize()
        async for _ in provider.events():
            pass
    finally:
        server.close()
        await server.wait_closed()

    assert done.is_set()
    pairs = b"".join(received)
    assert len(pairs) == 6400  # 100 ms of interleaved stereo int16 (1600 pairs x 4 bytes)
    assert pairs[:8] == b"\x01\x00\x02\x00\x01\x00\x02\x00"  # mic, system, mic, system…


async def test_finalize_flushes_and_closes_cleanly() -> None:
    received: list[bytes] = []
    done = asyncio.Event()

    async def handler(ws: ServerConnection) -> None:
        async for message in ws:
            if isinstance(message, bytes):
                received.append(message)
            elif json.loads(message).get("type") == "CloseStream":
                done.set()
                await ws.close()
                return

    server, port = await start_server(handler)
    provider = DeepgramProvider("test-key", url=f"ws://127.0.0.1:{port}")
    await provider.start(AsrSessionConfig())
    try:
        await provider.send_audio(FRAME, channel=0)
        await provider.finalize()
        events = [ev async for ev in provider.events()]
    finally:
        server.close()
        await server.wait_closed()

    assert done.is_set()
    assert received == [FRAME]
    assert any(isinstance(e, AsrStatusEvent) and e.state == "closed" for e in events)
