"""Pairs per-channel mono PCM into interleaved multichannel frames.

The mic and the loopback device run on independent hardware clocks, so their
callbacks are not in lockstep. The mixer buffers each channel and emits the
interleaved common prefix; if one channel lags more than `max_skew_ms` behind
(e.g. a stalled device), the laggard is padded with silence so the healthy
channel's audio keeps flowing to ASR.
"""

from __future__ import annotations

from array import array

SAMPLE_WIDTH = 2  # int16


def interleave(channels: list[bytes]) -> bytes:
    """Interleave equal-length mono int16 PCM buffers sample-by-sample."""
    if len(channels) == 1:
        return channels[0]
    parts = []
    for data in channels:
        samples = array("h")
        samples.frombytes(data)
        parts.append(samples)
    n = len(parts[0])
    out = array("h", bytes(len(parts) * n * SAMPLE_WIDTH))
    for idx, samples in enumerate(parts):
        out[idx :: len(parts)] = samples
    return out.tobytes()


class ChannelMixer:
    def __init__(self, channels: int, *, sample_rate: int = 16000, max_skew_ms: int = 500) -> None:
        if channels < 2:
            raise ValueError("ChannelMixer is only needed for 2+ channels")
        self._channels = channels
        self._max_skew_bytes = sample_rate * SAMPLE_WIDTH * max_skew_ms // 1000
        self._buffers: list[bytearray] = [bytearray() for _ in range(channels)]

    def add(self, channel: int, data: bytes) -> bytes:
        """Feed mono PCM for one channel; returns interleaved PCM ready to send
        (possibly empty)."""
        self._buffers[channel].extend(data)
        if max(len(b) for b in self._buffers) >= self._max_skew_bytes:
            self._pad_laggards()
        return self._drain_common()

    def flush(self) -> bytes:
        """Pad every channel to the longest and interleave the remainder."""
        self._pad_laggards()
        return self._drain_common()

    def _pad_laggards(self) -> None:
        target = max(len(b) for b in self._buffers)
        for buf in self._buffers:
            if len(buf) < target:
                buf.extend(bytes(target - len(buf)))

    def _drain_common(self) -> bytes:
        common = min(len(b) for b in self._buffers)
        common -= common % SAMPLE_WIDTH
        if common == 0:
            return b""
        chunk = [bytes(buf[:common]) for buf in self._buffers]
        for buf in self._buffers:
            del buf[:common]
        return interleave(chunk)
