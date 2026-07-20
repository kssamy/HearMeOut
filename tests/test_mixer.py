from __future__ import annotations

from array import array

import pytest

from meetnotes.mixer import ChannelMixer, interleave


def pcm(*samples: int) -> bytes:
    return array("h", samples).tobytes()


def test_interleave_two_channels() -> None:
    left = pcm(1, 2, 3)
    right = pcm(10, 20, 30)
    assert interleave([left, right]) == pcm(1, 10, 2, 20, 3, 30)


def test_interleave_mono_passthrough() -> None:
    assert interleave([pcm(5, 6)]) == pcm(5, 6)


def test_mixer_emits_only_common_prefix() -> None:
    mixer = ChannelMixer(2)
    assert mixer.add(0, pcm(1, 2)) == b""  # channel 1 has nothing yet
    out = mixer.add(1, pcm(10))
    assert out == pcm(1, 10)  # one sample pair ready; rest of ch0 stays buffered
    out = mixer.add(1, pcm(20))
    assert out == pcm(2, 20)


def test_mixer_pads_lagging_channel_past_skew() -> None:
    # max_skew_ms=1 at 16kHz → 32 bytes → 16 samples triggers padding.
    mixer = ChannelMixer(2, max_skew_ms=1)
    samples = tuple(range(1, 17))
    out = mixer.add(0, pcm(*samples))
    # Channel 1 never produced audio: it gets silence so ch0 still flows.
    expected = array("h")
    for s in samples:
        expected.extend([s, 0])
    assert out == expected.tobytes()


def test_mixer_flush_pads_remainder() -> None:
    mixer = ChannelMixer(2)
    mixer.add(0, pcm(7, 8))
    assert mixer.flush() == pcm(7, 0, 8, 0)
    assert mixer.flush() == b""


def test_mixer_rejects_mono() -> None:
    with pytest.raises(ValueError):
        ChannelMixer(1)
