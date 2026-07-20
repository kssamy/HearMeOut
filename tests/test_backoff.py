from __future__ import annotations

from itertools import islice

from meetnotes.asr.deepgram import backoff_delays


def test_backoff_doubles_to_cap() -> None:
    delays = list(islice(backoff_delays(0.25, 8.0), 8))
    assert delays == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_backoff_respects_custom_bounds() -> None:
    delays = list(islice(backoff_delays(0.05, 0.1), 4))
    assert delays == [0.05, 0.1, 0.1, 0.1]
