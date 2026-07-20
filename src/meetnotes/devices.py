"""Input-device enumeration and signal probing for the system-audio setup flow."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from .capture import peak

logger = logging.getLogger(__name__)


def list_input_devices() -> list[dict[str, Any]]:
    """Enumerate audio input devices, flagging BlackHole loopbacks."""
    import sounddevice as sd

    default_input = sd.default.device[0]
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        devices.append(
            {
                "index": index,
                "name": dev["name"],
                "input_channels": dev["max_input_channels"],
                "is_blackhole": "blackhole" in dev["name"].lower(),
                "is_default": index == default_input,
            }
        )
    return devices


def find_device_index(name: str) -> int | None:
    """Resolve a stored device name to today's index (indices shift across
    replug/reboot, so we persist the name)."""
    for dev in list_input_devices():
        if dev["name"] == name:
            return int(dev["index"])
    return None


class LevelProbe:
    """Short-lived input stream that reports peak level per block — drives the
    setup screen's level meter so the user can verify signal is flowing."""

    def __init__(
        self,
        device: int,
        on_level: Callable[[float], None],
        *,
        sample_rate: int = 16000,
        block_ms: int = 100,
    ) -> None:
        self._device = device
        self._on_level = on_level
        self._sample_rate = sample_rate
        self._blocksize = sample_rate * block_ms // 1000
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._blocksize,
            channels=1,
            dtype="int16",
            device=self._device,
            callback=self._on_audio,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.warning("Error closing probe stream", exc_info=True)
            self._stream = None

    # Runs on PortAudio's thread — sounddevice callback boundary.
    def _on_audio(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        level = min(1.0, peak(bytes(indata)) / 32768)
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._on_level, level)
