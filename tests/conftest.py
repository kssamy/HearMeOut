from __future__ import annotations

import hashlib
import math
import struct
import wave
from pathlib import Path

import pytest

from meetnotes.db import Database


class FakeEmbedder:
    """Deterministic, content-sensitive vectors — no torch required."""

    dim = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            seed = hashlib.sha256(text.lower().encode()).digest()
            raw = [seed[i % 32] / 255.0 - 0.5 for i in range(self.dim)]
            # Bias a few dimensions by keyword so related texts land close.
            for i, word in enumerate(("price", "enterprise", "widget", "faq")):
                if word in text.lower():
                    raw[i] = 10.0
            norm = sum(x * x for x in raw) ** 0.5 or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


@pytest.fixture
async def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def wav_fixture(tmp_path: Path) -> Path:
    """2 s of 440 Hz sine, 16 kHz mono int16 — deterministic PCM source."""
    path = tmp_path / "fixture.wav"
    sample_rate = 16000
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [
            int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            for i in range(sample_rate * 2)
        ]
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path
