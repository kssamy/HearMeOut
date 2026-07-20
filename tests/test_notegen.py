"""Note generation: formatting, chunking, and the generate flow with a fake client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from meetnotes.notegen import (
    NoteGenerationError,
    NoteGenerator,
    approx_tokens,
    chunk_segments,
    format_rough_notes,
    format_transcript,
)


def seg(text: str, *, start_ms: int = 0, channel: int = 0, speaker: int | None = 0) -> dict:
    return {"text": text, "start_ms": start_ms, "channel": channel, "speaker": speaker}


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


class FakeMessages:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._responses[len(self.calls) - 1]


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(responses)


def note_response(markdown: str = "# Note") -> SimpleNamespace:
    payload = {
        "markdown": markdown,
        "action_items": [
            {"task": "Ship it", "owner": "Sam", "due": "Friday", "source_ts": "01:20"}
        ],
        "decisions": ["Use SQLite"],
    }
    return SimpleNamespace(stop_reason="end_turn", content=[text_block(json.dumps(payload))])


# -- formatting ---------------------------------------------------------------


def test_format_transcript_labels_channels_and_timestamps() -> None:
    text = format_transcript(
        [
            seg("hello there", start_ms=65_000, channel=0, speaker=0),
            seg("hi back", start_ms=70_500, channel=1, speaker=1),
            seg("anonymous", start_ms=0, channel=0, speaker=None),
        ]
    )
    assert text.splitlines() == [
        "[01:05] Me/speaker 0: hello there",
        "[01:10] Them/speaker 1: hi back",
        "[00:00] Me: anonymous",
    ]


def test_format_rough_notes_skips_empty_lines() -> None:
    text = format_rough_notes(
        [
            {"ts_ms": 30_000, "text": "pricing concerns"},
            {"ts_ms": 31_000, "text": "   "},
            {"ts_ms": 90_000, "text": "follow up w/ legal"},
        ]
    )
    assert text.splitlines() == ["[00:30] pricing concerns", "[01:30] follow up w/ legal"]


# -- chunking -----------------------------------------------------------------


def test_chunk_segments_respects_token_limit() -> None:
    segments = [seg("word " * 100, start_ms=i * 1000) for i in range(20)]
    per_seg = approx_tokens("word " * 100) + 12
    chunks = chunk_segments(segments, token_limit=per_seg * 5)
    assert len(chunks) == 4
    assert sum(len(c) for c in chunks) == 20
    # Contiguous and ordered
    flat = [s["start_ms"] for chunk in chunks for s in chunk]
    assert flat == sorted(flat)


def test_chunk_segments_single_oversized_segment() -> None:
    chunks = chunk_segments([seg("x" * 4000)], token_limit=10)
    assert len(chunks) == 1


# -- generate -----------------------------------------------------------------


async def test_generate_expands_notes_with_transcript() -> None:
    client = FakeClient([note_response("# Sprint sync")])
    generator = NoteGenerator("key", client=client)  # type: ignore[arg-type]
    note = await generator.generate(
        title="Sprint sync",
        segments=[seg("we agreed to use SQLite", start_ms=80_000)],
        rough_lines=[{"ts_ms": 79_000, "text": "sqlite decision"}],
        template="general",
    )
    assert note.markdown == "# Sprint sync"
    assert note.action_items[0]["owner"] == "Sam"
    assert note.decisions == ["Use SQLite"]
    assert note.model == "claude-sonnet-4-6"

    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["output_config"]["format"]["type"] == "json_schema"
    user_text = call["messages"][0]["content"]
    # Both the user's notes and the transcript reach the prompt.
    assert "[01:19] sqlite decision" in user_text
    assert "[01:20] Me/speaker 0: we agreed to use SQLite" in user_text
    assert "ground truth" in user_text


async def test_generate_map_reduces_long_transcripts() -> None:
    condensed = SimpleNamespace(
        stop_reason="end_turn", content=[text_block("[00:00] Me: condensed")]
    )
    client = FakeClient([condensed, condensed, note_response()])
    generator = NoteGenerator("key", client=client, chunk_token_limit=60)  # type: ignore[arg-type]
    segments = [seg("alpha " * 30, start_ms=0), seg("beta " * 30, start_ms=1000)]
    note = await generator.generate(
        title="Long", segments=segments, rough_lines=[], template="general"
    )
    assert note.markdown == "# Note"
    # Two condense calls (one per chunk) + one final note call.
    assert len(client.messages.calls) == 3
    final_prompt = client.messages.calls[-1]["messages"][0]["content"]
    assert "condensed" in final_prompt
    assert "alpha" not in final_prompt  # raw transcript replaced by the reduction


async def test_generate_surfaces_refusal_as_error() -> None:
    client = FakeClient([SimpleNamespace(stop_reason="refusal", content=[])])
    generator = NoteGenerator("key", client=client)  # type: ignore[arg-type]
    with pytest.raises(NoteGenerationError):
        await generator.generate(title="t", segments=[], rough_lines=[], template="general")


async def test_generate_rejects_unparseable_output() -> None:
    client = FakeClient([SimpleNamespace(stop_reason="end_turn", content=[text_block("not json")])])
    generator = NoteGenerator("key", client=client)  # type: ignore[arg-type]
    with pytest.raises(NoteGenerationError):
        await generator.generate(title="t", segments=[], rough_lines=[], template="general")
