"""Meeting-note generation via the Claude API (claude-sonnet-4-6).

The prompt's core contract: the user's rough notes decide what matters; the
diarized transcript is the ground truth that corrects and completes them. This
must expand the user's notes — not produce a generic meeting summary.

Transcripts over ~150K tokens are chunked and map-reduced: each chunk is
condensed with attribution/timestamps/specifics preserved, then the note is
generated from the condensed transcript.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
NOTE_MAX_TOKENS = 8192
CONDENSE_MAX_TOKENS = 4096
CHUNK_TOKEN_LIMIT = 150_000
APPROX_CHARS_PER_TOKEN = 4

TEMPLATES: dict[str, str] = {
    "general": (
        "Structure: ## Summary (2-4 sentences), ## Notes (the expanded user notes, "
        "organized under the user's own headings/threads), ## Action items, ## Decisions."
    ),
    "standup": (
        "Structure: one ## section per participant with what they did, what they'll do, "
        "and blockers; then ## Action items and ## Decisions."
    ),
    "one_on_one": (
        "Structure: ## Topics discussed (one subsection per topic), ## Feedback, "
        "## Follow-ups, ## Decisions. Keep it personal and specific to the two speakers."
    ),
}
DEFAULT_TEMPLATE = "general"

SYSTEM_PROMPT = """\
You are a meeting-notes assistant inside a note-taking app. During the meeting the user
jotted rough, fragmentary notes; the app also captured a diarized transcript with
timestamps. Channel "Me" is the user; channel "Them" is the other side of the call.

Your job is to EXPAND THE USER'S NOTES into a polished meeting note:
- The user's notes decide what matters: preserve their structure, ordering, emphasis,
  and any shorthand headings they used. Every point the user wrote must appear,
  corrected and completed.
- The transcript is the ground truth: fix inaccuracies in the notes, resolve their
  fragments into full statements, and pull in the concrete specifics they skipped —
  names, numbers, dates, owners, exact phrasing of commitments.
- Do NOT write a generic meeting summary that ignores the user's notes. Only when the
  user took no notes at all should you fall back to summarizing the transcript under
  the template structure.
- Reference transcript timestamps (mm:ss) where they help the reader jump to the moment.
- Action items and decisions must come from the transcript or notes — never invent any.
  For each action item give the task, owner (if identifiable), due date (if stated),
  and the mm:ss transcript timestamp it came from.
- Note lines starting with "> [KB]" are reference snippets the user pulled in from
  their knowledge base during the meeting. Treat them as facts the user flagged as
  relevant: work the cited fact into the note where it belongs and KEEP its source
  attribution (the "— <document title>, p.<page>" suffix) next to the fact."""

# Structured-output schema: guarantees parseable markdown + action_items + decisions.
NOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "markdown": {
            "type": "string",
            "description": "The complete meeting note as GitHub-flavored markdown.",
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "due": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "source_ts": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "mm:ss transcript timestamp backing this item",
                    },
                },
                "required": ["task", "owner", "due", "source_ts"],
                "additionalProperties": False,
            },
        },
        "decisions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["markdown", "action_items", "decisions"],
    "additionalProperties": False,
}


class NoteGenerationError(Exception):
    """Note generation failed with a human-readable reason."""


@dataclass(frozen=True, slots=True)
class GeneratedNote:
    markdown: str
    action_items: list[dict[str, Any]]
    decisions: list[str]
    model: str


class NoteGeneratorLike(Protocol):
    async def generate(
        self,
        *,
        title: str,
        segments: list[dict[str, Any]],
        rough_lines: list[dict[str, Any]],
        template: str,
    ) -> GeneratedNote: ...


def mmss(ms: int) -> str:
    total = max(0, ms // 1000)
    return f"{total // 60:02d}:{total % 60:02d}"


def format_transcript(segments: list[dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        who = "Them" if seg.get("channel") == 1 else "Me"
        speaker = seg.get("speaker")
        label = f"{who}/speaker {speaker}" if speaker is not None else who
        lines.append(f"[{mmss(int(seg.get('start_ms') or 0))}] {label}: {seg['text']}")
    return "\n".join(lines)


def format_rough_notes(rough_lines: list[dict[str, Any]]) -> str:
    lines = []
    for line in rough_lines:
        text = str(line.get("text") or "").rstrip()
        if text:
            lines.append(f"[{mmss(int(line.get('ts_ms') or 0))}] {text}")
    return "\n".join(lines)


def approx_tokens(text: str) -> int:
    return len(text) // APPROX_CHARS_PER_TOKEN


def chunk_segments(
    segments: list[dict[str, Any]], token_limit: int = CHUNK_TOKEN_LIMIT
) -> list[list[dict[str, Any]]]:
    """Split segments into contiguous chunks whose formatted text stays under
    token_limit. A single oversized segment still gets its own chunk."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for seg in segments:
        seg_tokens = approx_tokens(str(seg["text"])) + 12  # label + timestamp overhead
        if current and current_tokens + seg_tokens > token_limit:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(seg)
        current_tokens += seg_tokens
    if current:
        chunks.append(current)
    return chunks


class NoteGenerator:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = MODEL,
        chunk_token_limit: int = CHUNK_TOKEN_LIMIT,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client if client is not None else AsyncAnthropic(api_key=api_key)
        self._model = model
        self._chunk_token_limit = chunk_token_limit

    async def generate(
        self,
        *,
        title: str,
        segments: list[dict[str, Any]],
        rough_lines: list[dict[str, Any]],
        template: str,
    ) -> GeneratedNote:
        transcript = format_transcript(segments)
        if approx_tokens(transcript) > self._chunk_token_limit:
            transcript = await self._map_reduce(segments)
        notes_text = format_rough_notes(rough_lines) or "(the user took no notes)"
        template_text = TEMPLATES.get(template, TEMPLATES[DEFAULT_TEMPLATE])

        user_content = (
            f"Meeting: {title}\n\n"
            f"Template — {template}: {template_text}\n\n"
            f"## The user's rough notes (per-line [mm:ss] timestamps)\n{notes_text}\n\n"
            f"## Diarized transcript (ground truth)\n{transcript or '(no transcript)'}"
        )

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=NOTE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            output_config={"format": {"type": "json_schema", "schema": NOTE_SCHEMA}},
            messages=[{"role": "user", "content": user_content}],
        )
        if response.stop_reason == "refusal":
            raise NoteGenerationError("The model declined to process this transcript.")
        if response.stop_reason == "max_tokens":
            raise NoteGenerationError(
                "The note was too long to generate — try a shorter meeting or regenerate."
            )
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise NoteGenerationError(f"Unparseable model output: {exc}") from exc
        return GeneratedNote(
            markdown=str(data["markdown"]),
            action_items=list(data["action_items"]),
            decisions=[str(d) for d in data["decisions"]],
            model=self._model,
        )

    async def _map_reduce(self, segments: list[dict[str, Any]]) -> str:
        """Condense an oversized transcript chunk by chunk, preserving the facts
        the note generator needs (attribution, timestamps, numbers, commitments)."""
        chunks = chunk_segments(segments, self._chunk_token_limit)
        logger.info("Transcript over token limit; map-reducing %d chunks", len(chunks))
        condensed: list[str] = []
        for index, chunk in enumerate(chunks):
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=CONDENSE_MAX_TOKENS,
                system=(
                    "Condense this meeting-transcript chunk to under a quarter of its "
                    "length. Preserve speaker attribution (Me/Them, speaker numbers), "
                    "[mm:ss] timestamps on every retained point, and ALL decisions, "
                    "action items, names, numbers, and dates verbatim. Output plain "
                    "text lines in the same [mm:ss] Speaker: ... format."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Chunk {index + 1} of {len(chunks)}:\n\n" + format_transcript(chunk)
                        ),
                    }
                ],
            )
            text = next((b.text for b in response.content if b.type == "text"), "")
            condensed.append(text)
        return "\n".join(condensed)
