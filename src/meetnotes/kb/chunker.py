"""Split extracted blocks into ~500-token chunks with 60-token overlap.

Token counts use the same chars/4 heuristic as notegen. Chunks prefer to end
on block (heading/paragraph) boundaries; a single oversized block is split on
sentence boundaries with overlap carried between the pieces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import Block

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 60
CHARS_PER_TOKEN = 4

CHUNK_CHARS = CHUNK_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    heading: str
    page: int | None
    start_char: int
    end_char: int


def chunk_blocks(
    blocks: list[Block],
    *,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: list[Block] = []
    pending_len = 0

    def flush() -> None:
        nonlocal pending, pending_len
        if not pending:
            return
        text = "\n\n".join(b.text for b in pending)
        first = pending[0]
        chunks.append(
            Chunk(
                text=text,
                heading=first.heading,
                page=first.page,
                start_char=first.start_char,
                end_char=first.start_char + len(text),
            )
        )
        pending = []
        pending_len = 0

    for block in blocks:
        if len(block.text) > chunk_chars:
            flush()
            chunks.extend(_split_long_block(block, chunk_chars, overlap_chars))
            continue
        boundary_change = pending and (
            block.heading != pending[0].heading or block.page != pending[0].page
        )
        if boundary_change or pending_len + len(block.text) > chunk_chars:
            flush()
        pending.append(block)
        pending_len += len(block.text) + 2
    flush()
    return chunks


def _split_long_block(block: Block, chunk_chars: int, overlap_chars: int) -> list[Chunk]:
    sentences = _sentences(block.text)
    pieces: list[Chunk] = []
    current = ""
    current_start = 0  # offset within the block text

    def flush(next_start: int) -> None:
        nonlocal current, current_start
        text = current.strip()
        if text:
            pieces.append(
                Chunk(
                    text=text,
                    heading=block.heading,
                    page=block.page,
                    start_char=block.start_char + current_start,
                    end_char=block.start_char + current_start + len(text),
                )
            )
        current = ""
        current_start = next_start

    offset = 0
    for sentence in sentences:
        if current and len(current) + len(sentence) > chunk_chars:
            overlap = current[-overlap_chars:] if overlap_chars else ""
            flush(offset - len(overlap))
            current = overlap
        if not current:
            current_start = offset if not current else current_start
        current += sentence
        offset += len(sentence)
    flush(offset)
    return pieces


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for i, part in enumerate(parts):
        out.append(part + (" " if i < len(parts) - 1 else ""))
    return out
