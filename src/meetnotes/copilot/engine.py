"""CopilotEngine: watch the live transcript, retrieve KB context, emit cards.

Runs only while the meeting is recording. Every stage failure downgrades to a
`copilot.status` event — the engine never crashes the meeting session.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from ..bus import EventBus
from ..db import Database
from ..kb.embedder import Embedder
from .extract import ExtractResult, QueryExtractor
from .rrf import rrf_merge
from .suppression import (
    AUTO_RATE_CAP_S,
    RELEVANCE_FLOOR_DEFAULT,
    SuppressionState,
    can_auto_fire,
    filter_candidates,
    record_dismissed,
    record_shown,
)
from .triggers import EntityIndex, TriggerDetector, TriggerKind, is_question

logger = logging.getLogger(__name__)

WINDOW_S = 60.0
TICK_S = 5.0
VECTOR_K = 8
KEYWORD_K = 8
TOP_CHUNKS = 3
SNIPPET_CHARS = 280


class RecordingStateLike(Protocol):
    """What the engine needs from MeetingSession: its current state."""

    @property
    def state(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Card:
    id: str
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_path: str
    heading: str
    page: int | None
    snippet: str
    trigger_kind: str


def make_snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    flattened = re.sub(r"\s+", " ", text).strip()
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1].rsplit(" ", 1)[0] + "…"


def first_sentence(text: str) -> str:
    flattened = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"[.!?](\s|$)", flattened)
    return flattened[: match.end()].strip() if match else flattened[:160]


class CopilotEngine:
    def __init__(
        self,
        meeting_id: str,
        *,
        db: Database,
        bus: EventBus,
        session: RecordingStateLike,
        embedder: Embedder | None,
        extractor: QueryExtractor | None,
        relevance_floor: float = RELEVANCE_FLOOR_DEFAULT,
        rate_cap_s: float = AUTO_RATE_CAP_S,
        window_s: float = WINDOW_S,
    ) -> None:
        self.meeting_id = meeting_id
        self._db = db
        self._bus = bus
        self._session = session
        self._embedder = embedder
        self._extractor = extractor
        self._floor = relevance_floor
        self._rate_cap_s = rate_cap_s
        self._window_s = window_s
        self._window: deque[tuple[float, str]] = deque()
        self._suppression = SuppressionState()
        self._detector: TriggerDetector | None = None
        self._tick_task: asyncio.Task[None] | None = None
        self._cycle_task: asyncio.Task[None] | None = None
        self.cards: dict[str, Card] = {}

    async def start(self) -> None:
        self._detector = TriggerDetector(EntityIndex(await self._db.kb_entity_terms()))
        self._tick_task = asyncio.create_task(self._tick_loop(), name="copilot-tick")
        if self._embedder is not None:
            # Warm the embedding model off the hot path so the first card is fast.
            warmup = asyncio.create_task(
                asyncio.to_thread(self._embedder.embed, ["warmup"]), name="copilot-warmup"
            )
            warmup.add_done_callback(lambda t: t.exception())

    async def stop(self) -> None:
        for task in (self._tick_task, self._cycle_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    # -- inputs ------------------------------------------------------------

    async def on_final_segment(self, segment: dict[str, Any]) -> None:
        if self._session.state != "recording" or self._detector is None:
            return
        text = str(segment.get("text") or "")
        now = time.monotonic()
        self._window.append((now, text))
        self._prune_window(now)
        kind = self._detector.note_segment(text, now)
        if kind is not None:
            await self.trigger(kind)

    async def trigger(self, kind: TriggerKind, *, hint: str | None = None) -> None:
        if self._session.state != "recording" and kind != TriggerKind.HOTKEY:
            return
        if self._cycle_task is not None and not self._cycle_task.done():
            return  # one cycle at a time
        if kind != TriggerKind.HOTKEY and not can_auto_fire(
            self._suppression, time.monotonic(), self._rate_cap_s
        ):
            return  # rate-capped: skip before spending a Haiku call
        self._cycle_task = asyncio.create_task(
            self._cycle(kind, hint), name=f"copilot-cycle-{kind.value}"
        )

    def dismiss(self, card_id: str) -> Card | None:
        card = self.cards.get(card_id)
        if card is not None:
            record_dismissed(self._suppression, card.doc_id)
        return card

    # -- internals ---------------------------------------------------------

    def _prune_window(self, now: float) -> None:
        while self._window and now - self._window[0][0] > self._window_s:
            self._window.popleft()

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            if self._session.state != "recording" or self._detector is None:
                continue
            if self._detector.tick(time.monotonic()) is not None:
                await self.trigger(TriggerKind.PERIODIC)

    def _window_text(self) -> str:
        return "\n".join(text for _, text in self._window)

    def _fallback_queries(self, hint: str | None) -> list[str]:
        if hint:
            return [hint]
        for _, text in reversed(self._window):
            if is_question(text):
                return [text]
        recent = [text for _, text in list(self._window)[-2:]]
        return [" ".join(recent)] if recent else []

    async def _cycle(self, kind: TriggerKind, hint: str | None) -> None:
        started = time.monotonic()
        timings: dict[str, float] = {}
        try:
            await self._status("thinking", f"Looking things up ({kind.value})…")
            window_text = self._window_text()
            if not window_text and not hint:
                await self._status("idle", "")
                return

            queries = await self._extract_queries(window_text, hint)
            timings["extract"] = time.monotonic() - started
            if not queries:
                await self._status("idle", "")
                return

            candidates = await self._retrieve(queries)
            timings["retrieve"] = time.monotonic() - started - timings["extract"]

            if kind == TriggerKind.HOTKEY:
                # Hotkey bypasses all suppression (floor, dedupe, dismissals) —
                # only per-doc collapse applies. The user asked; show something.
                kept = filter_candidates(candidates, SuppressionState(), floor=0.0)
                kept = kept[:TOP_CHUNKS]
            else:
                kept = filter_candidates(candidates, self._suppression, floor=self._floor)
                kept = kept[:1]  # one automatic card at a time
            if not kept:
                await self._status("idle", "")
                return

            automatic = kind != TriggerKind.HOTKEY
            record_shown(
                self._suppression,
                [c["id"] for c in kept],
                time.monotonic(),
                automatic=automatic,
            )
            for cand in kept:
                await self._emit_card(cand, kind)
            await self._status("idle", "")
            timings["total"] = time.monotonic() - started
            logger.debug("copilot cycle %s timings: %s", kind.value, timings)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Copilot cycle failed")
            await self._status("error", f"Copilot hiccup: {exc}")

    async def _extract_queries(self, window_text: str, hint: str | None) -> list[str]:
        result: ExtractResult | None = None
        if self._extractor is not None and window_text:
            result = await self._extractor.extract(window_text)
        if result is None:
            return self._fallback_queries(hint)
        queries = list(result.queries)
        if result.open_question and result.open_question not in queries:
            queries.insert(0, result.open_question)
        if hint and hint not in queries:
            queries.insert(0, hint)
        return queries[:3] or self._fallback_queries(hint)

    async def _retrieve(self, queries: list[str]) -> list[dict[str, Any]]:
        """Hybrid: per-query vector + keyword, RRF-fused, top chunks with
        similarity attached where known."""
        similarity: dict[str, float] = {}
        ranked_lists: list[list[str]] = []
        vectors: list[list[float]] = []
        if self._embedder is not None and self._db.vec_available:
            vectors = await asyncio.to_thread(self._embedder.embed, queries)
        for index, query in enumerate(queries):
            if vectors:
                hits = await self._db.kb_vector_search(vectors[index], VECTOR_K)
                ranked_lists.append([cid for cid, _ in hits])
                for cid, sim in hits:
                    similarity[cid] = max(similarity.get(cid, -1.0), sim)
            ranked_lists.append(await self._db.kb_keyword_search(query, KEYWORD_K))
        fused = rrf_merge(ranked_lists)[: TOP_CHUNKS * 3]  # headroom for suppression
        rows = await self._db.get_kb_chunks(fused)
        return [
            {
                "id": row["id"],
                "doc_id": row["doc_id"],
                "doc_title": row["doc_title"],
                "doc_path": row["doc_path"],
                "heading": row["heading"] or "",
                "page": row["page"],
                "text": row["text"],
                "similarity": similarity.get(row["id"]),
            }
            for row in rows[: TOP_CHUNKS * 3]
        ][: TOP_CHUNKS * 3]

    async def _emit_card(self, cand: dict[str, Any], kind: TriggerKind) -> None:
        card = Card(
            id=uuid4().hex,
            chunk_id=cand["id"],
            doc_id=cand["doc_id"],
            doc_title=cand["doc_title"],
            doc_path=cand["doc_path"],
            heading=cand["heading"],
            page=cand["page"],
            snippet=make_snippet(cand["text"]),
            trigger_kind=kind.value,
        )
        self.cards[card.id] = card
        await self._bus.broadcast(
            "copilot.card",
            {
                "meeting_id": self.meeting_id,
                "id": card.id,
                "doc_title": card.doc_title,
                "heading": card.heading,
                "snippet": card.snippet,
                "page": card.page,
                "trigger_kind": card.trigger_kind,
                "source": "meeting" if card.doc_path.startswith("meeting://") else "doc",
            },
        )

    async def _status(self, state: str, message: str) -> None:
        await self._bus.broadcast(
            "copilot.status",
            {"meeting_id": self.meeting_id, "state": state, "message": message},
        )
