"""Copilot units: triggers, suppression, RRF, Haiku parsing — plus e2e engine."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from conftest import FakeEmbedder
from meetnotes.copilot.engine import CopilotEngine
from meetnotes.copilot.extract import QueryExtractor, parse_extract_response
from meetnotes.copilot.rrf import rrf_merge
from meetnotes.copilot.suppression import (
    SuppressionState,
    can_auto_fire,
    filter_candidates,
    passes_floor,
    record_dismissed,
    record_shown,
)
from meetnotes.copilot.triggers import EntityIndex, TriggerDetector, TriggerKind, is_question
from meetnotes.db import Database
from meetnotes.kb.ingest import ingest_paths

# -- triggers -----------------------------------------------------------------


def make_detector(terms: list[str] | None = None) -> TriggerDetector:
    return TriggerDetector(EntityIndex(terms or []))


def test_question_trigger() -> None:
    det = make_detector()
    assert det.note_segment("What was the enterprise price?", now=0.0) == TriggerKind.QUESTION
    assert det.note_segment("We shipped it yesterday.", now=1.0) is None
    assert not is_question("Really?")  # too short to be a retrieval-worthy question


def test_entity_trigger_on_title_and_heading_terms() -> None:
    det = make_detector(["Acme Pricing", "Enterprise"])
    assert det.note_segment("let's talk about acme pricing today", 0.0) == TriggerKind.ENTITY
    assert det.note_segment("the enterprise tier came up", 1.0) == TriggerKind.ENTITY
    assert det.note_segment("nothing relevant here at all", 2.0) is None


def test_entity_index_ignores_stopwords() -> None:
    index = EntityIndex(["The Widget Handbook"])
    assert index.match("this widget is great") is not None
    assert index.match("the handbook says so") is not None
    assert index.match("this that these those") is None


def test_periodic_trigger_needs_interval_and_segments() -> None:
    det = make_detector()
    assert det.tick(0.0) is None  # first tick only arms the timer
    for i in range(3):
        det.note_segment(f"segment {i} words here", now=float(i))
    assert det.tick(10.0) is None  # interval not elapsed
    assert det.tick(31.0) == TriggerKind.PERIODIC
    assert det.tick(62.0) is None  # no new segments since the last fire
    det.note_segment("one", 63.0)
    det.note_segment("two", 64.0)
    assert det.tick(95.0) is None  # only 2 new segments (< 3)
    det.note_segment("three", 96.0)
    assert det.tick(126.0) == TriggerKind.PERIODIC


# -- suppression --------------------------------------------------------------


def cand(cid: str, doc: str, sim: float | None = 0.9) -> dict[str, Any]:
    return {"id": cid, "doc_id": doc, "similarity": sim}


def test_relevance_floor() -> None:
    assert passes_floor(0.36, 0.35)
    assert not passes_floor(0.34, 0.35)
    assert passes_floor(None, 0.35)  # keyword-only hits aren't floored
    state = SuppressionState()
    assert filter_candidates([cand("c1", "d1", sim=0.1)], state, floor=0.35) == []


def test_dedupe_never_reshows_chunk() -> None:
    state = SuppressionState()
    record_shown(state, ["c1"], now=0.0, automatic=True)
    kept = filter_candidates([cand("c1", "d1"), cand("c2", "d2")], state)
    assert [c["id"] for c in kept] == ["c2"]


def test_collapse_one_card_per_doc() -> None:
    kept = filter_candidates(
        [cand("c1", "d1"), cand("c2", "d1"), cand("c3", "d2")], SuppressionState()
    )
    assert [c["id"] for c in kept] == ["c1", "c3"]


def test_rate_cap_auto_only() -> None:
    state = SuppressionState()
    assert can_auto_fire(state, now=0.0)
    record_shown(state, ["c1"], now=0.0, automatic=True)
    assert not can_auto_fire(state, now=60.0)
    assert can_auto_fire(state, now=121.0)
    # Hotkey cards don't reset the auto clock.
    record_shown(state, ["c2"], now=130.0, automatic=False)
    assert can_auto_fire(state, now=130.0)


def test_dismissal_silences_doc() -> None:
    state = SuppressionState()
    record_dismissed(state, "d1")
    kept = filter_candidates([cand("c1", "d1"), cand("c2", "d2")], state)
    assert [c["doc_id"] for c in kept] == ["d2"]


# -- RRF ----------------------------------------------------------------------


def test_rrf_prefers_items_in_both_lists() -> None:
    merged = rrf_merge([["a", "b", "c"], ["b", "d"]])
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_deterministic_tiebreak() -> None:
    assert rrf_merge([["x"], ["y"]]) == ["x", "y"]


# -- Haiku response parsing ---------------------------------------------------


def test_parse_valid_json() -> None:
    result = parse_extract_response(
        '{"queries": ["enterprise price"], "entities": ["Acme"], "open_question": null}'
    )
    assert result is not None
    assert result.queries == ["enterprise price"]
    assert result.entities == ["Acme"]
    assert result.open_question is None


def test_parse_fenced_json() -> None:
    result = parse_extract_response(
        '```json\n{"queries": ["a", "b", "c", "d"], "entities": [], "open_question": "why?"}\n```'
    )
    assert result is not None
    assert result.queries == ["a", "b", "c"]  # capped at 3
    assert result.open_question == "why?"


def test_parse_malformed_returns_none() -> None:
    assert parse_extract_response("not json at all") is None
    assert parse_extract_response('{"nope": true}') is None
    assert parse_extract_response('["list"]') is None


async def test_extractor_timeout_falls_back() -> None:
    class SlowMessages:
        async def create(self, **kwargs: Any) -> Any:
            await asyncio.sleep(1.0)

    extractor = QueryExtractor(
        "key",
        timeout_s=0.05,
        client=SimpleNamespace(messages=SlowMessages()),  # type: ignore[arg-type]
    )
    assert await extractor.extract("some window") is None
    assert extractor.calls == 1


# -- end-to-end engine --------------------------------------------------------


class CannedExtractor:
    """Stands in for Haiku: returns fixed queries; counts calls."""

    def __init__(self, queries: list[str]) -> None:
        self.queries = queries
        self.calls = 0

    async def extract(self, window_text: str) -> Any:
        self.calls += 1
        return SimpleNamespace(queries=list(self.queries), entities=[], open_question=None)


class RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))

    def cards(self) -> list[dict[str, Any]]:
        return [d for t, d in self.events if t == "copilot.card"]


async def make_engine(
    db: Database, extractor: Any, **kwargs: Any
) -> tuple[CopilotEngine, RecordingBus]:
    bus = RecordingBus()
    session = SimpleNamespace(state="recording")
    engine = CopilotEngine(
        "m1",
        db=db,
        bus=bus,  # type: ignore[arg-type]
        session=session,
        embedder=FakeEmbedder() if db.vec_available else None,
        extractor=extractor,
        **kwargs,
    )
    await engine.start()
    return engine, bus


async def seed_kb(db: Database, tmp_path: Any) -> None:
    (tmp_path / "pricing.md").write_text(
        "# Acme Pricing\n\n## Enterprise\n\nThe enterprise price is $99 per seat per month.\n"
    )
    await ingest_paths(db, FakeEmbedder() if db.vec_available else None, [tmp_path])


async def drive_segment(engine: CopilotEngine, text: str) -> None:
    await engine.on_final_segment({"text": text})
    if engine._cycle_task is not None:
        await engine._cycle_task


async def test_e2e_question_to_card(db: Database, tmp_path: Any) -> None:
    await seed_kb(db, tmp_path)
    extractor = CannedExtractor(["enterprise price"])
    engine, bus = await make_engine(db, extractor, relevance_floor=0.0)
    try:
        await drive_segment(engine, "So what is the enterprise price?")
        cards = bus.cards()
        assert len(cards) == 1
        assert cards[0]["doc_title"] == "Acme Pricing"
        assert cards[0]["trigger_kind"] == "question"
        assert "99" in cards[0]["snippet"]
        assert extractor.calls == 1
        # Statuses bookend the cycle: thinking → idle.
        states = [d["state"] for t, d in bus.events if t == "copilot.status"]
        assert states[0] == "thinking" and states[-1] == "idle"
    finally:
        await engine.stop()


async def test_e2e_dismiss_silences_doc_but_hotkey_bypasses(db: Database, tmp_path: Any) -> None:
    await seed_kb(db, tmp_path)
    extractor = CannedExtractor(["enterprise price"])
    engine, bus = await make_engine(db, extractor, relevance_floor=0.0, rate_cap_s=0.0)
    try:
        await drive_segment(engine, "What is the acme pricing enterprise price?")
        card = bus.cards()[0]
        engine.dismiss(card["id"])
        # Same topic again: the doc is silenced, no new automatic card.
        await drive_segment(engine, "Tell me the enterprise price again?")
        assert len(bus.cards()) == 1
        # Hotkey bypasses all suppression: the dismissed doc comes back on demand.
        await engine.trigger(TriggerKind.HOTKEY)
        if engine._cycle_task is not None:
            await engine._cycle_task
        assert len(bus.cards()) == 2
        assert bus.cards()[1]["doc_title"] == "Acme Pricing"
        assert bus.cards()[1]["trigger_kind"] == "hotkey"
    finally:
        await engine.stop()


async def test_e2e_rate_cap_blocks_second_auto_card(db: Database, tmp_path: Any) -> None:
    await seed_kb(db, tmp_path)
    (tmp_path / "other.md").write_text("# Other Doc\n\nThe widget ships in blue.\n")
    await ingest_paths(db, FakeEmbedder() if db.vec_available else None, [tmp_path])
    extractor = CannedExtractor(["enterprise price widget"])
    engine, bus = await make_engine(db, extractor, relevance_floor=0.0)
    try:
        await drive_segment(engine, "What is the enterprise price?")
        assert len(bus.cards()) == 1
        await drive_segment(engine, "And what about the widget color?")
        assert len(bus.cards()) == 1  # rate-capped, and zero extra Haiku calls
        assert extractor.calls == 1
    finally:
        await engine.stop()


async def test_e2e_no_extractor_uses_fallback_query(db: Database, tmp_path: Any) -> None:
    await seed_kb(db, tmp_path)
    engine, bus = await make_engine(db, None, relevance_floor=0.0)
    try:
        await drive_segment(engine, "What is the enterprise price per seat?")
        cards = bus.cards()
        assert len(cards) == 1
        assert cards[0]["doc_title"] == "Acme Pricing"
    finally:
        await engine.stop()


async def test_e2e_floor_suppresses_weak_matches(db: Database, tmp_path: Any) -> None:
    if not db.vec_available:
        return  # floor applies to vector similarity only
    await seed_kb(db, tmp_path)
    extractor = CannedExtractor(["cosmic nebula gardening"])
    engine, bus = await make_engine(db, extractor, relevance_floor=0.99)
    try:
        # A question fires an automatic cycle; the query is unrelated to the KB,
        # so vector hits fall below the (extreme) floor and keyword search finds
        # no shared terms — nothing is shown.
        await drive_segment(engine, "Should we discuss cosmic nebula gardening now?")
        assert bus.cards() == []
    finally:
        await engine.stop()
