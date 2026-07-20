"""Suppression rules — pure functions over explicit state, each one tested."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RELEVANCE_FLOOR_DEFAULT = 0.35
AUTO_RATE_CAP_S = 120.0


@dataclass
class SuppressionState:
    shown_chunk_ids: set[str] = field(default_factory=set)
    dismissed_doc_ids: set[str] = field(default_factory=set)
    last_auto_card_at: float | None = None


def passes_floor(similarity: float | None, floor: float) -> bool:
    """Vector similarity below the floor shows nothing. Keyword-only hits
    (no similarity score) are not floored — BM25 rank already gated them."""
    return similarity is None or similarity >= floor


def can_auto_fire(state: SuppressionState, now: float, cap_s: float = AUTO_RATE_CAP_S) -> bool:
    return state.last_auto_card_at is None or now - state.last_auto_card_at >= cap_s


def filter_candidates(
    candidates: list[dict[str, Any]],
    state: SuppressionState,
    *,
    floor: float = RELEVANCE_FLOOR_DEFAULT,
) -> list[dict[str, Any]]:
    """Apply floor, chunk dedupe, dismissal memory, and per-doc collapse.

    Candidates are best-first dicts with at least: id, doc_id, similarity
    (float | None). Collapse keeps the best chunk per doc, preserving order.
    """
    seen_docs: set[str] = set()
    kept: list[dict[str, Any]] = []
    for cand in candidates:
        if not passes_floor(cand.get("similarity"), floor):
            continue
        if cand["id"] in state.shown_chunk_ids:
            continue
        if cand["doc_id"] in state.dismissed_doc_ids:
            continue
        if cand["doc_id"] in seen_docs:
            continue  # collapse: one card per doc
        seen_docs.add(cand["doc_id"])
        kept.append(cand)
    return kept


def record_shown(
    state: SuppressionState, chunk_ids: list[str], now: float, *, automatic: bool
) -> None:
    state.shown_chunk_ids.update(chunk_ids)
    if automatic:
        state.last_auto_card_at = now


def record_dismissed(state: SuppressionState, doc_id: str) -> None:
    state.dismissed_doc_ids.add(doc_id)
