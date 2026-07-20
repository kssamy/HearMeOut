"""Reciprocal rank fusion for merging ranked retrieval lists."""

from __future__ import annotations

RRF_K = 60


def rrf_merge(ranked_lists: list[list[str]], *, k: int = RRF_K) -> list[str]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (k + rank).

    Returns ids best-first. Ties break by first appearance for determinism.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores, key=lambda item: (-scores[item], first_seen[item]))
