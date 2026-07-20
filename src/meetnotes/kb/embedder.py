"""Embedding interface with a local sentence-transformers implementation.

The local model (all-MiniLM-L6-v2, 384-dim) keeps ingestion and retrieval
offline and free. Imports are deferred so the app (and the test suite) never
loads torch unless an embedding is actually needed.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)

EMBED_DIM = 384
_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Sync interface — call via asyncio.to_thread from async code."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LocalEmbedder:
    """sentence-transformers wrapper; the model loads lazily on first use
    (first ever run also downloads ~90 MB of weights)."""

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any = None

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in vec] for vec in vectors]


_local_embedder: LocalEmbedder | None = None


def get_local_embedder() -> LocalEmbedder:
    """Process-wide singleton so the model loads at most once."""
    global _local_embedder
    if _local_embedder is None:
        _local_embedder = LocalEmbedder()
    return _local_embedder
