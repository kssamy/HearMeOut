"""Trigger state machine — pure, clock-injected, unit-testable in isolation."""

from __future__ import annotations

import re
from enum import Enum

PERIODIC_INTERVAL_S = 30.0
PERIODIC_MIN_SEGMENTS = 3

_STOPWORDS = frozenset(
    "the a an and or but of for with from this that these those your our their its "  # noqa: SIM905
    "was were are is be been being have has had will would could should can may "
    "what when where which who whom whose why how about into over under between "
    "there here they them then than you not all any some more most other".split()
)


class TriggerKind(Enum):
    PERIODIC = "periodic"
    QUESTION = "question"
    ENTITY = "entity"
    HOTKEY = "hotkey"


def is_question(text: str) -> bool:
    """Question shape: interrogative sentence ending in Deepgram's '?'."""
    stripped = text.strip()
    return stripped.endswith("?") and len(stripped.split()) >= 3


class EntityIndex:
    """In-memory index over KB doc titles and heading terms, built at meeting
    start. Matches whole phrases or distinctive (non-stopword) tokens."""

    def __init__(self, terms: list[str]) -> None:
        self._phrases: set[str] = set()
        self._tokens: set[str] = set()
        for term in terms:
            phrase = term.strip().lower()
            if len(phrase) >= 4:
                self._phrases.add(phrase)
            for token in re.findall(r"[a-z0-9]{4,}", phrase):
                if token not in _STOPWORDS:
                    self._tokens.add(token)

    def match(self, text: str) -> str | None:
        """Returns the matched phrase/token, or None."""
        lowered = text.lower()
        for phrase in self._phrases:
            if phrase in lowered:
                return phrase
        for token in re.findall(r"[a-z0-9]{4,}", lowered):
            if token in self._tokens:
                return str(token)
        return None


class TriggerDetector:
    """Decides when a retrieval cycle should fire. All timestamps are passed
    in (monotonic seconds) so tests control the clock."""

    def __init__(
        self,
        entity_index: EntityIndex,
        *,
        periodic_interval_s: float = PERIODIC_INTERVAL_S,
        periodic_min_segments: int = PERIODIC_MIN_SEGMENTS,
    ) -> None:
        self._entities = entity_index
        self._interval = periodic_interval_s
        self._min_segments = periodic_min_segments
        self._segments_since_periodic = 0
        self._last_periodic_at: float | None = None

    def note_segment(self, text: str, now: float) -> TriggerKind | None:
        """Called on every finalized segment; question/entity win immediately."""
        if self._last_periodic_at is None:
            self._last_periodic_at = now
        self._segments_since_periodic += 1
        if is_question(text):
            return TriggerKind.QUESTION
        if self._entities.match(text) is not None:
            return TriggerKind.ENTITY
        return None

    def tick(self, now: float) -> TriggerKind | None:
        """Called on a timer; fires `periodic` at most every interval, and only
        when enough new material has arrived to be worth a retrieval."""
        if self._last_periodic_at is None:
            self._last_periodic_at = now
            return None
        if now - self._last_periodic_at < self._interval:
            return None
        if self._segments_since_periodic < self._min_segments:
            return None
        self._last_periodic_at = now
        self._segments_since_periodic = 0
        return TriggerKind.PERIODIC

    def entity_match(self, text: str) -> str | None:
        return self._entities.match(text)
