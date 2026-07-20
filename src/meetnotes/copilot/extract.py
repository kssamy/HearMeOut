"""Query extraction from the transcript window via Claude Haiku.

One small call per cycle. The response must be JSON only; fenced output is
tolerated, anything unparseable is logged and skipped — the loop never dies
on a bad model response. A 2 s timeout falls back to the raw question text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

EXTRACT_MODEL = "claude-haiku-4-5-20251001"
EXTRACT_TIMEOUT_S = 2.0
MAX_QUERIES = 3

EXTRACT_SYSTEM = """\
You extract search queries from a live meeting-transcript window so relevant
reference documents can be retrieved. Return ONLY JSON, no prose, no fences:
{"queries": ["..."], "entities": ["..."], "open_question": "..." or null}
- queries: at most 3 short keyword-style search queries covering what is being
  discussed RIGHT NOW (favor the end of the window).
- entities: product names, features, companies, or document-like terms mentioned.
- open_question: the currently unanswered question verbatim, else null."""


@dataclass(frozen=True, slots=True)
class ExtractResult:
    queries: list[str]
    entities: list[str] = field(default_factory=list)
    open_question: str | None = None


def parse_extract_response(text: str) -> ExtractResult | None:
    """Parse the model's JSON (tolerating code fences). None on any mismatch."""
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Unparseable extraction response: %.120s", text)
        return None
    if not isinstance(data, dict):
        return None
    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list):
        return None
    queries = [str(q).strip() for q in raw_queries if str(q).strip()][:MAX_QUERIES]
    raw_entities = data.get("entities")
    entities = (
        [str(e).strip() for e in raw_entities if str(e).strip()]
        if isinstance(raw_entities, list)
        else []
    )
    open_question = data.get("open_question")
    return ExtractResult(
        queries=queries,
        entities=entities,
        open_question=str(open_question) if open_question else None,
    )


class QueryExtractor:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = EXTRACT_MODEL,
        timeout_s: float = EXTRACT_TIMEOUT_S,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client if client is not None else AsyncAnthropic(api_key=api_key)
        self._model = model
        self._timeout_s = timeout_s
        self.calls = 0  # visible for the "zero calls when OFF" acceptance check

    async def extract(self, window_text: str) -> ExtractResult | None:
        """None means: use the caller's fallback query. Never raises."""
        self.calls += 1
        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model,
                    max_tokens=300,
                    system=EXTRACT_SYSTEM,
                    messages=[{"role": "user", "content": window_text}],
                ),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            logger.debug("Query extraction timed out after %.1fs", self._timeout_s)
            return None
        except Exception as exc:
            logger.warning("Query extraction failed: %s", exc)
            return None
        text = next((b.text for b in response.content if b.type == "text"), "")
        return parse_extract_response(text)
