"""Broadcast hub pushing server events to every connected browser tab."""

from __future__ import annotations

import json
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    def register(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, event_type: str, data: dict[str, object]) -> None:
        if not self._clients:
            return
        payload = json.dumps({"type": event_type, "data": data})
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                logger.debug("Dropping dead websocket client", exc_info=True)
                self._clients.discard(ws)
