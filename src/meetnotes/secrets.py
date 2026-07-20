"""API key storage in the OS keychain via `keyring`.

Keys never touch disk, env files, or logs. These calls are synchronous
(Keychain can block on an unlock prompt) — call them via asyncio.to_thread.
"""

from __future__ import annotations

import keyring

_SERVICE = "meetnotes"
_DEEPGRAM_KEY = "deepgram_api_key"
_ANTHROPIC_KEY = "anthropic_api_key"


def get_deepgram_key() -> str | None:
    return keyring.get_password(_SERVICE, _DEEPGRAM_KEY)


def set_deepgram_key(value: str) -> None:
    keyring.set_password(_SERVICE, _DEEPGRAM_KEY, value)


def get_anthropic_key() -> str | None:
    return keyring.get_password(_SERVICE, _ANTHROPIC_KEY)


def set_anthropic_key(value: str) -> None:
    keyring.set_password(_SERVICE, _ANTHROPIC_KEY, value)
