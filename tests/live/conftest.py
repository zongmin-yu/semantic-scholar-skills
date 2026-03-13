from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
import time
from typing import Any

import pytest

from semantic_scholar_skills.core import S2Client, cleanup_client, get_default_client
from semantic_scholar_skills.core.exceptions import S2RateLimitError


_PLACEHOLDER_VALUES = {"", "none", "null", "false"}
_LIVE_REQUEST_DELAY_SECONDS = 1.5
_MAX_429_RETRIES = 3
_429_BASE_WAIT = 5.0

# Module-level timestamp shared across fixture instances so the throttle
# carries over between tests (prevents 429 when tests run in sequence).
# Note: asyncio.Lock cannot be module-level because pytest-asyncio creates
# a new event loop per test; instead we create a lock per fixture invocation.
_global_last_request_at = 0.0


@pytest.fixture(autouse=True)
def require_live_api_key() -> str:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key.lower() in _PLACEHOLDER_VALUES:
        pytest.skip("SEMANTIC_SCHOLAR_API_KEY not set")
    return api_key


@pytest.fixture
async def live_client(require_live_api_key: str) -> AsyncIterator[S2Client]:
    global _global_last_request_at

    client = get_default_client()
    transport = client._transport
    original_request_json = transport.request_json
    request_lock = asyncio.Lock()

    async def throttled_request_json(*args: Any, **kwargs: Any) -> Any:
        global _global_last_request_at

        async with request_lock:
            now = time.monotonic()
            delay = _LIVE_REQUEST_DELAY_SECONDS - (now - _global_last_request_at)
            if delay > 0:
                await asyncio.sleep(delay)
            _global_last_request_at = time.monotonic()

            for attempt in range(_MAX_429_RETRIES + 1):
                try:
                    return await original_request_json(*args, **kwargs)
                except S2RateLimitError:
                    if attempt == _MAX_429_RETRIES:
                        raise
                    wait = _429_BASE_WAIT * (2 ** attempt)
                    await asyncio.sleep(wait)
                    _global_last_request_at = time.monotonic()

    transport.request_json = throttled_request_json
    try:
        yield client
    finally:
        transport.request_json = original_request_json
        await cleanup_client()
