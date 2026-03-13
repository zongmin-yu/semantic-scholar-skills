from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import os
import time
from typing import Any

import pytest

from semantic_scholar_skills.core import S2Client, cleanup_client, get_default_client


_PLACEHOLDER_VALUES = {"", "none", "null", "false"}
_LIVE_REQUEST_DELAY_SECONDS = 1.5


@pytest.fixture(autouse=True)
def require_live_api_key() -> str:
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key.lower() in _PLACEHOLDER_VALUES:
        pytest.skip("SEMANTIC_SCHOLAR_API_KEY not set")
    return api_key


@pytest.fixture
async def live_client(require_live_api_key: str) -> AsyncIterator[S2Client]:
    client = get_default_client()
    transport = client._transport
    request_lock = asyncio.Lock()
    last_request_started_at = 0.0
    original_request_json = transport.request_json

    async def throttled_request_json(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_request_started_at

        async with request_lock:
            now = time.monotonic()
            delay = _LIVE_REQUEST_DELAY_SECONDS - (now - last_request_started_at)
            if delay > 0:
                await asyncio.sleep(delay)
            last_request_started_at = time.monotonic()
            return await original_request_json(*args, **kwargs)

    transport.request_json = throttled_request_json
    try:
        yield client
    finally:
        transport.request_json = original_request_json
        await cleanup_client()

