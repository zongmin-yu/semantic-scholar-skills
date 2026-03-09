from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import semantic_scholar_skills.mcp.bridge as bridge
from semantic_scholar_skills.config import Config, ErrorType
from semantic_scholar_skills.mcp import create_error_response


@dataclass
class MockMakeRequestController:
    monkeypatch: pytest.MonkeyPatch
    calls: list[dict[str, Any]] = field(default_factory=list)
    queue: deque[Any] = field(default_factory=deque)
    default_response: Any = None

    def install(self, module):
        self.monkeypatch.setattr(module, "make_request", self.fake_make_request)
        return self

    def queue_responses(self, *items):
        self.queue.extend(items)

    async def fake_make_request(
        self,
        endpoint,
        params=None,
        api_key_override=None,
        method="GET",
        json=None,
        base_url=None,
    ):
        self.calls.append(
            {
                "endpoint": endpoint,
                "params": params,
                "api_key_override": api_key_override,
                "method": method,
                "json": json,
                "base_url": base_url,
            }
        )

        if self.queue:
            next_item = self.queue.popleft()
            if isinstance(next_item, Exception):
                raise next_item
            return next_item

        return self.default_response


@pytest.fixture
def mock_make_request(monkeypatch):
    return MockMakeRequestController(monkeypatch=monkeypatch)


@pytest.fixture
def mock_error_response():
    def factory(*, status_code=None, response="not found", retry_after="60", authenticated=True, timeout=False):
        if timeout:
            return create_error_response(
                ErrorType.TIMEOUT,
                f"Request timed out after {Config.TIMEOUT} seconds",
            )

        if status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded. Consider using an API key for higher limits.",
                {
                    "status_code": 429,
                    "retry_after": retry_after,
                    "authenticated": authenticated,
                },
            )

        if status_code is not None:
            return create_error_response(
                ErrorType.API_ERROR,
                f"HTTP error: {status_code}",
                {
                    "status_code": status_code,
                    "response": response,
                },
            )

        return create_error_response(ErrorType.API_ERROR, "HTTP error: 500", {"status_code": 500, "response": response})

    return factory


@pytest_asyncio.fixture
async def bridge_client():
    async with AsyncClient(transport=ASGITransport(app=bridge.app), base_url="http://test") as client:
        yield client


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer bridge-token"}
