from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_scholar_skills.config import Config, ErrorType
from semantic_scholar_skills.core import transport as transport_module


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: calls the real Semantic Scholar API")


@dataclass
class MockRequestController:
    calls: list[dict[str, Any]] = field(default_factory=list)
    queue: deque[Any] = field(default_factory=deque)
    default_response: Any = None

    def queue_responses(self, *items: Any) -> None:
        self.queue.extend(items)

    async def fake_request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        api_key_override: str | None = None,
        method: str = "GET",
        json: Any = None,
        base_url: str | None = None,
    ) -> Any:
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
def mock_request_controller() -> MockRequestController:
    return MockRequestController()


@pytest.fixture
def mock_error_payload():
    def factory(
        *,
        error_type: ErrorType | str = ErrorType.API_ERROR,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        status_code: int | None = None,
        response: str = "not found",
        retry_after: str = "60",
        authenticated: bool = True,
    ) -> dict[str, Any]:
        resolved_type = error_type.value if isinstance(error_type, ErrorType) else error_type

        if resolved_type == ErrorType.VALIDATION.value:
            payload_message = message or "Validation failed"
            payload_details = details or {}
        elif resolved_type == ErrorType.RATE_LIMIT.value:
            payload_message = message or "Rate limit exceeded. Consider using an API key for higher limits."
            payload_details = {
                "status_code": 429,
                "retry_after": retry_after,
                "authenticated": authenticated,
                "response": response,
            }
            if details:
                payload_details.update(details)
        elif resolved_type == ErrorType.TIMEOUT.value:
            payload_message = message or f"Request timed out after {Config.TIMEOUT} seconds"
            payload_details = details or {}
        else:
            payload_status = 500 if status_code is None else status_code
            payload_message = message or f"HTTP error: {payload_status}"
            payload_details = {
                "status_code": payload_status,
                "response": response,
            }
            if details:
                payload_details.update(details)

        return {
            "error": {
                "type": resolved_type,
                "message": payload_message,
                "details": payload_details,
            }
        }

    return factory


@pytest.fixture(autouse=True)
def reset_transport_state() -> None:
    transport_module.http_client = None
    transport_module.rate_limiter._events.clear()
    transport_module.rate_limiter._locks.clear()
    yield
    transport_module.http_client = None
    transport_module.rate_limiter._events.clear()
    transport_module.rate_limiter._locks.clear()
