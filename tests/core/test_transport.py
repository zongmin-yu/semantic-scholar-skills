from __future__ import annotations

from typing import Any

import httpx
import pytest

from semantic_scholar_skills.config import Config, ErrorType
from semantic_scholar_skills.core import transport as transport_module
from semantic_scholar_skills.core.exceptions import (
    S2ApiError,
    S2NotFoundError,
    S2RateLimitError,
    S2TimeoutError,
    S2ValidationError,
)


USER_AGENT = "semantic-scholar-skills/1.0 (+https://github.com/zongmin-yu/semantic-scholar-skills)"


class JsonResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class RecordingAsyncClient:
    def __init__(self, *, payload: Any = None, exc: Exception | None = None) -> None:
        self.payload = payload
        self.exc = exc
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json: Any = None,
    ) -> Any:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "headers": headers or {},
                "json": json,
            }
        )
        if self.exc is not None:
            raise self.exc
        return JsonResponse(self.payload)

    async def aclose(self) -> None:
        self.closed = True


def make_http_status_error(
    status_code: int,
    *,
    url: str,
    method: str = "GET",
    text: str = "error",
    headers: dict[str, str] | None = None,
) -> httpx.HTTPStatusError:
    request = httpx.Request(method, url)
    response = httpx.Response(
        status_code=status_code,
        request=request,
        text=text,
        headers=headers,
    )
    return httpx.HTTPStatusError(f"HTTP error: {status_code}", request=request, response=response)


@pytest.mark.parametrize(
    ("payload", "exc_type"),
    [
        (
            {
                "error": {
                    "type": "validation",
                    "message": "Invalid request",
                    "details": {"field": "query"},
                }
            },
            S2ValidationError,
        ),
        (
            {
                "error": {
                    "type": "rate_limit",
                    "message": "Too many requests",
                    "details": {
                        "status_code": 429,
                        "retry_after": "90",
                        "authenticated": True,
                        "response": "slow down",
                    },
                }
            },
            S2RateLimitError,
        ),
        (
            {
                "error": {
                    "type": "timeout",
                    "message": f"Request timed out after {Config.TIMEOUT} seconds",
                    "details": {},
                }
            },
            S2TimeoutError,
        ),
        (
            {
                "error": {
                    "type": "api_error",
                    "message": "HTTP error: 404",
                    "details": {"status_code": 404, "response": "missing"},
                }
            },
            S2NotFoundError,
        ),
        (
            {
                "error": {
                    "type": "api_error",
                    "message": "HTTP error: 500",
                    "details": {"status_code": 500, "response": "boom"},
                }
            },
            S2ApiError,
        ),
    ],
)
def test_error_dict_to_exception_maps_payload_types(payload: dict[str, Any], exc_type: type[Exception]) -> None:
    error = transport_module.error_dict_to_exception(
        payload,
        endpoint="/paper/search",
        method="GET",
        params={"query": "attention"},
        json_body=None,
        base_url=None,
    )

    assert isinstance(error, exc_type)


def test_error_dict_to_exception_preserves_rate_limit_metadata() -> None:
    error = transport_module.error_dict_to_exception(
        {
            "error": {
                "type": ErrorType.RATE_LIMIT.value,
                "message": "Too many requests",
                "details": {
                    "status_code": 429,
                    "retry_after": "90",
                    "authenticated": True,
                    "response": "slow down",
                },
            }
        },
        endpoint="/paper/search",
        method="GET",
        params={"query": "attention"},
    )

    assert isinstance(error, S2RateLimitError)
    assert error.status_code == 429
    assert error.retry_after == "90"
    assert error.authenticated is True
    assert error.endpoint == "/paper/search"
    assert error.method == "GET"
    assert error.params == {"query": "attention"}
    assert error.response_text == "slow down"


def test_error_dict_to_exception_preserves_not_found_metadata() -> None:
    error = transport_module.error_dict_to_exception(
        {
            "error": {
                "type": ErrorType.API_ERROR.value,
                "message": "HTTP error: 404",
                "details": {"status_code": 404, "response": "missing"},
            }
        },
        endpoint="/paper/abc",
        method="GET",
        params={"fields": "title"},
    )

    assert isinstance(error, S2NotFoundError)
    assert error.status_code == 404
    assert error.endpoint == "/paper/abc"
    assert error.method == "GET"
    assert error.params == {"fields": "title"}
    assert error.response_text == "missing"


def test_get_api_key_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    assert transport_module.get_api_key() is None


@pytest.mark.parametrize("placeholder", ["", "none", "null", "false"])
def test_get_api_key_treats_placeholder_values_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    placeholder: str,
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", placeholder)

    assert transport_module.get_api_key() is None


def test_get_api_key_returns_real_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "real-token")

    assert transport_module.get_api_key() == "real-token"


@pytest.mark.asyncio
async def test_request_json_uses_default_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = RecordingAsyncClient(payload={"ok": True})

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    result = await transport_module.S2Transport().request_json(
        "/paper/search",
        params={"query": "attention"},
    )

    assert result == {"ok": True}
    assert fake_client.calls == [
        {
            "method": "GET",
            "url": f"{Config.BASE_URL}/paper/search",
            "params": {"query": "attention"},
            "headers": {"User-Agent": USER_AGENT},
            "json": None,
        }
    ]


@pytest.mark.asyncio
async def test_request_json_respects_base_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = RecordingAsyncClient(payload={"ok": True})

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    await transport_module.S2Transport().request_json(
        "/papers",
        method="POST",
        json={"positivePaperIds": ["p1"]},
        base_url=Config.RECOMMENDATIONS_BASE_URL,
    )

    assert fake_client.calls[0]["url"] == f"{Config.RECOMMENDATIONS_BASE_URL}/papers"
    assert fake_client.calls[0]["method"] == "POST"
    assert fake_client.calls[0]["json"] == {"positivePaperIds": ["p1"]}


@pytest.mark.asyncio
async def test_request_json_uses_absolute_endpoint_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = RecordingAsyncClient(payload={"ok": True})

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    await transport_module.S2Transport().request_json("https://example.com/custom")

    assert fake_client.calls[0]["url"] == "https://example.com/custom"


@pytest.mark.asyncio
async def test_request_json_sends_api_key_header_when_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = RecordingAsyncClient(payload={"ok": True})

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: "token-123")

    await transport_module.S2Transport().request_json("/paper/abc")

    assert fake_client.calls[0]["headers"] == {
        "x-api-key": "token-123",
        "User-Agent": USER_AGENT,
    }


@pytest.mark.asyncio
async def test_request_json_omits_api_key_header_when_unauthenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = RecordingAsyncClient(payload={"ok": True})

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    await transport_module.S2Transport().request_json("/paper/abc")

    assert "x-api-key" not in fake_client.calls[0]["headers"]
    assert fake_client.calls[0]["headers"]["User-Agent"] == USER_AGENT


@pytest.mark.asyncio
async def test_request_json_maps_429_to_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"{Config.BASE_URL}/paper/search"
    fake_client = RecordingAsyncClient(
        exc=make_http_status_error(
            429,
            url=url,
            text="slow down",
            headers={"retry-after": "45"},
        )
    )

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: "token-123")

    with pytest.raises(S2RateLimitError) as excinfo:
        await transport_module.S2Transport().request_json("/paper/search", params={"query": "attention"})

    error = excinfo.value
    assert error.status_code == 429
    assert error.retry_after == "45"
    assert error.authenticated is True
    assert error.endpoint == "/paper/search"
    assert error.method == "GET"
    assert error.params == {"query": "attention"}
    assert error.response_text == "slow down"


@pytest.mark.asyncio
async def test_request_json_maps_404_to_not_found_error(monkeypatch: pytest.MonkeyPatch) -> None:
    url = f"{Config.BASE_URL}/paper/abc"
    fake_client = RecordingAsyncClient(
        exc=make_http_status_error(404, url=url, text="missing")
    )

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    with pytest.raises(S2NotFoundError) as excinfo:
        await transport_module.S2Transport().request_json("/paper/abc", params={"fields": "title"})

    error = excinfo.value
    assert error.status_code == 404
    assert error.endpoint == "/paper/abc"
    assert error.method == "GET"
    assert error.params == {"fields": "title"}
    assert error.response_text == "missing"


@pytest.mark.asyncio
async def test_request_json_maps_timeout_to_timeout_error(monkeypatch: pytest.MonkeyPatch) -> None:
    request = httpx.Request("GET", f"{Config.BASE_URL}/paper/search")
    fake_client = RecordingAsyncClient(
        exc=httpx.ReadTimeout("timed out", request=request)
    )

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    with pytest.raises(S2TimeoutError) as excinfo:
        await transport_module.S2Transport().request_json("/paper/search", params={"query": "attention"})

    error = excinfo.value
    assert error.endpoint == "/paper/search"
    assert error.method == "GET"
    assert error.timeout_seconds == Config.TIMEOUT


@pytest.mark.asyncio
async def test_request_json_maps_unexpected_exception_to_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = RecordingAsyncClient(exc=RuntimeError("boom"))

    async def fake_initialize_client() -> RecordingAsyncClient:
        return fake_client

    monkeypatch.setattr(transport_module, "initialize_client", fake_initialize_client)
    monkeypatch.setattr(transport_module, "get_api_key", lambda: None)

    with pytest.raises(S2ApiError) as excinfo:
        await transport_module.S2Transport().request_json(
            "/paper/search",
            params={"query": "attention"},
            method="POST",
            json={"ids": ["p1"]},
            base_url=Config.RECOMMENDATIONS_BASE_URL,
        )

    error = excinfo.value
    assert error.message == "boom"
    assert error.endpoint == "/paper/search"
    assert error.method == "POST"
    assert error.params == {"query": "attention"}
    assert error.json_body == {"ids": ["p1"]}
    assert error.base_url == Config.RECOMMENDATIONS_BASE_URL


@pytest.mark.asyncio
async def test_cleanup_client_resets_global_http_client() -> None:
    client = await transport_module.initialize_client()

    assert transport_module.http_client is client

    await transport_module.cleanup_client()

    assert transport_module.http_client is None
