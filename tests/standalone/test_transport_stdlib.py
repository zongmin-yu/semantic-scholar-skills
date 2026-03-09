from __future__ import annotations

from io import BytesIO
import json
import socket
import urllib.error

import pytest

from semantic_scholar_skills.config import Config
from semantic_scholar_skills.core.exceptions import S2ApiError, S2NotFoundError, S2RateLimitError, S2TimeoutError
from semantic_scholar_skills.standalone.transport_stdlib import StdlibTransport


class FakeResponse:
    def __init__(self, status=200, headers=None, payload=None, text=None) -> None:
        self.status = status
        self.headers = headers or {}
        if text is None and payload is not None:
            text = json.dumps(payload)
        self._text = text or ""

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeOpenerQueue:
    def __init__(self) -> None:
        self.items: list[object] = []
        self.calls: list[dict[str, object]] = []

    def queue(self, *items: object) -> None:
        self.items.extend(items)

    def __call__(self, request, timeout=None):
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": {key.lower(): value for key, value in request.header_items()},
                "data": request.data,
                "timeout": timeout,
            }
        )
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def fake_opener_queue() -> FakeOpenerQueue:
    return FakeOpenerQueue()


@pytest.fixture
def recorded_sleep():
    calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    fake_sleep.calls = calls
    return fake_sleep


@pytest.fixture
def fixed_clock():
    current = {"value": 100.0}

    def clock() -> float:
        return current["value"]

    return clock


def make_http_error(status_code: int, *, url: str, text: str = "error", headers=None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url=url,
        code=status_code,
        msg=f"HTTP error: {status_code}",
        hdrs=headers or {},
        fp=BytesIO(text.encode("utf-8")),
    )


@pytest.mark.asyncio
async def test_stdlib_transport_builds_graph_api_get_request_and_parses_json(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(FakeResponse(payload={"ok": True}))
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    result = await transport.request_json("/paper/search", params={"query": "attention"})

    assert result == {"ok": True}
    assert fake_opener_queue.calls[0]["url"] == f"{Config.BASE_URL}/paper/search?query=attention"
    assert fake_opener_queue.calls[0]["method"] == "GET"
    assert fake_opener_queue.calls[0]["headers"]["user-agent"].startswith("semantic-scholar-skills/1.0")


@pytest.mark.asyncio
async def test_stdlib_transport_uses_recommendations_base_url_for_post_requests(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(FakeResponse(payload={"recommendedPapers": []}))
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    result = await transport.request_json(
        "/papers",
        method="POST",
        json={"positivePaperIds": ["p1"]},
        base_url=Config.RECOMMENDATIONS_BASE_URL,
    )

    assert result == {"recommendedPapers": []}
    assert fake_opener_queue.calls[0]["url"] == f"{Config.RECOMMENDATIONS_BASE_URL}/papers"
    assert fake_opener_queue.calls[0]["method"] == "POST"
    assert fake_opener_queue.calls[0]["headers"]["content-type"] == "application/json"
    assert fake_opener_queue.calls[0]["data"] == b'{"positivePaperIds": ["p1"]}'


@pytest.mark.asyncio
async def test_stdlib_transport_prefers_api_key_override_over_environment(
    monkeypatch,
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "env-token")
    fake_opener_queue.queue(FakeResponse(payload={"ok": True}))
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    await transport.request_json("/paper/search", api_key_override="override-token")

    assert fake_opener_queue.calls[0]["headers"]["x-api-key"] == "override-token"


@pytest.mark.asyncio
async def test_stdlib_transport_treats_placeholder_env_values_as_missing(
    monkeypatch,
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "null")
    fake_opener_queue.queue(FakeResponse(payload={"ok": True}))
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    await transport.request_json("/paper/search")

    assert "x-api-key" not in fake_opener_queue.calls[0]["headers"]


@pytest.mark.asyncio
async def test_stdlib_transport_retries_http_429_and_honors_retry_after_header(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(
        make_http_error(429, url=f"{Config.BASE_URL}/paper/search", headers={"Retry-After": "7"}),
        FakeResponse(payload={"ok": True}),
    )
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    result = await transport.request_json("/paper/search")

    assert result == {"ok": True}
    assert len(fake_opener_queue.calls) == 2
    assert recorded_sleep.calls == [7.0]


@pytest.mark.asyncio
async def test_stdlib_transport_retries_transient_503_then_succeeds(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(
        make_http_error(503, url=f"{Config.BASE_URL}/paper/search", text="unavailable"),
        FakeResponse(payload={"ok": True}),
    )
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    result = await transport.request_json("/paper/search")

    assert result == {"ok": True}
    assert len(fake_opener_queue.calls) == 2
    assert recorded_sleep.calls == [0.5]


@pytest.mark.asyncio
async def test_stdlib_transport_maps_404_to_not_found_error_without_retry(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(make_http_error(404, url=f"{Config.BASE_URL}/paper/missing", text="missing"))
    transport = StdlibTransport(opener=fake_opener_queue, sleeper=recorded_sleep, clock=fixed_clock)

    with pytest.raises(S2NotFoundError) as exc_info:
        await transport.request_json("/paper/missing")

    assert exc_info.value.status_code == 404
    assert len(fake_opener_queue.calls) == 1


@pytest.mark.asyncio
async def test_stdlib_transport_maps_socket_timeout_to_timeout_error(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(socket.timeout("timed out"))
    transport = StdlibTransport(
        opener=fake_opener_queue,
        sleeper=recorded_sleep,
        clock=fixed_clock,
        max_retries=0,
    )

    with pytest.raises(S2TimeoutError):
        await transport.request_json("/paper/search")


@pytest.mark.asyncio
async def test_stdlib_transport_preserves_request_metadata_on_api_error(
    fake_opener_queue,
    recorded_sleep,
    fixed_clock,
) -> None:
    fake_opener_queue.queue(make_http_error(500, url=f"{Config.RECOMMENDATIONS_BASE_URL}/papers", text="boom"))
    transport = StdlibTransport(
        opener=fake_opener_queue,
        sleeper=recorded_sleep,
        clock=fixed_clock,
        max_retries=0,
    )

    with pytest.raises(S2ApiError) as exc_info:
        await transport.request_json(
            "/papers",
            params={"limit": 5},
            method="POST",
            json={"positivePaperIds": ["p1"]},
            base_url=Config.RECOMMENDATIONS_BASE_URL,
        )

    error = exc_info.value
    assert error.status_code == 500
    assert error.endpoint == "/papers"
    assert error.method == "POST"
    assert error.params == {"limit": 5}
    assert error.json_body == {"positivePaperIds": ["p1"]}
    assert error.base_url == Config.RECOMMENDATIONS_BASE_URL
    assert error.response_text == "boom"
