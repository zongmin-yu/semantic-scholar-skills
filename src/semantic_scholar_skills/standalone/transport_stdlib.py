from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from collections import deque
from typing import Any, Awaitable, Callable
import urllib.error
import urllib.parse
import urllib.request

from ..config import Config, RateLimitConfig
from ..core.exceptions import S2ApiError, S2NotFoundError, S2RateLimitError, S2TimeoutError

logger = logging.getLogger("semantic_scholar_skills")
USER_AGENT = "semantic-scholar-skills/1.0 (+https://github.com/zongmin-yu/semantic-scholar-skills)"


def _normalize_key(key: str | None) -> str | None:
    if key is None:
        return None
    normalized = str(key).strip()
    if normalized.lower() in {"", "none", "null", "false"}:
        return None
    return normalized


def _get_api_key(api_key_override: str | None = None) -> str | None:
    normalized = _normalize_key(api_key_override)
    if normalized:
        return normalized
    env_value = _normalize_key(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    if env_value is None:
        raw_value = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if raw_value:
            logger.warning("SEMANTIC_SCHOLAR_API_KEY is set to a placeholder value; treating as not set.")
        else:
            logger.warning("No SEMANTIC_SCHOLAR_API_KEY set. Using unauthenticated access with lower rate limits.")
    return env_value


class _StdlibRateLimiter:
    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[Any]] | None = None,
    ) -> None:
        self._clock = clock or time.monotonic
        self._sleep = sleeper or asyncio.sleep
        self._events: dict[str, deque[float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _bucket_key(self, endpoint: str, base_url: str | None = None) -> str:
        if base_url and "recommendations" in base_url:
            return "/recommendations"
        if "recommendations" in endpoint:
            return "/recommendations"
        if "/author/search" in endpoint:
            return "/author/search"
        if "/paper/search" in endpoint:
            return "/paper/search"
        if "/paper/batch" in endpoint:
            return "/paper/batch"
        if "/author/batch" in endpoint:
            return "/author/batch"
        return "/default"

    def _get_rate_limit(self, endpoint: str, *, authenticated: bool) -> tuple[int, int]:
        if not authenticated:
            return RateLimitConfig.UNAUTHENTICATED_LIMIT
        if any(restricted in endpoint for restricted in RateLimitConfig.RESTRICTED_ENDPOINTS):
            if "batch" in endpoint:
                return RateLimitConfig.BATCH_LIMIT
            if "search" in endpoint:
                return RateLimitConfig.SEARCH_LIMIT
            if "recommendations" in endpoint:
                return RateLimitConfig.RECOMMENDATIONS_LIMIT
            return RateLimitConfig.SEARCH_LIMIT
        return RateLimitConfig.DEFAULT_LIMIT

    async def acquire(
        self,
        endpoint: str,
        *,
        authenticated: bool,
        base_url: str | None = None,
    ) -> None:
        bucket = self._bucket_key(endpoint, base_url)
        if bucket not in self._locks:
            self._locks[bucket] = asyncio.Lock()
            self._events[bucket] = deque()

        async with self._locks[bucket]:
            limit_endpoint = bucket if bucket != "/default" else endpoint
            requests, seconds = self._get_rate_limit(limit_endpoint, authenticated=authenticated)
            if requests <= 0 or seconds <= 0:
                return

            events = self._events[bucket]
            while True:
                now = self._clock()
                cutoff = now - float(seconds)
                while events and events[0] <= cutoff:
                    events.popleft()
                if len(events) < int(requests):
                    events.append(now)
                    return
                delay = (events[0] + float(seconds)) - now
                if delay > 0:
                    await self._sleep(delay)


class _RetryableHttpError(Exception):
    def __init__(self, *, status_code: int, response_text: str, retry_after: str | None) -> None:
        self.status_code = status_code
        self.response_text = response_text
        self.retry_after = retry_after
        super().__init__(f"HTTP error: {status_code}")


class StdlibTransport:
    def __init__(
        self,
        *,
        timeout: int | None = None,
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[Any]] | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self._timeout = Config.TIMEOUT if timeout is None else timeout
        self._opener = opener or urllib.request.urlopen
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or asyncio.sleep
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        self._rate_limiter = _StdlibRateLimiter(clock=self._clock, sleeper=self._sleeper)

    async def request_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        api_key_override: str | None = None,
        method: str = "GET",
        json: Any = None,
        base_url: str | None = None,
    ) -> Any:
        api_key = _get_api_key(api_key_override)
        authenticated = bool(api_key)
        method_upper = method.upper()

        for attempt in range(self._max_retries + 1):
            await self._rate_limiter.acquire(endpoint, authenticated=authenticated, base_url=base_url)
            try:
                return await asyncio.to_thread(
                    self._request_json_sync,
                    endpoint,
                    params=params,
                    api_key=api_key,
                    method=method_upper,
                    json_body=json,
                    base_url=base_url,
                )
            except _RetryableHttpError as exc:
                if attempt >= self._max_retries:
                    raise S2RateLimitError(
                        message="Rate limit exceeded. Consider using an API key for higher limits."
                        if exc.status_code == 429
                        else f"HTTP error: {exc.status_code}",
                        details={},
                        status_code=exc.status_code,
                        endpoint=endpoint,
                        method=method_upper,
                        params=params,
                        json_body=json,
                        base_url=base_url,
                        response_text=exc.response_text,
                        retry_after=exc.retry_after,
                        authenticated=authenticated,
                    ) if exc.status_code == 429 else S2ApiError(
                        message=f"HTTP error: {exc.status_code}",
                        details={},
                        status_code=exc.status_code,
                        endpoint=endpoint,
                        method=method_upper,
                        params=params,
                        json_body=json,
                        base_url=base_url,
                        response_text=exc.response_text,
                    )
                if exc.status_code == 429 and exc.retry_after:
                    await self._sleeper(float(exc.retry_after))
                else:
                    await self._sleeper(self._retry_backoff_seconds * (2**attempt))
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                is_timeout = isinstance(exc, (socket.timeout, TimeoutError))
                if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, socket.timeout):
                    is_timeout = True
                if attempt < self._max_retries:
                    await self._sleeper(self._retry_backoff_seconds * (2**attempt))
                    continue
                if is_timeout:
                    raise S2TimeoutError(
                        message=f"Request timed out after {self._timeout} seconds",
                        details={},
                        endpoint=endpoint,
                        method=method_upper,
                        timeout_seconds=self._timeout,
                    ) from exc
                raise S2ApiError(
                    message=str(exc),
                    details={},
                    endpoint=endpoint,
                    method=method_upper,
                    params=params,
                    json_body=json,
                    base_url=base_url,
                ) from exc
        raise AssertionError("unreachable")

    def _build_request(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None,
        api_key: str | None,
        method: str,
        json_body: Any,
        base_url: str | None,
    ) -> tuple[str, urllib.request.Request]:
        if endpoint.startswith(("http://", "https://")):
            url = endpoint
        else:
            url = f"{base_url or Config.BASE_URL}{endpoint}"

        encoded_params = urllib.parse.urlencode(params or {}, doseq=False)
        if encoded_params:
            joiner = "&" if "?" in url else "?"
            url = f"{url}{joiner}{encoded_params}"

        body: bytes | None = None
        headers = {"User-Agent": USER_AGENT}
        if api_key:
            headers["x-api-key"] = api_key

        if method != "GET" and json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        return url, urllib.request.Request(url=url, data=body, headers=headers, method=method)

    def _request_json_sync(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None,
        api_key: str | None,
        method: str,
        json_body: Any,
        base_url: str | None,
    ) -> Any:
        url, request = self._build_request(
            endpoint,
            params=params,
            api_key=api_key,
            method=method,
            json_body=json_body,
            base_url=base_url,
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                payload_bytes = response.read()
                text = payload_bytes.decode("utf-8") if payload_bytes else ""
                try:
                    return json.loads(text) if text else None
                except json.JSONDecodeError as exc:
                    raise S2ApiError(
                        message="Invalid JSON response",
                        details={},
                        status_code=getattr(response, "status", None),
                        endpoint=endpoint,
                        method=method,
                        params=params,
                        json_body=json_body,
                        base_url=base_url,
                        response_text=text,
                    ) from exc
        except urllib.error.HTTPError as exc:
            response_bytes = exc.read()
            response_text = response_bytes.decode("utf-8", errors="replace") if response_bytes else ""
            status_code = exc.code
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            if status_code in {429, 500, 502, 503, 504}:
                raise _RetryableHttpError(
                    status_code=status_code,
                    response_text=response_text,
                    retry_after=retry_after,
                ) from exc
            if status_code == 404:
                raise S2NotFoundError(
                    message="HTTP error: 404",
                    details={},
                    status_code=404,
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    json_body=json_body,
                    base_url=base_url,
                    response_text=response_text,
                ) from exc
            raise S2ApiError(
                message=f"HTTP error: {status_code}",
                details={},
                status_code=status_code,
                endpoint=endpoint,
                method=method,
                params=params,
                json_body=json_body,
                base_url=base_url,
                response_text=response_text,
            ) from exc
