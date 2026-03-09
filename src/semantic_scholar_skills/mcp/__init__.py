from typing import Any, Dict, Optional

from fastmcp import FastMCP

from ..config import ErrorType
from ..core.exceptions import (
    S2ApiError,
    S2Error,
    S2RateLimitError,
    S2TimeoutError,
    S2ValidationError,
)
from ..core.transport import (
    RateLimiter,
    cleanup_client,
    default_transport,
    get_api_key,
    initialize_client,
    rate_limiter,
)

mcp = FastMCP("Semantic Scholar Server")


def create_error_response(
    error_type: ErrorType,
    message: str,
    details: Optional[Dict] = None,
) -> Dict:
    return {
        "error": {
            "type": error_type.value,
            "message": message,
            "details": details or {},
        }
    }


def s2_exception_to_error_response(exc: S2Error) -> dict[str, Any]:
    if isinstance(exc, S2ValidationError):
        return create_error_response(ErrorType.VALIDATION, exc.message, exc.details)
    if isinstance(exc, S2RateLimitError):
        return create_error_response(
            ErrorType.RATE_LIMIT,
            exc.message,
            {
                "status_code": 429,
                "retry_after": exc.retry_after,
                "authenticated": exc.authenticated,
            },
        )
    if isinstance(exc, S2TimeoutError):
        return create_error_response(ErrorType.TIMEOUT, exc.message, {})
    if isinstance(exc, S2ApiError):
        details = {}
        if exc.status_code is not None:
            details["status_code"] = exc.status_code
        if exc.response_text is not None:
            details["response"] = exc.response_text
        return create_error_response(ErrorType.API_ERROR, exc.message, details)
    return create_error_response(ErrorType.API_ERROR, str(exc), {})


async def make_request(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    api_key_override: Optional[str] = None,
    method: str = "GET",
    json: Any = None,
    base_url: Optional[str] = None,
) -> Any:
    try:
        return await default_transport.request_json(
            endpoint,
            params=params,
            api_key_override=api_key_override,
            method=method,
            json=json,
            base_url=base_url,
        )
    except S2Error as exc:
        return s2_exception_to_error_response(exc)
