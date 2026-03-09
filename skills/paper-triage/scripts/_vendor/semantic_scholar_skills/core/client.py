from __future__ import annotations

from typing import Any, Optional, Protocol

from .requests import (
    AuthorBatchDetailsRequest,
    AuthorDetailsRequest,
    AuthorPapersRequest,
    AuthorSearchRequest,
    PaperAutocompleteRequest,
    PaperAuthorsRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperCitationsRequest,
    PaperDetailsRequest,
    PaperRecommendationsMultiRequest,
    PaperRecommendationsSingleRequest,
    PaperReferencesRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    RequestModel,
    SnippetSearchRequest,
)


class SupportsRequestJson(Protocol):
    async def request_json(
        self,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        api_key_override: Optional[str] = None,
        method: str = "GET",
        json: Any = None,
        base_url: Optional[str] = None,
    ) -> Any: ...


class S2Client:
    def __init__(self, transport: SupportsRequestJson) -> None:
        self._transport = transport

    async def _request(
        self,
        request: RequestModel,
        *,
        api_key_override: Optional[str] = None,
    ) -> Any:
        return await self._transport.request_json(
            request.endpoint,
            params=request.to_params(),
            api_key_override=api_key_override,
            method=request.method,
            json=request.to_json(),
            base_url=request.base_url,
        )

    async def search_papers(
        self,
        request: PaperRelevanceSearchRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def bulk_search_papers(
        self,
        request: PaperBulkSearchRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def match_paper_title(
        self,
        request: PaperTitleSearchRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def get_paper(
        self,
        request: PaperDetailsRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def batch_papers(
        self,
        request: PaperBatchDetailsRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> list[dict[str, Any] | None]:
        return await self._request(request, api_key_override=api_key_override)

    async def get_paper_authors(
        self,
        request: PaperAuthorsRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def get_paper_citations(
        self,
        request: PaperCitationsRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def get_paper_references(
        self,
        request: PaperReferencesRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def autocomplete_papers(
        self,
        request: PaperAutocompleteRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def search_snippets(
        self,
        request: SnippetSearchRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def search_authors(
        self,
        request: AuthorSearchRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def get_author(
        self,
        request: AuthorDetailsRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def get_author_papers(
        self,
        request: AuthorPapersRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def batch_authors(
        self,
        request: AuthorBatchDetailsRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> list[dict[str, Any] | None]:
        return await self._request(request, api_key_override=api_key_override)

    async def recommend_for_paper(
        self,
        request: PaperRecommendationsSingleRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)

    async def recommend_from_papers(
        self,
        request: PaperRecommendationsMultiRequest,
        *,
        api_key_override: Optional[str] = None,
    ) -> dict[str, Any]:
        return await self._request(request, api_key_override=api_key_override)


def get_default_client() -> S2Client:
    from .transport import default_transport

    return S2Client(default_transport)


def make_compat_client(make_request_callable) -> S2Client:
    from .transport import MakeRequestCompatTransport

    return S2Client(MakeRequestCompatTransport(make_request_callable))
