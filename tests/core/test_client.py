from __future__ import annotations

from typing import Any, Callable

import pytest

from semantic_scholar_skills.config import ErrorType
from semantic_scholar_skills.core import (
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
    S2ApiError,
    S2NotFoundError,
    S2RateLimitError,
    S2TimeoutError,
    S2ValidationError,
    SnippetSearchRequest,
    make_compat_client,
)


def assert_single_call(
    calls: list[dict[str, Any]],
    *,
    endpoint: str,
    params: dict[str, Any] | None,
    api_key_override: str | None = None,
    method: str = "GET",
    json: Any = None,
    base_url: str | None = None,
) -> None:
    assert calls == [
        {
            "endpoint": endpoint,
            "params": params,
            "api_key_override": api_key_override,
            "method": method,
            "json": json,
            "base_url": base_url,
        }
    ]


CLIENT_CASES = [
    pytest.param(
        "search_papers",
        lambda: PaperRelevanceSearchRequest(query="attention", limit=5),
        {"total": 1, "offset": 0, "data": [{"paperId": "p1"}]},
        id="search_papers",
    ),
    pytest.param(
        "bulk_search_papers",
        lambda: PaperBulkSearchRequest(
            query=" transformer ",
            token="next-1",
            fields=["paperId", "title"],
            sort="citationCount:desc",
            publication_types=["Conference"],
            open_access_pdf=True,
            min_citation_count=10,
            publication_date_or_year="2020-01-01:2024-12-31",
            venue=["NeurIPS"],
            fields_of_study=["Computer Science"],
        ),
        {"total": 1, "token": "next-2", "data": [{"paperId": "p1"}]},
        id="bulk_search_papers",
    ),
    pytest.param(
        "match_paper_title",
        lambda: PaperTitleSearchRequest(query="Attention is All You Need"),
        {"paperId": "p1", "title": "Attention is All You Need"},
        id="match_paper_title",
    ),
    pytest.param(
        "get_paper",
        lambda: PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"]),
        {"paperId": "CorpusId:215416146", "title": "Attention"},
        id="get_paper",
    ),
    pytest.param(
        "batch_papers",
        lambda: PaperBatchDetailsRequest(
            paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"],
            fields="title,year",
        ),
        [{"paperId": "649def34f8be52c8b66281af98ae884c09aef38b"}],
        id="batch_papers",
    ),
    pytest.param(
        "get_paper_authors",
        lambda: PaperAuthorsRequest(
            paper_id="paper-123",
            fields=["name", "affiliations"],
            offset=5,
            limit=25,
        ),
        {"offset": 5, "next": 30, "data": [{"authorId": "a1"}]},
        id="get_paper_authors",
    ),
    pytest.param(
        "get_paper_citations",
        lambda: PaperCitationsRequest(
            paper_id="paper-123",
            fields=["title", "contexts"],
            offset=0,
            limit=10,
        ),
        {"offset": 0, "next": 10, "data": [{"paperId": "c1"}]},
        id="get_paper_citations",
    ),
    pytest.param(
        "get_paper_references",
        lambda: PaperReferencesRequest(
            paper_id="paper-123",
            fields=["title", "contexts"],
            offset=0,
            limit=10,
        ),
        {"offset": 0, "next": 10, "data": [{"paperId": "r1"}]},
        id="get_paper_references",
    ),
    pytest.param(
        "autocomplete_papers",
        lambda: PaperAutocompleteRequest(query="graph neural"),
        {"matches": [{"paperId": "p1"}]},
        id="autocomplete_papers",
    ),
    pytest.param(
        "search_snippets",
        lambda: SnippetSearchRequest(
            query="transformer attention",
            fields=["snippet.text", "paper.title"],
            limit=5,
            paper_ids=["p1", "p2"],
            authors=["Author One", "Author Two"],
            min_citation_count=50,
            inserted_before="2025-01-01",
            publication_date_or_year="2020-01-01:2024-12-31",
            year="2020-2024",
            venue=["NeurIPS", "ICML"],
            fields_of_study=["Computer Science", "Mathematics"],
        ),
        {"data": [{"snippet": {"text": "attention is all you need"}}]},
        id="search_snippets",
    ),
    pytest.param(
        "search_authors",
        lambda: AuthorSearchRequest(
            query="Andrew Ng",
            fields=["name", "paperCount"],
            offset=10,
            limit=25,
        ),
        {"total": 1, "offset": 10, "data": [{"authorId": "1741101"}]},
        id="search_authors",
    ),
    pytest.param(
        "get_author",
        lambda: AuthorDetailsRequest(author_id="1741101", fields=["name", "hIndex"]),
        {"authorId": "1741101", "name": "Andrew Ng"},
        id="get_author",
    ),
    pytest.param(
        "get_author_papers",
        lambda: AuthorPapersRequest(
            author_id="1741101",
            fields=["title", "year"],
            offset=0,
            limit=10,
        ),
        {"offset": 0, "next": 10, "data": [{"paperId": "p1"}]},
        id="get_author_papers",
    ),
    pytest.param(
        "batch_authors",
        lambda: AuthorBatchDetailsRequest(
            author_ids=["1741101", "2061296"],
            fields="name,paperCount",
        ),
        [{"authorId": "1741101"}, {"authorId": "2061296"}],
        id="batch_authors",
    ),
    pytest.param(
        "recommend_for_paper",
        lambda: PaperRecommendationsSingleRequest(
            paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            fields="title,year",
            limit=25,
            from_pool="all-cs",
        ),
        {"recommendedPapers": [{"paperId": "p1"}]},
        id="recommend_for_paper",
    ),
    pytest.param(
        "recommend_from_papers",
        lambda: PaperRecommendationsMultiRequest(
            positive_paper_ids=["p1", "p2"],
            negative_paper_ids=["p3"],
            fields="title,year",
            limit=25,
        ),
        {"recommendedPapers": [{"paperId": "p4"}]},
        id="recommend_from_papers",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("method_name", "request_factory", "payload"), CLIENT_CASES)
async def test_client_methods_delegate_to_transport(
    mock_request_controller,
    method_name: str,
    request_factory: Callable[[], Any],
    payload: Any,
) -> None:
    request = request_factory()
    client = make_compat_client(mock_request_controller.fake_request)
    mock_request_controller.queue_responses(payload)

    result = await getattr(client, method_name)(request, api_key_override="override-token")

    assert result == payload
    assert_single_call(
        mock_request_controller.calls,
        endpoint=request.endpoint,
        params=request.to_params(),
        api_key_override="override-token",
        method=request.method,
        json=request.to_json(),
        base_url=request.base_url,
    )


@pytest.mark.asyncio
async def test_client_compat_transport_raises_validation_error(mock_request_controller, mock_error_payload) -> None:
    request = PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"])
    client = make_compat_client(mock_request_controller.fake_request)
    mock_request_controller.queue_responses(
        mock_error_payload(
            error_type=ErrorType.VALIDATION,
            message="Bad request",
            details={"field": "paper_id"},
        )
    )

    with pytest.raises(S2ValidationError) as excinfo:
        await client.get_paper(request)

    assert excinfo.value.message == "Bad request"
    assert excinfo.value.details == {"field": "paper_id"}


@pytest.mark.asyncio
async def test_client_compat_transport_raises_rate_limit_error(mock_request_controller, mock_error_payload) -> None:
    request = PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"])
    client = make_compat_client(mock_request_controller.fake_request)
    mock_request_controller.queue_responses(
        mock_error_payload(
            error_type=ErrorType.RATE_LIMIT,
            response="slow down",
            retry_after="45",
            authenticated=False,
        )
    )

    with pytest.raises(S2RateLimitError) as excinfo:
        await client.get_paper(request)

    error = excinfo.value
    assert error.status_code == 429
    assert error.retry_after == "45"
    assert error.authenticated is False
    assert error.endpoint == request.endpoint
    assert error.method == request.method
    assert error.params == request.to_params()
    assert error.json_body == request.to_json()
    assert error.base_url == request.base_url
    assert error.response_text == "slow down"


@pytest.mark.asyncio
async def test_client_compat_transport_raises_timeout_error(mock_request_controller, mock_error_payload) -> None:
    request = PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"])
    client = make_compat_client(mock_request_controller.fake_request)
    mock_request_controller.queue_responses(
        mock_error_payload(error_type=ErrorType.TIMEOUT)
    )

    with pytest.raises(S2TimeoutError) as excinfo:
        await client.get_paper(request)

    error = excinfo.value
    assert error.endpoint == request.endpoint
    assert error.method == request.method
    assert error.timeout_seconds == 30


@pytest.mark.asyncio
async def test_client_compat_transport_raises_not_found_error(mock_request_controller, mock_error_payload) -> None:
    request = PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"])
    client = make_compat_client(mock_request_controller.fake_request)
    mock_request_controller.queue_responses(
        mock_error_payload(status_code=404, response="missing")
    )

    with pytest.raises(S2NotFoundError) as excinfo:
        await client.get_paper(request)

    error = excinfo.value
    assert error.status_code == 404
    assert error.endpoint == request.endpoint
    assert error.method == request.method
    assert error.params == request.to_params()
    assert error.json_body == request.to_json()
    assert error.base_url == request.base_url
    assert error.response_text == "missing"


@pytest.mark.asyncio
async def test_client_compat_transport_raises_api_error(mock_request_controller, mock_error_payload) -> None:
    request = PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"])
    client = make_compat_client(mock_request_controller.fake_request)
    mock_request_controller.queue_responses(
        mock_error_payload(status_code=500, response="boom")
    )

    with pytest.raises(S2ApiError) as excinfo:
        await client.get_paper(request)

    error = excinfo.value
    assert error.status_code == 500
    assert error.endpoint == request.endpoint
    assert error.method == request.method
    assert error.params == request.to_params()
    assert error.json_body == request.to_json()
    assert error.base_url == request.base_url
    assert error.response_text == "boom"
