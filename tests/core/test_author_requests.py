from __future__ import annotations

import pytest

from semantic_scholar_skills.core import (
    AuthorBatchDetailsRequest,
    AuthorDetailsRequest,
    AuthorPapersRequest,
    AuthorSearchRequest,
    S2ValidationError,
)


def assert_validation_error(
    excinfo: pytest.ExceptionInfo[S2ValidationError],
    message: str,
    *,
    field: str | None = None,
    details: dict[str, object] | None = None,
) -> None:
    error = excinfo.value
    assert error.message == message
    assert error.field == field
    assert error.details == (details or {})


def test_author_search_serializes_params() -> None:
    request = AuthorSearchRequest(
        query="Andrew Ng",
        fields=["name", "paperCount"],
        offset=10,
        limit=25,
    )

    assert request.endpoint == "/author/search"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {
        "query": "Andrew Ng",
        "offset": 10,
        "limit": 25,
        "fields": "name,paperCount",
    }
    assert request.to_json() is None


def test_author_search_empty_query_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        AuthorSearchRequest(query="   ")

    assert_validation_error(excinfo, "Query string cannot be empty", field="query")


def test_author_search_limit_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        AuthorSearchRequest(query="Andrew Ng", limit=1001)

    assert_validation_error(
        excinfo,
        "Limit cannot exceed 1000",
        field="limit",
        details={"max_limit": 1000},
    )


def test_author_details_serializes_params() -> None:
    request = AuthorDetailsRequest(author_id="1741101", fields=["name", "hIndex"])

    assert request.endpoint == "/author/1741101"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"fields": "name,hIndex"}
    assert request.to_json() is None


def test_author_details_empty_author_id_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        AuthorDetailsRequest(author_id="   ")

    assert_validation_error(excinfo, "Author ID cannot be empty", field="author_id")


def test_author_papers_serializes_params() -> None:
    request = AuthorPapersRequest(
        author_id="1741101",
        fields=["title", "year"],
        offset=0,
        limit=10,
    )

    assert request.endpoint == "/author/1741101/papers"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"offset": 0, "limit": 10, "fields": "title,year"}
    assert request.to_json() is None


def test_author_papers_limit_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        AuthorPapersRequest(author_id="1741101", limit=1001)

    assert_validation_error(
        excinfo,
        "Limit cannot exceed 1000",
        field="limit",
        details={"max_limit": 1000},
    )


def test_author_batch_details_posts_ids_json() -> None:
    request = AuthorBatchDetailsRequest(
        author_ids=["1741101", "2061296"],
        fields="name,paperCount",
    )

    assert request.endpoint == "/author/batch"
    assert request.method == "POST"
    assert request.base_url is None
    assert request.to_params() == {"fields": "name,paperCount"}
    assert request.to_json() == {"ids": ["1741101", "2061296"]}


def test_author_batch_details_empty_ids_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        AuthorBatchDetailsRequest(author_ids=[], fields="name,paperCount")

    assert_validation_error(excinfo, "Author IDs list cannot be empty", field="author_ids")
