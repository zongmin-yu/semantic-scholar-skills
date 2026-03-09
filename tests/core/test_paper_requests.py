from __future__ import annotations

import pytest

from semantic_scholar_skills.core import (
    PaperAutocompleteRequest,
    PaperAuthorsRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperCitationsRequest,
    PaperDetailsRequest,
    PaperReferencesRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    S2ValidationError,
    SnippetSearchRequest,
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


def test_paper_relevance_search_defaults_and_limit_clamp() -> None:
    request = PaperRelevanceSearchRequest(query="attention", limit=250)

    assert request.endpoint == "/paper/search"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {
        "query": "attention",
        "offset": 0,
        "limit": 100,
        "fields": "title,abstract,year,citationCount,authors,url",
    }
    assert request.to_json() is None


def test_paper_relevance_search_serializes_optional_filters() -> None:
    request = PaperRelevanceSearchRequest(
        query="attention",
        fields=["paperId", "title"],
        publication_types=["Conference"],
        open_access_pdf=True,
        min_citation_count=10,
        year="2024",
        venue=["NeurIPS"],
        fields_of_study=["Computer Science"],
        offset=5,
        limit=25,
    )

    assert request.to_params() == {
        "query": "attention",
        "offset": 5,
        "limit": 25,
        "fields": "paperId,title",
        "publicationTypes": "Conference",
        "openAccessPdf": "true",
        "minCitationCount": 10,
        "year": "2024",
        "venue": "NeurIPS",
        "fieldsOfStudy": "Computer Science",
    }


def test_paper_relevance_search_empty_query_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperRelevanceSearchRequest(query="   ")

    assert_validation_error(excinfo, "Query string cannot be empty", field="query")


def test_paper_bulk_search_serializes_all_filters() -> None:
    request = PaperBulkSearchRequest(
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
    )

    assert request.endpoint == "/paper/search/bulk"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {
        "query": "transformer",
        "token": "next-1",
        "fields": "paperId,title",
        "sort": "citationCount:desc",
        "publicationTypes": "Conference",
        "openAccessPdf": "true",
        "minCitationCount": "10",
        "publicationDateOrYear": "2020-01-01:2024-12-31",
        "venue": "NeurIPS",
        "fieldsOfStudy": "Computer Science",
    }
    assert request.to_json() is None


def test_paper_bulk_search_invalid_sort_field_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperBulkSearchRequest(sort="year:desc")

    assert_validation_error(
        excinfo,
        "Invalid sort field. Must be one of: paperId, publicationDate, citationCount",
        field="sort",
    )


def test_paper_bulk_search_invalid_sort_order_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperBulkSearchRequest(sort="paperId:sideways")

    assert_validation_error(
        excinfo,
        "Invalid sort order. Must be one of: asc, desc",
        field="sort",
    )


def test_paper_title_search_defaults_use_default_fields() -> None:
    request = PaperTitleSearchRequest(query="Attention is All You Need")

    assert request.endpoint == "/paper/search/match"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {
        "query": "Attention is All You Need",
        "fields": "title,abstract,year,citationCount,authors,url",
    }
    assert request.to_json() is None


def test_paper_title_search_empty_query_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperTitleSearchRequest(query="   ")

    assert_validation_error(excinfo, "Query string cannot be empty", field="query")


def test_paper_details_serializes_fields() -> None:
    request = PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year"])

    assert request.endpoint == "/paper/CorpusId:215416146"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"fields": "title,year"}
    assert request.to_json() is None


def test_paper_details_empty_id_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperDetailsRequest(paper_id="   ")

    assert_validation_error(excinfo, "Paper ID cannot be empty", field="paper_id")


def test_paper_batch_details_posts_ids_json() -> None:
    request = PaperBatchDetailsRequest(
        paper_ids=["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"],
        fields="title,year",
    )

    assert request.endpoint == "/paper/batch"
    assert request.method == "POST"
    assert request.base_url is None
    assert request.to_params() == {"fields": "title,year"}
    assert request.to_json() == {"ids": ["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}


def test_paper_batch_details_empty_ids_raise() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperBatchDetailsRequest(paper_ids=[], fields="title,year")

    assert_validation_error(excinfo, "Paper IDs list cannot be empty", field="paper_ids")


def test_paper_authors_serializes_params() -> None:
    request = PaperAuthorsRequest(
        paper_id="paper-123",
        fields=["name", "affiliations"],
        offset=5,
        limit=25,
    )

    assert request.endpoint == "/paper/paper-123/authors"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"offset": 5, "limit": 25, "fields": "name,affiliations"}
    assert request.to_json() is None


def test_paper_authors_limit_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperAuthorsRequest(paper_id="paper-123", limit=1001)

    assert_validation_error(
        excinfo,
        "Limit cannot exceed 1000",
        field="limit",
        details={"max_limit": 1000},
    )


def test_paper_citations_serializes_params() -> None:
    request = PaperCitationsRequest(
        paper_id="paper-123",
        fields=["title", "contexts"],
        offset=0,
        limit=10,
    )

    assert request.endpoint == "/paper/paper-123/citations"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"offset": 0, "limit": 10, "fields": "title,contexts"}
    assert request.to_json() is None


def test_paper_citations_invalid_field_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperCitationsRequest(paper_id="paper-123", fields=["badField"])

    error = excinfo.value
    assert error.message == "Invalid fields: badField"
    assert error.field == "fields"
    assert "valid_fields" in error.details


def test_paper_references_serializes_params() -> None:
    request = PaperReferencesRequest(
        paper_id="paper-123",
        fields=["title", "contexts"],
        offset=0,
        limit=10,
    )

    assert request.endpoint == "/paper/paper-123/references"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"offset": 0, "limit": 10, "fields": "title,contexts"}
    assert request.to_json() is None


def test_paper_references_limit_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperReferencesRequest(paper_id="paper-123", limit=1001)

    assert_validation_error(
        excinfo,
        "Limit cannot exceed 1000",
        field="limit",
        details={"max_limit": 1000},
    )


def test_paper_references_invalid_field_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperReferencesRequest(paper_id="paper-123", fields=["badField"])

    error = excinfo.value
    assert error.message == "Invalid fields: badField"
    assert error.field == "fields"
    assert "valid_fields" in error.details


def test_paper_autocomplete_truncates_queries_to_one_hundred_characters() -> None:
    request = PaperAutocompleteRequest(query="a" * 150)

    assert request.endpoint == "/paper/autocomplete"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {"query": "a" * 100}
    assert request.to_json() is None


def test_paper_autocomplete_empty_query_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperAutocompleteRequest(query="   ")

    assert_validation_error(excinfo, "Query string cannot be empty", field="query")


def test_snippet_search_serializes_all_filters() -> None:
    request = SnippetSearchRequest(
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
    )

    assert request.endpoint == "/snippet/search"
    assert request.method == "GET"
    assert request.base_url is None
    assert request.to_params() == {
        "query": "transformer attention",
        "limit": 5,
        "fields": "snippet.text,paper.title",
        "paperIds": "p1,p2",
        "authors": "Author One,Author Two",
        "minCitationCount": 50,
        "insertedBefore": "2025-01-01",
        "publicationDateOrYear": "2020-01-01:2024-12-31",
        "year": "2020-2024",
        "venue": "NeurIPS,ICML",
        "fieldsOfStudy": "Computer Science,Mathematics",
    }
    assert request.to_json() is None


def test_snippet_search_too_many_authors_raises() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        SnippetSearchRequest(
            query="transformer attention",
            authors=[f"Author {index}" for index in range(11)],
        )

    assert_validation_error(
        excinfo,
        "Cannot filter by more than 10 authors",
        field="authors",
        details={"max_authors": 10},
    )


@pytest.mark.parametrize(
    ("limit", "message", "details"),
    [
        (0, "Limit must be at least 1", {"min_limit": 1}),
        (1001, "Limit cannot exceed 1000", {"max_limit": 1000}),
    ],
)
def test_snippet_search_limit_bounds_raise(
    limit: int,
    message: str,
    details: dict[str, object],
) -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        SnippetSearchRequest(query="transformer attention", limit=limit)

    assert_validation_error(excinfo, message, field="limit", details=details)
