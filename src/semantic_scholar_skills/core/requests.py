from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

from ..config import (
    AuthorDetailFields,
    CitationReferenceFields,
    Config,
    PaperFields,
    VALID_FIELDS_OF_STUDY,
    VALID_PUBLICATION_TYPES,
    VALID_RECOMMENDATION_POOLS,
)
from .exceptions import S2ValidationError

VALID_BULK_SORT_FIELDS = ["paperId", "publicationDate", "citationCount"]
VALID_BULK_SORT_ORDERS = ["asc", "desc"]


class RequestModel:
    method: str = "GET"
    base_url: Optional[str] = None

    @property
    def endpoint(self) -> str:
        raise NotImplementedError

    def to_params(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> Any:
        return None


def _raise_validation(message: str, details: Optional[dict[str, Any]] = None, field: Optional[str] = None) -> None:
    raise S2ValidationError(message=message, details=details or {}, field=field)


def _validate_fields(fields: list[str], valid_fields: set[str]) -> None:
    invalid_fields = set(fields) - valid_fields
    if invalid_fields:
        _raise_validation(
            f"Invalid fields: {', '.join(invalid_fields)}",
            {"valid_fields": list(valid_fields)},
            field="fields",
        )


def _validate_csv_fields(fields: str, valid_fields: set[str]) -> None:
    field_list = fields.split(",")
    invalid_fields = set(field_list) - valid_fields
    if invalid_fields:
        _raise_validation(
            f"Invalid fields: {', '.join(invalid_fields)}",
            {"valid_fields": list(valid_fields)},
            field="fields",
        )


def _quote_path_identifier(identifier: str) -> str:
    return quote(identifier, safe=":")


@dataclass(slots=True)
class PaperRelevanceSearchRequest(RequestModel):
    query: str
    fields: Optional[list[str]] = None
    publication_types: Optional[list[str]] = None
    open_access_pdf: bool = False
    min_citation_count: Optional[int] = None
    year: Optional[str] = None
    venue: Optional[list[str]] = None
    fields_of_study: Optional[list[str]] = None
    offset: int = 0
    limit: int = 10

    @property
    def endpoint(self) -> str:
        return "/paper/search"

    def __post_init__(self) -> None:
        if not self.query.strip():
            _raise_validation("Query string cannot be empty", field="query")
        if self.fields is None:
            self.fields = list(PaperFields.DEFAULT)
        else:
            _validate_fields(self.fields, PaperFields.VALID_FIELDS)
        self.limit = min(self.limit, 100)

    def to_params(self) -> dict[str, Any]:
        params = {
            "query": self.query,
            "offset": self.offset,
            "limit": self.limit,
            "fields": ",".join(self.fields or []),
        }
        if self.publication_types:
            params["publicationTypes"] = ",".join(self.publication_types)
        if self.open_access_pdf:
            params["openAccessPdf"] = "true"
        if self.min_citation_count is not None:
            params["minCitationCount"] = self.min_citation_count
        if self.year:
            params["year"] = self.year
        if self.venue:
            params["venue"] = ",".join(self.venue)
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)
        return params


@dataclass(slots=True)
class PaperBulkSearchRequest(RequestModel):
    query: Optional[str] = None
    token: Optional[str] = None
    fields: Optional[list[str]] = None
    sort: Optional[str] = None
    publication_types: Optional[list[str]] = None
    open_access_pdf: bool = False
    min_citation_count: Optional[int] = None
    publication_date_or_year: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[list[str]] = None
    fields_of_study: Optional[list[str]] = None

    @property
    def endpoint(self) -> str:
        return "/paper/search/bulk"

    def __post_init__(self) -> None:
        if self.fields:
            _validate_fields(self.fields, PaperFields.VALID_FIELDS)
        if self.sort:
            try:
                field, order = self.sort.split(":")
            except ValueError as exc:
                _raise_validation("Sort must be in format 'field:order'", field="sort")
                raise AssertionError("unreachable") from exc
            if field not in VALID_BULK_SORT_FIELDS:
                _raise_validation(
                    "Invalid sort field. Must be one of: paperId, publicationDate, citationCount",
                    field="sort",
                )
            if order not in VALID_BULK_SORT_ORDERS:
                _raise_validation(
                    "Invalid sort order. Must be one of: asc, desc",
                    field="sort",
                )
        if self.publication_types:
            invalid_types = set(self.publication_types) - VALID_PUBLICATION_TYPES
            if invalid_types:
                _raise_validation(
                    f"Invalid publication types: {', '.join(invalid_types)}",
                    {"valid_types": list(VALID_PUBLICATION_TYPES)},
                    field="publication_types",
                )
        if self.min_citation_count is not None and self.min_citation_count < 0:
            _raise_validation("Minimum citation count cannot be negative", field="min_citation_count")
        if self.fields_of_study:
            invalid_fields = set(self.fields_of_study) - VALID_FIELDS_OF_STUDY
            if invalid_fields:
                _raise_validation(
                    f"Invalid fields of study: {', '.join(invalid_fields)}",
                    {"valid_fields": list(VALID_FIELDS_OF_STUDY)},
                    field="fields_of_study",
                )

    def to_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.query:
            params["query"] = self.query.strip()
        if self.token:
            params["token"] = self.token
        if self.fields:
            params["fields"] = ",".join(self.fields)
        if self.sort:
            params["sort"] = self.sort
        if self.publication_types:
            params["publicationTypes"] = ",".join(self.publication_types)
        if self.open_access_pdf:
            params["openAccessPdf"] = "true"
        if self.min_citation_count is not None:
            params["minCitationCount"] = str(self.min_citation_count)
        if self.publication_date_or_year:
            params["publicationDateOrYear"] = self.publication_date_or_year
        elif self.year:
            params["year"] = self.year
        if self.venue:
            params["venue"] = ",".join(self.venue)
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)
        return params


@dataclass(slots=True)
class PaperTitleSearchRequest(RequestModel):
    query: str
    fields: Optional[list[str]] = None
    publication_types: Optional[list[str]] = None
    open_access_pdf: bool = False
    min_citation_count: Optional[int] = None
    year: Optional[str] = None
    venue: Optional[list[str]] = None
    fields_of_study: Optional[list[str]] = None

    @property
    def endpoint(self) -> str:
        return "/paper/search/match"

    def __post_init__(self) -> None:
        if not self.query.strip():
            _raise_validation("Query string cannot be empty", field="query")
        if self.fields is None:
            self.fields = list(PaperFields.DEFAULT)
        else:
            _validate_fields(self.fields, PaperFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {"query": self.query, "fields": ",".join(self.fields or [])}
        if self.publication_types:
            params["publicationTypes"] = ",".join(self.publication_types)
        if self.open_access_pdf:
            params["openAccessPdf"] = "true"
        if self.min_citation_count is not None:
            params["minCitationCount"] = str(self.min_citation_count)
        if self.year:
            params["year"] = self.year
        if self.venue:
            params["venue"] = ",".join(self.venue)
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)
        return params


@dataclass(slots=True)
class PaperDetailsRequest(RequestModel):
    paper_id: str
    fields: Optional[list[str]] = None

    @property
    def endpoint(self) -> str:
        return f"/paper/{_quote_path_identifier(self.paper_id)}"

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            _raise_validation("Paper ID cannot be empty", field="paper_id")

    def to_params(self) -> dict[str, Any]:
        params = {}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class PaperBatchDetailsRequest(RequestModel):
    paper_ids: list[str]
    fields: Optional[str] = None
    method: str = "POST"

    @property
    def endpoint(self) -> str:
        return "/paper/batch"

    def __post_init__(self) -> None:
        if not self.paper_ids:
            _raise_validation("Paper IDs list cannot be empty", field="paper_ids")
        if len(self.paper_ids) > 500:
            _raise_validation(
                "Cannot process more than 500 paper IDs at once",
                {"max_papers": 500, "received": len(self.paper_ids)},
                field="paper_ids",
            )
        if self.fields:
            _validate_csv_fields(self.fields, PaperFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {}
        if self.fields:
            params["fields"] = self.fields
        return params

    def to_json(self) -> Any:
        return {"ids": self.paper_ids}


@dataclass(slots=True)
class PaperAuthorsRequest(RequestModel):
    paper_id: str
    fields: Optional[list[str]] = None
    offset: int = 0
    limit: int = 100

    @property
    def endpoint(self) -> str:
        return f"/paper/{self.paper_id}/authors"

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            _raise_validation("Paper ID cannot be empty", field="paper_id")
        if self.limit > 1000:
            _raise_validation("Limit cannot exceed 1000", {"max_limit": 1000}, field="limit")
        if self.fields:
            _validate_fields(self.fields, AuthorDetailFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {"offset": self.offset, "limit": self.limit}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class PaperCitationsRequest(RequestModel):
    paper_id: str
    fields: Optional[list[str]] = None
    offset: int = 0
    limit: int = 100

    @property
    def endpoint(self) -> str:
        return f"/paper/{self.paper_id}/citations"

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            _raise_validation("Paper ID cannot be empty", field="paper_id")
        if self.limit > 1000:
            _raise_validation("Limit cannot exceed 1000", {"max_limit": 1000}, field="limit")
        if self.fields:
            _validate_fields(self.fields, CitationReferenceFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {"offset": self.offset, "limit": self.limit}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class PaperReferencesRequest(RequestModel):
    paper_id: str
    fields: Optional[list[str]] = None
    offset: int = 0
    limit: int = 100

    @property
    def endpoint(self) -> str:
        return f"/paper/{self.paper_id}/references"

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            _raise_validation("Paper ID cannot be empty", field="paper_id")
        if self.limit > 1000:
            _raise_validation("Limit cannot exceed 1000", {"max_limit": 1000}, field="limit")
        if self.fields:
            _validate_fields(self.fields, CitationReferenceFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {"offset": self.offset, "limit": self.limit}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class PaperAutocompleteRequest(RequestModel):
    query: str

    @property
    def endpoint(self) -> str:
        return "/paper/autocomplete"

    def __post_init__(self) -> None:
        if not self.query.strip():
            _raise_validation("Query string cannot be empty", field="query")

    def to_params(self) -> dict[str, Any]:
        return {"query": self.query[:100]}


@dataclass(slots=True)
class SnippetSearchRequest(RequestModel):
    query: str
    fields: Optional[list[str]] = None
    limit: int = 10
    paper_ids: Optional[list[str]] = None
    authors: Optional[list[str]] = None
    min_citation_count: Optional[int] = None
    inserted_before: Optional[str] = None
    publication_date_or_year: Optional[str] = None
    year: Optional[str] = None
    venue: Optional[list[str]] = None
    fields_of_study: Optional[list[str]] = None

    @property
    def endpoint(self) -> str:
        return "/snippet/search"

    def __post_init__(self) -> None:
        if not self.query.strip():
            _raise_validation("Query string cannot be empty", field="query")
        if self.limit > 1000:
            _raise_validation("Limit cannot exceed 1000", {"max_limit": 1000}, field="limit")
        if self.limit < 1:
            _raise_validation("Limit must be at least 1", {"min_limit": 1}, field="limit")
        if self.authors and len(self.authors) > 10:
            _raise_validation("Cannot filter by more than 10 authors", {"max_authors": 10}, field="authors")
        if self.paper_ids and len(self.paper_ids) > 100:
            _raise_validation(
                "Cannot filter by more than 100 paper IDs",
                {"max_paper_ids": 100},
                field="paper_ids",
            )

    def to_params(self) -> dict[str, Any]:
        params = {"query": self.query, "limit": self.limit}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        if self.paper_ids:
            params["paperIds"] = ",".join(self.paper_ids)
        if self.authors:
            params["authors"] = ",".join(self.authors)
        if self.min_citation_count is not None:
            params["minCitationCount"] = self.min_citation_count
        if self.inserted_before:
            params["insertedBefore"] = self.inserted_before
        if self.publication_date_or_year:
            params["publicationDateOrYear"] = self.publication_date_or_year
        if self.year:
            params["year"] = self.year
        if self.venue:
            params["venue"] = ",".join(self.venue)
        if self.fields_of_study:
            params["fieldsOfStudy"] = ",".join(self.fields_of_study)
        return params


@dataclass(slots=True)
class AuthorSearchRequest(RequestModel):
    query: str
    fields: Optional[list[str]] = None
    offset: int = 0
    limit: int = 100

    @property
    def endpoint(self) -> str:
        return "/author/search"

    def __post_init__(self) -> None:
        if not self.query.strip():
            _raise_validation("Query string cannot be empty", field="query")
        if self.limit > 1000:
            _raise_validation("Limit cannot exceed 1000", {"max_limit": 1000}, field="limit")
        if self.fields:
            _validate_fields(self.fields, AuthorDetailFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {"query": self.query, "offset": self.offset, "limit": self.limit}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class AuthorDetailsRequest(RequestModel):
    author_id: str
    fields: Optional[list[str]] = None

    @property
    def endpoint(self) -> str:
        return f"/author/{self.author_id}"

    def __post_init__(self) -> None:
        if not self.author_id.strip():
            _raise_validation("Author ID cannot be empty", field="author_id")
        if self.fields:
            _validate_fields(self.fields, AuthorDetailFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class AuthorPapersRequest(RequestModel):
    author_id: str
    fields: Optional[list[str]] = None
    offset: int = 0
    limit: int = 100

    @property
    def endpoint(self) -> str:
        return f"/author/{self.author_id}/papers"

    def __post_init__(self) -> None:
        if not self.author_id.strip():
            _raise_validation("Author ID cannot be empty", field="author_id")
        if self.limit > 1000:
            _raise_validation("Limit cannot exceed 1000", {"max_limit": 1000}, field="limit")

    def to_params(self) -> dict[str, Any]:
        params = {"offset": self.offset, "limit": self.limit}
        if self.fields:
            params["fields"] = ",".join(self.fields)
        return params


@dataclass(slots=True)
class AuthorBatchDetailsRequest(RequestModel):
    author_ids: list[str]
    fields: Optional[str] = None
    method: str = "POST"

    @property
    def endpoint(self) -> str:
        return "/author/batch"

    def __post_init__(self) -> None:
        if not self.author_ids:
            _raise_validation("Author IDs list cannot be empty", field="author_ids")
        if len(self.author_ids) > 1000:
            _raise_validation(
                "Cannot process more than 1000 author IDs at once",
                {"max_authors": 1000, "received": len(self.author_ids)},
                field="author_ids",
            )
        if self.fields:
            _validate_csv_fields(self.fields, AuthorDetailFields.VALID_FIELDS)

    def to_params(self) -> dict[str, Any]:
        params = {}
        if self.fields:
            params["fields"] = self.fields
        return params

    def to_json(self) -> Any:
        return {"ids": self.author_ids}


@dataclass(slots=True)
class PaperRecommendationsSingleRequest(RequestModel):
    paper_id: str
    fields: Optional[str] = None
    limit: int = 100
    from_pool: str = "recent"
    base_url: str = Config.RECOMMENDATIONS_BASE_URL

    @property
    def endpoint(self) -> str:
        return f"/papers/forpaper/{self.paper_id}"

    def __post_init__(self) -> None:
        if self.limit > 500:
            _raise_validation(
                "Cannot request more than 500 recommendations",
                {"max_limit": 500, "requested": self.limit},
                field="limit",
            )
        if self.from_pool not in VALID_RECOMMENDATION_POOLS:
            _raise_validation(
                "Invalid paper pool specified",
                {"valid_pools": list(VALID_RECOMMENDATION_POOLS)},
                field="from_pool",
            )

    def to_params(self) -> dict[str, Any]:
        params = {"limit": self.limit, "from": self.from_pool}
        if self.fields:
            params["fields"] = self.fields
        return params


@dataclass(slots=True)
class PaperRecommendationsMultiRequest(RequestModel):
    positive_paper_ids: list[str]
    negative_paper_ids: Optional[list[str]] = None
    fields: Optional[str] = None
    limit: int = 100
    method: str = "POST"
    base_url: str = Config.RECOMMENDATIONS_BASE_URL

    @property
    def endpoint(self) -> str:
        return "/papers"

    def __post_init__(self) -> None:
        if not self.positive_paper_ids:
            _raise_validation("Must provide at least one positive paper ID", field="positive_paper_ids")
        if self.limit > 500:
            _raise_validation(
                "Cannot request more than 500 recommendations",
                {"max_limit": 500, "requested": self.limit},
                field="limit",
            )

    def to_params(self) -> dict[str, Any]:
        params = {"limit": self.limit}
        if self.fields:
            params["fields"] = self.fields
        return params

    def to_json(self) -> Any:
        return {
            "positivePaperIds": self.positive_paper_ids,
            "negativePaperIds": self.negative_paper_ids or [],
        }
