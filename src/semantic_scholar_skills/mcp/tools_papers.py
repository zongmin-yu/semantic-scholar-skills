"""
Paper-related API endpoints for the Semantic Scholar API.
"""

from typing import Dict, List, Optional

from fastmcp import Context

from ..config import ErrorType
from ..core.client import S2Client, make_compat_client
from ..core.exceptions import S2ApiError, S2Error, S2ValidationError
from ..core.requests import (
    PaperAutocompleteRequest,
    PaperAuthorsRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperCitationsRequest,
    PaperDetailsRequest,
    PaperReferencesRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    SnippetSearchRequest,
)
from . import create_error_response, make_request, mcp, s2_exception_to_error_response


def _client() -> S2Client:
    return make_compat_client(make_request)


@mcp.tool()
async def paper_relevance_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 10,
) -> Dict:
    try:
        request = PaperRelevanceSearchRequest(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
            offset=offset,
            limit=limit,
        )
        return await _client().search_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_bulk_search(
    context: Context,
    query: Optional[str] = None,
    token: Optional[str] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    publication_date_or_year: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    try:
        request = PaperBulkSearchRequest(
            query=query,
            token=token,
            fields=fields,
            sort=sort,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        return await _client().bulk_search_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_title_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    try:
        request = PaperTitleSearchRequest(
            query=query,
            fields=fields,
            publication_types=publication_types,
            open_access_pdf=open_access_pdf,
            min_citation_count=min_citation_count,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        return await _client().match_paper_title(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "No matching paper found",
                {"original_query": query},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_details(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
) -> Dict:
    try:
        request = PaperDetailsRequest(paper_id=paper_id, fields=fields)
        return await _client().get_paper(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_batch_details(
    context: Context,
    paper_ids: List[str],
    fields: Optional[str] = None,
) -> Dict:
    try:
        request = PaperBatchDetailsRequest(paper_ids=paper_ids, fields=fields)
        return await _client().batch_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_authors(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    try:
        request = PaperAuthorsRequest(
            paper_id=paper_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_paper_authors(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_citations(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    try:
        request = PaperCitationsRequest(
            paper_id=paper_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_paper_citations(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_references(
    context: Context,
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    try:
        request = PaperReferencesRequest(
            paper_id=paper_id,
            fields=fields,
            offset=offset,
            limit=limit,
        )
        return await _client().get_paper_references(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2ApiError as exc:
        if "404" in exc.message:
            return create_error_response(
                ErrorType.VALIDATION,
                "Paper not found",
                {"paper_id": paper_id},
            )
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def paper_autocomplete(
    context: Context,
    query: str,
) -> Dict:
    try:
        request = PaperAutocompleteRequest(query=query)
        return await _client().autocomplete_papers(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)


@mcp.tool()
async def snippet_search(
    context: Context,
    query: str,
    fields: Optional[List[str]] = None,
    limit: int = 10,
    paper_ids: Optional[List[str]] = None,
    authors: Optional[List[str]] = None,
    min_citation_count: Optional[int] = None,
    inserted_before: Optional[str] = None,
    publication_date_or_year: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    try:
        request = SnippetSearchRequest(
            query=query,
            fields=fields,
            limit=limit,
            paper_ids=paper_ids,
            authors=authors,
            min_citation_count=min_citation_count,
            inserted_before=inserted_before,
            publication_date_or_year=publication_date_or_year,
            year=year,
            venue=venue,
            fields_of_study=fields_of_study,
        )
        return await _client().search_snippets(request)
    except S2ValidationError as exc:
        return s2_exception_to_error_response(exc)
    except S2Error as exc:
        return s2_exception_to_error_response(exc)
