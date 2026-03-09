import pytest

import semantic_scholar_skills.mcp.tools_authors as authors_api


pytestmark = pytest.mark.asyncio


def assert_single_call(
    mock_make_request,
    *,
    endpoint,
    params,
    api_key_override=None,
    method="GET",
    json=None,
    base_url=None,
):
    assert mock_make_request.calls == [
        {
            "endpoint": endpoint,
            "params": params,
            "api_key_override": api_key_override,
            "method": method,
            "json": json,
            "base_url": base_url,
        }
    ]


def assert_validation_error(result, message, details=None):
    assert result == {
        "error": {
            "type": "validation",
            "message": message,
            "details": details or {},
        }
    }


async def test_author_search_happy_path(mock_make_request):
    payload = {"total": 1, "offset": 10, "data": [{"authorId": "1741101"}]}
    mock_make_request.install(authors_api).queue_responses(payload)

    result = await authors_api.author_search.fn(
        None,
        query="Andrew Ng",
        fields=["name", "paperCount"],
        offset=10,
        limit=25,
    )

    assert result == payload
    assert_single_call(
        mock_make_request,
        endpoint="/author/search",
        params={
            "query": "Andrew Ng",
            "offset": 10,
            "limit": 25,
            "fields": "name,paperCount",
        },
    )


async def test_author_search_empty_query_validation(mock_make_request):
    result = await authors_api.author_search.fn(None, query="   ")

    assert_validation_error(result, "Query string cannot be empty")
    assert mock_make_request.calls == []


async def test_author_search_rate_limit_passthrough(mock_make_request, mock_error_response):
    error = mock_error_response(status_code=429)
    mock_make_request.install(authors_api).queue_responses(error)

    result = await authors_api.author_search.fn(
        None,
        query="Andrew Ng",
        fields=["name", "paperCount"],
        offset=10,
        limit=25,
    )

    assert result == error
    assert_single_call(
        mock_make_request,
        endpoint="/author/search",
        params={
            "query": "Andrew Ng",
            "offset": 10,
            "limit": 25,
            "fields": "name,paperCount",
        },
    )


async def test_author_details_happy_path(mock_make_request):
    payload = {"authorId": "1741101", "name": "Andrew Ng"}
    mock_make_request.install(authors_api).queue_responses(payload)

    result = await authors_api.author_details.fn(None, author_id="1741101", fields=["name", "hIndex"])

    assert result == payload
    assert_single_call(
        mock_make_request,
        endpoint="/author/1741101",
        params={"fields": "name,hIndex"},
    )


async def test_author_details_empty_author_id_validation(mock_make_request):
    result = await authors_api.author_details.fn(None, author_id="   ")

    assert_validation_error(result, "Author ID cannot be empty")
    assert mock_make_request.calls == []


async def test_author_details_404_maps_to_author_not_found(mock_make_request, mock_error_response):
    mock_make_request.install(authors_api).queue_responses(mock_error_response(status_code=404))

    result = await authors_api.author_details.fn(None, author_id="1741101", fields=["name", "hIndex"])

    assert_validation_error(result, "Author not found", {"author_id": "1741101"})
    assert_single_call(
        mock_make_request,
        endpoint="/author/1741101",
        params={"fields": "name,hIndex"},
    )


async def test_author_papers_happy_path(mock_make_request):
    payload = {"offset": 0, "next": 10, "data": [{"paperId": "p1"}]}
    mock_make_request.install(authors_api).queue_responses(payload)

    result = await authors_api.author_papers.fn(
        None,
        author_id="1741101",
        fields=["title", "year"],
        offset=0,
        limit=10,
    )

    assert result == payload
    assert_single_call(
        mock_make_request,
        endpoint="/author/1741101/papers",
        params={"offset": 0, "limit": 10, "fields": "title,year"},
    )


async def test_author_papers_limit_validation(mock_make_request):
    result = await authors_api.author_papers.fn(None, author_id="1741101", limit=1001)

    assert_validation_error(result, "Limit cannot exceed 1000", {"max_limit": 1000})
    assert mock_make_request.calls == []


async def test_author_papers_404_maps_to_author_not_found(mock_make_request, mock_error_response):
    mock_make_request.install(authors_api).queue_responses(mock_error_response(status_code=404))

    result = await authors_api.author_papers.fn(
        None,
        author_id="1741101",
        fields=["title", "year"],
        offset=0,
        limit=10,
    )

    assert_validation_error(result, "Author not found", {"author_id": "1741101"})
    assert_single_call(
        mock_make_request,
        endpoint="/author/1741101/papers",
        params={"offset": 0, "limit": 10, "fields": "title,year"},
    )


async def test_author_batch_details_happy_path(mock_make_request):
    payload = [{"authorId": "1741101"}, {"authorId": "2061296"}]
    mock_make_request.install(authors_api).queue_responses(payload)

    result = await authors_api.author_batch_details.fn(
        None,
        author_ids=["1741101", "2061296"],
        fields="name,paperCount",
    )

    assert result == payload
    assert_single_call(
        mock_make_request,
        endpoint="/author/batch",
        params={"fields": "name,paperCount"},
        method="POST",
        json={"ids": ["1741101", "2061296"]},
    )


async def test_author_batch_details_empty_ids_validation(mock_make_request):
    result = await authors_api.author_batch_details.fn(None, author_ids=[], fields="name,paperCount")

    assert_validation_error(result, "Author IDs list cannot be empty")
    assert mock_make_request.calls == []


async def test_author_batch_details_rate_limit_passthrough(mock_make_request, mock_error_response):
    error = mock_error_response(status_code=429)
    mock_make_request.install(authors_api).queue_responses(error)

    result = await authors_api.author_batch_details.fn(
        None,
        author_ids=["1741101", "2061296"],
        fields="name,paperCount",
    )

    assert result == error
    assert_single_call(
        mock_make_request,
        endpoint="/author/batch",
        params={"fields": "name,paperCount"},
        method="POST",
        json={"ids": ["1741101", "2061296"]},
    )
