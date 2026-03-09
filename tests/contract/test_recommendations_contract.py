import pytest

import semantic_scholar_skills.mcp.tools_recommendations as recommendations_api
from semantic_scholar_skills.config import Config


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


async def test_get_paper_recommendations_single_happy_path(mock_make_request):
    payload = {"recommendedPapers": [{"paperId": "p1"}]}
    mock_make_request.install(recommendations_api).queue_responses(payload)

    result = await recommendations_api.get_paper_recommendations_single.fn(
        None,
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        fields="title,year",
        limit=25,
        from_pool="all-cs",
    )

    assert result == payload
    assert_single_call(
        mock_make_request,
        endpoint="/papers/forpaper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        params={"limit": 25, "from": "all-cs", "fields": "title,year"},
        base_url=Config.RECOMMENDATIONS_BASE_URL,
    )


async def test_get_paper_recommendations_single_invalid_pool_validation(mock_make_request):
    result = await recommendations_api.get_paper_recommendations_single.fn(
        None,
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        from_pool="legacy",
    )

    assert_validation_error(
        result,
        "Invalid paper pool specified",
        {"valid_pools": ["recent", "all-cs"]},
    )
    assert mock_make_request.calls == []


async def test_get_paper_recommendations_single_404_maps_to_paper_not_found(mock_make_request, mock_error_response):
    mock_make_request.install(recommendations_api).queue_responses(mock_error_response(status_code=404))

    result = await recommendations_api.get_paper_recommendations_single.fn(
        None,
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        fields="title,year",
        limit=25,
        from_pool="all-cs",
    )

    assert_validation_error(
        result,
        "Paper not found",
        {"paper_id": "204e3073870fae3d05bcbc2f6a8e263d9b72e776"},
    )
    assert_single_call(
        mock_make_request,
        endpoint="/papers/forpaper/204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        params={"limit": 25, "from": "all-cs", "fields": "title,year"},
        base_url=Config.RECOMMENDATIONS_BASE_URL,
    )


async def test_get_paper_recommendations_multi_happy_path(mock_make_request):
    payload = {"recommendedPapers": [{"paperId": "p4"}]}
    mock_make_request.install(recommendations_api).queue_responses(payload)

    result = await recommendations_api.get_paper_recommendations_multi.fn(
        None,
        positive_paper_ids=["p1", "p2"],
        negative_paper_ids=["p3"],
        fields="title,year",
        limit=25,
    )

    assert result == payload
    assert_single_call(
        mock_make_request,
        endpoint="/papers",
        params={"limit": 25, "fields": "title,year"},
        method="POST",
        json={"positivePaperIds": ["p1", "p2"], "negativePaperIds": ["p3"]},
        base_url=Config.RECOMMENDATIONS_BASE_URL,
    )


async def test_get_paper_recommendations_multi_requires_positive_ids(mock_make_request):
    result = await recommendations_api.get_paper_recommendations_multi.fn(None, positive_paper_ids=[])

    assert_validation_error(result, "Must provide at least one positive paper ID")
    assert mock_make_request.calls == []


async def test_get_paper_recommendations_multi_404_maps_to_missing_input_papers(
    mock_make_request,
    mock_error_response,
):
    mock_make_request.install(recommendations_api).queue_responses(mock_error_response(status_code=404))

    result = await recommendations_api.get_paper_recommendations_multi.fn(
        None,
        positive_paper_ids=["p1", "p2"],
        negative_paper_ids=["p3"],
        fields="title,year",
        limit=25,
    )

    assert_validation_error(
        result,
        "One or more input papers not found",
        {"positive_ids": ["p1", "p2"], "negative_ids": ["p3"]},
    )
    assert_single_call(
        mock_make_request,
        endpoint="/papers",
        params={"limit": 25, "fields": "title,year"},
        method="POST",
        json={"positivePaperIds": ["p1", "p2"], "negativePaperIds": ["p3"]},
        base_url=Config.RECOMMENDATIONS_BASE_URL,
    )
