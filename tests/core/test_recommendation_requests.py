from __future__ import annotations

import pytest

from semantic_scholar_skills.config import Config
from semantic_scholar_skills.core import (
    PaperRecommendationsMultiRequest,
    PaperRecommendationsSingleRequest,
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


def test_paper_recommendations_single_serializes_params_and_base_url() -> None:
    request = PaperRecommendationsSingleRequest(
        paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
        fields="title,year",
        limit=25,
        from_pool="all-cs",
    )

    assert request.endpoint == "/papers/forpaper/204e3073870fae3d05bcbc2f6a8e263d9b72e776"
    assert request.method == "GET"
    assert request.base_url == Config.RECOMMENDATIONS_BASE_URL
    assert request.to_params() == {"limit": 25, "from": "all-cs", "fields": "title,year"}
    assert request.to_json() is None


def test_paper_recommendations_single_invalid_pool_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperRecommendationsSingleRequest(
            paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            from_pool="legacy",
        )

    assert_validation_error(
        excinfo,
        "Invalid paper pool specified",
        field="from_pool",
        details={"valid_pools": ["recent", "all-cs"]},
    )


def test_paper_recommendations_single_limit_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperRecommendationsSingleRequest(
            paper_id="204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            limit=501,
        )

    assert_validation_error(
        excinfo,
        "Cannot request more than 500 recommendations",
        field="limit",
        details={"max_limit": 500, "requested": 501},
    )


def test_paper_recommendations_multi_serializes_json_and_base_url() -> None:
    request = PaperRecommendationsMultiRequest(
        positive_paper_ids=["p1", "p2"],
        negative_paper_ids=["p3"],
        fields="title,year",
        limit=25,
    )

    assert request.endpoint == "/papers"
    assert request.method == "POST"
    assert request.base_url == Config.RECOMMENDATIONS_BASE_URL
    assert request.to_params() == {"limit": 25, "fields": "title,year"}
    assert request.to_json() == {
        "positivePaperIds": ["p1", "p2"],
        "negativePaperIds": ["p3"],
    }


def test_paper_recommendations_multi_requires_positive_ids() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperRecommendationsMultiRequest(positive_paper_ids=[])

    assert_validation_error(
        excinfo,
        "Must provide at least one positive paper ID",
        field="positive_paper_ids",
    )


def test_paper_recommendations_multi_limit_validation() -> None:
    with pytest.raises(S2ValidationError) as excinfo:
        PaperRecommendationsMultiRequest(positive_paper_ids=["p1"], limit=501)

    assert_validation_error(
        excinfo,
        "Cannot request more than 500 recommendations",
        field="limit",
        details={"max_limit": 500, "requested": 501},
    )
