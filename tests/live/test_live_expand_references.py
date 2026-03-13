from __future__ import annotations

import pytest

from semantic_scholar_skills.engine import expand_references

ATTENTION_PAPER_ID = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_expand_references_returns_candidates_for_attention_seed(live_client) -> None:
    result = await expand_references(
        live_client,
        [ATTENTION_PAPER_ID],
        recommendation_limit=10,
        per_bucket_limit=1,
    )

    assert result.considered_candidates >= 1
    assert len(result.closest_neighbors) >= 1

