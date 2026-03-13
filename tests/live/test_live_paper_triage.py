from __future__ import annotations

import pytest

from semantic_scholar_skills.engine import paper_triage


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_paper_triage_returns_shortlist_for_rag_query(live_client) -> None:
    result = await paper_triage(
        live_client,
        "retrieval augmented generation",
        shortlist_size=3,
        relevance_limit=5,
        bulk_candidate_limit=5,
        snippet_candidate_limit=1,
        snippet_limit_per_paper=1,
    )

    assert len(result.shortlist) >= 1
    assert result.shortlist[0].paper.paper_id
