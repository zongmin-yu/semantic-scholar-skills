from __future__ import annotations

import pytest

from semantic_scholar_skills.engine import trace_citations

ATTENTION_PAPER_ID = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_trace_citations_returns_edges_for_attention_seed(live_client) -> None:
    result = await trace_citations(
        live_client,
        ATTENTION_PAPER_ID,
        max_references=10,
        max_citations=10,
    )

    total_edges = (
        len(result.foundations)
        + len(result.direct_descendants)
        + len(result.bridge_nodes)
        + len(result.weak_edges)
        + len(result.second_hop)
    )
    assert total_edges >= 1

