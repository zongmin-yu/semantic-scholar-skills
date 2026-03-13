from __future__ import annotations

import re

import pytest

from semantic_scholar_skills.engine import resolve_paper

HEX_PAPER_ID_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_resolve_attention_is_all_you_need_by_title(live_client) -> None:
    resolved = await resolve_paper(
        live_client,
        "Attention Is All You Need",
        include_alternatives=False,
    )

    assert HEX_PAPER_ID_RE.fullmatch(resolved.paper.paper_id)
    assert resolved.source == "title_match"
    assert "Attention" in resolved.paper.title

