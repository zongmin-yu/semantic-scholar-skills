from __future__ import annotations

import importlib
import pytest

from semantic_scholar_skills.core import PaperCitationsRequest, PaperReferencesRequest, S2ValidationError
from semantic_scholar_skills.engine.models import AuthorSummary, PaperSummary, ResolvedPaper
from semantic_scholar_skills.engine.trace_citations import TRACE_EDGE_FIELDS, trace_citations


@pytest.fixture
def focal_resolved() -> ResolvedPaper:
    return ResolvedPaper(
        query="Attention Is All You Need",
        normalized_query="Attention Is All You Need",
        match_type="title",
        source="title_match",
        confidence=0.95,
        paper=PaperSummary(
            paper_id="p-attn",
            title="Attention Is All You Need",
            year=2017,
            authors=(AuthorSummary(author_id="a1", name="Ashish Vaswani"),),
            citation_count=12000,
            influential_citation_count=3500,
        ),
    )


def patch_resolve_paper(monkeypatch, focal_resolved: ResolvedPaper) -> None:
    async def fake_resolve_paper(client, query, **kwargs):
        return focal_resolved

    trace_module = importlib.import_module("semantic_scholar_skills.engine.trace_citations")
    monkeypatch.setattr(trace_module, "resolve_paper", fake_resolve_paper)


@pytest.mark.asyncio
async def test_trace_citations_resolves_focal_and_fetches_references_and_citations(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
    sample_flat_edge_payload,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue("get_paper_references", sample_flat_edge_payload)
    stub_s2_client.queue("get_paper_citations", {"data": []})

    result = await trace_citations(stub_s2_client, "Attention Is All You Need")

    assert result.focal.paper.paper_id == "p-attn"
    assert isinstance(stub_s2_client.calls[0][1], PaperReferencesRequest)
    assert isinstance(stub_s2_client.calls[1][1], PaperCitationsRequest)
    assert stub_s2_client.calls[0][1].fields == TRACE_EDGE_FIELDS
    assert stub_s2_client.calls[1][1].fields == TRACE_EDGE_FIELDS


@pytest.mark.asyncio
async def test_trace_citations_extracts_edge_metadata_from_flat_payload_items(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
    sample_flat_edge_payload,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue("get_paper_references", sample_flat_edge_payload)
    stub_s2_client.queue("get_paper_citations", {"data": []})

    result = await trace_citations(stub_s2_client, "Attention Is All You Need")

    edge = result.foundations[0]
    assert edge.paper.paper_id == "p-ref-1"
    assert edge.contexts == ("We follow the encoder-decoder setup.",)
    assert edge.intents == ("Background",)
    assert edge.is_influential is True


@pytest.mark.asyncio
async def test_trace_citations_extracts_edge_metadata_from_nested_payload_items(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
    sample_nested_edge_payload,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue("get_paper_references", {"data": []})
    stub_s2_client.queue("get_paper_citations", sample_nested_edge_payload)

    result = await trace_citations(stub_s2_client, "Attention Is All You Need")

    edge = result.direct_descendants[0]
    assert edge.paper.paper_id == "p-rag"
    assert edge.paper.title == "Retrieval-Augmented Generation"
    assert edge.contexts == ("The proposed method extends BERT with retrieval.",)
    assert edge.intents == ("Extends",)


@pytest.mark.asyncio
async def test_trace_citations_scores_and_sorts_foundations_and_direct_descendants(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue(
        "get_paper_references",
        {
            "data": [
                {
                    "paperId": "p-ref-top",
                    "title": "Top Reference",
                    "year": 2014,
                    "citationCount": 9000,
                    "influentialCitationCount": 1200,
                    "contexts": ["Important prior work.", "A second rich context for this reference."],
                    "intents": ["Background"],
                    "isInfluential": True,
                },
                {
                    "paperId": "p-ref-low",
                    "title": "Lower Reference",
                    "year": 2016,
                    "citationCount": 200,
                    "influentialCitationCount": 20,
                    "contexts": ["Brief mention."],
                    "intents": ["Uses"],
                    "isInfluential": False,
                },
            ]
        },
    )
    stub_s2_client.queue(
        "get_paper_citations",
        {
            "data": [
                {
                    "paperId": "p-cite-top",
                    "title": "Top Citation",
                    "year": 2020,
                    "citationCount": 1800,
                    "influentialCitationCount": 240,
                    "contexts": ["Builds directly on the transformer architecture."],
                    "intents": ["Extends"],
                    "isInfluential": True,
                }
            ]
        },
    )

    result = await trace_citations(stub_s2_client, "Attention Is All You Need")

    assert result.foundations[0].paper.paper_id == "p-ref-top"
    assert result.direct_descendants[0].paper.paper_id == "p-cite-top"
    assert result.foundations[0].score >= result.foundations[-1].score


@pytest.mark.asyncio
async def test_trace_citations_populates_bridge_nodes_from_mid_confidence_edges(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue(
        "get_paper_references",
        {
            "data": [
                {
                    "paperId": "p-bridge",
                    "title": "Bridge Paper",
                    "year": 2018,
                    "citationCount": 5000,
                    "influentialCitationCount": 500,
                    "contexts": [
                        "This work compares transformer architectures to sequence models with extensive methodological detail.",
                        "It provides a second explanatory context about connecting otherwise separate citation neighborhoods.",
                        "A third rich context adds more evidence for bridge-like usage.",
                    ],
                    "intents": ["Background", "Compare"],
                    "isInfluential": False,
                }
            ]
        },
    )
    stub_s2_client.queue("get_paper_citations", {"data": []})

    result = await trace_citations(stub_s2_client, "Attention Is All You Need")

    assert result.bridge_nodes[0].paper.paper_id == "p-bridge"


@pytest.mark.asyncio
async def test_trace_citations_populates_weak_edges_when_context_and_intent_are_missing(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue(
        "get_paper_references",
        {
            "data": [
                {
                    "paperId": "p-weak",
                    "title": "Weak Paper",
                    "year": 2016,
                    "citationCount": 1,
                    "influentialCitationCount": 0,
                    "contexts": [],
                    "intents": [],
                    "isInfluential": False,
                }
            ]
        },
    )
    stub_s2_client.queue("get_paper_citations", {"data": []})

    result = await trace_citations(stub_s2_client, "Attention Is All You Need")

    assert result.weak_edges[0].paper.paper_id == "p-weak"


@pytest.mark.asyncio
async def test_trace_citations_expands_second_hop_only_for_top_influential_edges_when_depth_is_two(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
    sample_flat_edge_payload,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)
    stub_s2_client.queue("get_paper_references", sample_flat_edge_payload)
    stub_s2_client.queue("get_paper_citations", {"data": []})
    stub_s2_client.queue(
        "get_paper_references",
        {
            "data": [
                {
                    "paperId": "p-second-hop",
                    "title": "Second Hop Paper",
                    "year": 2012,
                    "citationCount": 4000,
                    "influentialCitationCount": 500,
                    "contexts": ["A second-hop context."],
                    "intents": ["Background"],
                    "isInfluential": True,
                }
            ]
        },
    )

    result = await trace_citations(stub_s2_client, "Attention Is All You Need", depth=2)

    assert result.second_hop[0].paper.paper_id == "p-second-hop"
    assert result.second_hop[0].depth == 2
    assert [name for name, _ in stub_s2_client.calls].count("get_paper_references") == 2


@pytest.mark.asyncio
async def test_trace_citations_rejects_invalid_depth(
    monkeypatch,
    stub_s2_client,
    focal_resolved,
) -> None:
    patch_resolve_paper(monkeypatch, focal_resolved)

    with pytest.raises(S2ValidationError) as exc_info:
        await trace_citations(stub_s2_client, "Attention Is All You Need", depth=3)

    assert exc_info.value.field == "depth"
