from __future__ import annotations

import asyncio
from dataclasses import replace

from ..core.client import S2Client
from ..core.exceptions import S2ValidationError
from ..core.requests import PaperCitationsRequest, PaperReferencesRequest
from .models import CitationEdge, CitationTraceResult
from .resolve import resolve_paper
from .scoring import context_richness_score, intent_signal_score, score_citation_edge

TRACE_EDGE_FIELDS: list[str] = [
    "paperId",
    "title",
    "abstract",
    "authors",
    "year",
    "venue",
    "url",
    "citationCount",
    "influentialCitationCount",
    "contexts",
    "intents",
    "isInfluential",
]


def _why_for_edge(edge: CitationEdge) -> tuple[str, ...]:
    why: list[str] = []
    components = edge.score_breakdown.component_dict() if edge.score_breakdown else {}
    if edge.is_influential and components.get("influence", 0.0) > 0.0:
        why.append("marked influential by Semantic Scholar")
    if components.get("context_richness", 0.0) >= 0.12:
        why.append("multiple rich citation contexts")
    if components.get("intent_signal", 0.0) >= 0.12:
        why.append("strong intent signal")
    if components.get("impact", 0.0) >= 0.10:
        why.append("high downstream citation impact")
    return tuple(why[:3])


def _score_edges(edges: list[CitationEdge]) -> list[CitationEdge]:
    scored: list[CitationEdge] = []
    for edge in edges:
        breakdown = score_citation_edge(edge)
        scored.append(
            replace(
                edge,
                score=breakdown.total,
                score_breakdown=breakdown,
                why=_why_for_edge(replace(edge, score_breakdown=breakdown)),
            )
        )
    return scored


def _normalize_edges(payload: dict, *, direction: str) -> list[CitationEdge]:
    data = payload.get("data", []) if isinstance(payload, dict) else []
    edges: list[CitationEdge] = []
    for item in data:
        if isinstance(item, dict):
            edges.append(CitationEdge.from_api_response(item, direction=direction))
    return edges


async def trace_citations(
    client: S2Client,
    focal_query: str,
    *,
    api_key_override: str | None = None,
    depth: int = 1,
    max_references: int = 50,
    max_citations: int = 50,
    second_hop_limit: int = 10,
) -> CitationTraceResult:
    if depth not in {1, 2}:
        raise S2ValidationError(message="Depth must be 1 or 2", field="depth")
    if max_references <= 0:
        raise S2ValidationError(message="max_references must be positive", field="max_references")
    if max_citations <= 0:
        raise S2ValidationError(message="max_citations must be positive", field="max_citations")
    if second_hop_limit <= 0:
        raise S2ValidationError(message="second_hop_limit must be positive", field="second_hop_limit")

    focal = await resolve_paper(
        client,
        focal_query,
        api_key_override=api_key_override,
        include_alternatives=False,
    )
    references_payload, citations_payload = await asyncio.gather(
        client.get_paper_references(
            PaperReferencesRequest(
                paper_id=focal.paper.paper_id,
                fields=TRACE_EDGE_FIELDS,
                offset=0,
                limit=min(max_references, 1000),
            ),
            api_key_override=api_key_override,
        ),
        client.get_paper_citations(
            PaperCitationsRequest(
                paper_id=focal.paper.paper_id,
                fields=TRACE_EDGE_FIELDS,
                offset=0,
                limit=min(max_citations, 1000),
            ),
            api_key_override=api_key_override,
        ),
    )

    reference_edges = _score_edges(_normalize_edges(references_payload, direction="reference"))
    citation_edges = _score_edges(_normalize_edges(citations_payload, direction="citation"))
    reference_edges.sort(key=lambda item: item.score, reverse=True)
    citation_edges.sort(key=lambda item: item.score, reverse=True)

    focal_year = focal.paper.year
    foundations = [
        edge
        for edge in reference_edges
        if edge.score >= 0.45
    ]
    foundations.sort(
        key=lambda item: (
            not (
                item.paper.year is not None
                and focal_year is not None
                and item.paper.year < focal_year
            ),
            -item.score,
        )
    )
    direct_descendants = [edge for edge in citation_edges if edge.score >= 0.45]

    used_ids = {edge.paper.paper_id for edge in foundations} | {edge.paper.paper_id for edge in direct_descendants}
    bridge_nodes = [
        edge
        for edge in [*reference_edges, *citation_edges]
        if edge.paper.paper_id not in used_ids
        and edge.score >= 0.40
        and (
            intent_signal_score(edge.intents) >= 0.60
            or context_richness_score(edge.contexts) >= 0.50
        )
    ]
    used_ids |= {edge.paper.paper_id for edge in bridge_nodes}
    weak_edges = [
        edge
        for edge in [*reference_edges, *citation_edges]
        if edge.paper.paper_id not in used_ids
        and edge.score < 0.35
        and not edge.contexts
        and not edge.intents
        and edge.is_influential is not True
    ]

    second_hop: list[CitationEdge] = []
    if depth == 2:
        second_hop_seeds = [
            edge
            for edge in [*foundations, *direct_descendants]
            if edge.is_influential is True or edge.score >= 0.60
        ][:second_hop_limit]

        async def fetch_second_hop(seed_edge: CitationEdge):
            if seed_edge.direction == "reference":
                return await client.get_paper_references(
                    PaperReferencesRequest(
                        paper_id=seed_edge.paper.paper_id,
                        fields=TRACE_EDGE_FIELDS,
                        offset=0,
                        limit=min(max_references, 1000),
                    ),
                    api_key_override=api_key_override,
                )
            return await client.get_paper_citations(
                PaperCitationsRequest(
                    paper_id=seed_edge.paper.paper_id,
                    fields=TRACE_EDGE_FIELDS,
                    offset=0,
                    limit=min(max_citations, 1000),
                ),
                api_key_override=api_key_override,
            )

        second_hop_payloads = await asyncio.gather(*(fetch_second_hop(edge) for edge in second_hop_seeds))
        for seed_edge, payload in zip(second_hop_seeds, second_hop_payloads):
            for edge in _normalize_edges(payload, direction=seed_edge.direction):
                scored = _score_edges([replace(edge, depth=2)])[0]
                second_hop.append(scored)

    return CitationTraceResult(
        focal=focal,
        foundations=tuple(foundations),
        direct_descendants=tuple(direct_descendants),
        bridge_nodes=tuple(bridge_nodes),
        weak_edges=tuple(weak_edges),
        second_hop=tuple(second_hop),
        reference_count_examined=len(reference_edges),
        citation_count_examined=len(citation_edges),
    )
