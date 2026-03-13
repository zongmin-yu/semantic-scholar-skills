from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from ..config import Config
from ..core.client import S2Client
from ..core.exceptions import S2Error, S2NotFoundError, S2ValidationError
from ..core.requests import (
    PaperAutocompleteRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    SnippetSearchRequest,
)
from .models import PaperSummary, ResolvedPaper, SnippetEvidence, TriageCandidate, TriageResult
from .resolve import normalize_paper_query
from .scoring import score_triage_candidate

TRIAGE_PAPER_FIELDS: tuple[str, ...] = (
    "paperId",
    "corpusId",
    "externalIds",
    "title",
    "abstract",
    "year",
    "venue",
    "url",
    "authors",
    "citationCount",
    "influentialCitationCount",
    "fieldsOfStudy",
    "publicationTypes",
)
SNIPPET_FIELDS: list[str] = ["snippet.text"]


def _needs_hydration(candidate: dict[str, Any]) -> bool:
    source = candidate.get("_source")
    if source == "autocomplete":
        return True
    return not candidate.get("abstract") or not candidate.get("authors")


def _rank_key(candidate: TriageCandidate) -> tuple[float, int, bool, int]:
    return (
        -candidate.score,
        -(candidate.paper.citation_count or 0),
        candidate.paper.year is None,
        -(candidate.paper.year or 0),
    )


def _candidate_why(candidate: TriageCandidate) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.title_match:
        reasons.append("exact title match")
    if candidate.snippet_evidence:
        reasons.append("strong snippet evidence")
    if candidate.paper.citation_count:
        reasons.append("high citation impact")
    if candidate.paper.year and candidate.score_breakdown and candidate.score_breakdown.component_dict().get("recency", 0.0) > 0.05:
        reasons.append("recent and relevant")
    if candidate.autocomplete_rank is not None:
        reasons.append("appeared in autocomplete suggestions")
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return tuple(deduped[:3])


async def paper_triage(
    client: S2Client,
    query: str,
    *,
    api_key_override: str | None = None,
    shortlist_size: int = 7,
    relevance_limit: int = 10,
    bulk_candidate_limit: int = 20,
    snippet_candidate_limit: int = 5,
    snippet_limit_per_paper: int = 3,
) -> TriageResult:
    normalized_query = normalize_paper_query(query)
    if not normalized_query:
        raise S2ValidationError(message="Query string cannot be empty", field="query")

    title_task = asyncio.create_task(
        client.match_paper_title(
            PaperTitleSearchRequest(query=normalized_query, fields=list(TRIAGE_PAPER_FIELDS)),
            api_key_override=api_key_override,
        )
    )
    autocomplete_task = asyncio.create_task(
        client.autocomplete_papers(
            PaperAutocompleteRequest(query=normalized_query),
            api_key_override=api_key_override,
        )
    )
    relevance_payload = await client.search_papers(
        PaperRelevanceSearchRequest(
            query=normalized_query,
            fields=list(TRIAGE_PAPER_FIELDS),
            limit=relevance_limit,
        ),
        api_key_override=api_key_override,
    )
    bulk_first_page = await client.bulk_search_papers(
        PaperBulkSearchRequest(
            query=normalized_query,
            fields=list(TRIAGE_PAPER_FIELDS),
            sort="citationCount:desc",
        ),
        api_key_override=api_key_override,
    )

    try:
        title_match_payload = await title_task
    except S2NotFoundError:
        title_match_payload = None
    except S2Error:
        raise
    try:
        autocomplete_payload = await autocomplete_task
    except S2NotFoundError:
        autocomplete_payload = {"matches": []}
    except S2Error:
        raise

    bulk_pages = list(bulk_first_page.get("data", [])) if isinstance(bulk_first_page, dict) else []
    if len(bulk_pages) < bulk_candidate_limit and isinstance(bulk_first_page, dict) and bulk_first_page.get("token"):
        bulk_second_page = await client.bulk_search_papers(
            PaperBulkSearchRequest(
                token=bulk_first_page["token"],
                fields=list(TRIAGE_PAPER_FIELDS),
                sort="citationCount:desc",
            ),
            api_key_override=api_key_override,
        )
        bulk_pages.extend(bulk_second_page.get("data", []))

    title_match_record: dict[str, Any] | None = None
    if isinstance(title_match_payload, dict):
        title_match_data = title_match_payload.get("data")
        if isinstance(title_match_data, list) and title_match_data:
            first_match = title_match_data[0]
            if isinstance(first_match, dict):
                title_match_record = first_match
        elif title_match_payload.get("paperId"):
            title_match_record = title_match_payload

    ordered_candidates: list[dict[str, Any]] = []
    if isinstance(title_match_record, dict) and title_match_record.get("paperId"):
        ordered_candidates.append({"_source": "title_match", "_title_match": True, **title_match_record})

    for index, match in enumerate(autocomplete_payload.get("matches", []) if isinstance(autocomplete_payload, dict) else [], start=1):
        paper_id = (
            str(match.get("id") or match.get("paperId"))
            if isinstance(match, dict) and (match.get("id") or match.get("paperId"))
            else None
        )
        if isinstance(match, dict) and paper_id:
            ordered_candidates.append(
                {
                    "_source": "autocomplete",
                    "_title_match": False,
                    "_autocomplete_rank": index,
                    **match,
                    "paperId": paper_id,
                }
            )

    for item in relevance_payload.get("data", []) if isinstance(relevance_payload, dict) else []:
        if isinstance(item, dict) and item.get("paperId"):
            ordered_candidates.append({"_source": "relevance", "_title_match": False, **item})

    for item in bulk_pages:
        if isinstance(item, dict) and item.get("paperId"):
            ordered_candidates.append({"_source": "bulk", "_title_match": False, **item})

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate in ordered_candidates:
        paper_id = candidate.get("paperId")
        if paper_id in seen_ids:
            continue
        deduped.append(candidate)
        seen_ids.add(paper_id)

    hydrate_ids = [candidate["paperId"] for candidate in deduped if _needs_hydration(candidate)]
    hydrated_by_id: dict[str, dict[str, Any]] = {}
    for index in range(0, len(hydrate_ids), min(Config.MAX_BATCH_SIZE, 500)):
        chunk = hydrate_ids[index : index + min(Config.MAX_BATCH_SIZE, 500)]
        hydrated = await client.batch_papers(
            PaperBatchDetailsRequest(
                paper_ids=chunk,
                fields=",".join(TRIAGE_PAPER_FIELDS),
            ),
            api_key_override=api_key_override,
        )
        for payload in hydrated:
            if not isinstance(payload, dict):
                continue
            paper_id = payload.get("paperId")
            if not paper_id:
                continue
            hydrated_by_id[str(paper_id)] = payload

    materialized: list[dict[str, Any]] = []
    for candidate in deduped:
        merged = dict(candidate)
        hydrated = hydrated_by_id.get(candidate["paperId"])
        if hydrated:
            merged.update(hydrated)
        materialized.append(merged)

    preliminary: list[TriageCandidate] = []
    for candidate in materialized:
        paper = PaperSummary.from_api_response(candidate)
        breakdown = score_triage_candidate(
            paper,
            query=normalized_query,
            snippets=(),
            title_match=bool(candidate.get("_title_match")),
            autocomplete_rank=candidate.get("_autocomplete_rank"),
        )
        preliminary.append(
            TriageCandidate(
                paper=paper,
                score=breakdown.total,
                score_breakdown=breakdown,
                autocomplete_rank=candidate.get("_autocomplete_rank"),
                title_match=bool(candidate.get("_title_match")),
            )
        )
    preliminary.sort(key=_rank_key)
    snippet_targets = preliminary[:snippet_candidate_limit]

    snippet_map: dict[str, tuple[SnippetEvidence, ...]] = {}

    async def fetch_snippets(candidate: TriageCandidate) -> tuple[str, tuple[SnippetEvidence, ...]]:
        payload = await client.search_snippets(
            SnippetSearchRequest(
                query=normalized_query,
                fields=SNIPPET_FIELDS,
                limit=snippet_limit_per_paper,
                paper_ids=[candidate.paper.paper_id],
            ),
            api_key_override=api_key_override,
        )
        evidence = tuple(
            SnippetEvidence.from_api_response(
                item,
                fallback_paper_id=candidate.paper.paper_id,
                fallback_paper_title=candidate.paper.title,
            )
            for item in payload.get("data", [])
            if isinstance(item, dict)
        )
        return candidate.paper.paper_id, evidence

    if snippet_targets:
        for paper_id, evidence in await asyncio.gather(*(fetch_snippets(candidate) for candidate in snippet_targets)):
            snippet_map[paper_id] = evidence

    final_candidates: list[TriageCandidate] = []
    for candidate in preliminary:
        snippets = snippet_map.get(candidate.paper.paper_id, ())
        breakdown = score_triage_candidate(
            candidate.paper,
            query=normalized_query,
            snippets=snippets,
            title_match=candidate.title_match,
            autocomplete_rank=candidate.autocomplete_rank,
        )
        final = replace(
            candidate,
            score=breakdown.total,
            score_breakdown=breakdown,
            snippet_evidence=snippets,
        )
        final_candidates.append(replace(final, why=_candidate_why(final)))

    final_candidates.sort(key=_rank_key)
    ranked_shortlist = tuple(
        replace(candidate, rank=index)
        for index, candidate in enumerate(final_candidates[:shortlist_size], start=1)
    )

    possible_interpretations: list[ResolvedPaper] = []
    if isinstance(title_match_record, dict) and title_match_record.get("paperId"):
        possible_interpretations.append(
            ResolvedPaper(
                query=query,
                normalized_query=normalized_query,
                match_type="title",
                source="title_match",
                confidence=0.95,
                paper=PaperSummary.from_api_response(title_match_record),
            )
        )
    for candidate in materialized:
        if candidate.get("_source") != "autocomplete":
            continue
        rank = candidate.get("_autocomplete_rank")
        if rank is None:
            continue
        possible_interpretations.append(
            ResolvedPaper(
                query=query,
                normalized_query=normalized_query,
                match_type="title",
                source="autocomplete",
                confidence=max(0.1, 0.70 - 0.05 * (rank - 1)),
                paper=PaperSummary.from_api_response(candidate),
            )
        )
        if len(possible_interpretations) >= 5:
            break

    return TriageResult(
        query=query,
        normalized_query=normalized_query,
        possible_interpretations=tuple(possible_interpretations[:5]),
        shortlist=ranked_shortlist,
        considered_candidates=len(materialized),
    )
