from __future__ import annotations

import asyncio
import re
from typing import Literal, Sequence

from ..config import Config
from ..core.client import S2Client
from ..core.exceptions import S2Error, S2NotFoundError, S2ValidationError
from ..core.requests import PaperAutocompleteRequest, PaperBatchDetailsRequest, PaperDetailsRequest, PaperTitleSearchRequest
from .models import PaperSummary, ResolvedPaper

RESOLVE_FIELDS: tuple[str, ...] = (
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
    "referenceCount",
    "fieldsOfStudy",
    "publicationTypes",
)
DOI_URL_PREFIXES: tuple[str, ...] = ("https://doi.org/", "http://doi.org/", "doi:")
HEX_PAPER_ID_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
CORPUS_ID_RE = re.compile(r"^CorpusId:\d+$", re.IGNORECASE)
GENERIC_PREFIX_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:.+$")


def normalize_paper_query(query: str) -> str:
    collapsed = " ".join(query.strip().split())
    lowered = collapsed.lower()
    for prefix in DOI_URL_PREFIXES:
        if lowered.startswith(prefix):
            return collapsed[len(prefix):].lstrip()
    return collapsed


def detect_query_kind(query: str) -> Literal["doi", "paper_id", "title"]:
    normalized = normalize_paper_query(query)
    lowered = normalized.lower()
    if lowered.startswith("10."):
        return "doi"
    if HEX_PAPER_ID_RE.match(normalized):
        return "paper_id"
    if CORPUS_ID_RE.match(normalized):
        return "paper_id"
    if " " not in normalized and GENERIC_PREFIX_ID_RE.match(normalized):
        return "paper_id"
    return "title"


def _resolve_request_fields() -> list[str]:
    return list(RESOLVE_FIELDS)


async def _hydrate_autocomplete_matches(
    client: S2Client,
    paper_ids: Sequence[str],
    *,
    api_key_override: str | None,
) -> tuple[PaperSummary, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for paper_id in paper_ids:
        if not paper_id or paper_id in seen:
            continue
        deduped.append(paper_id)
        seen.add(paper_id)
    if not deduped:
        return ()

    chunk_size = min(Config.MAX_BATCH_SIZE, 500)
    hydrated_by_id: dict[str, PaperSummary] = {}
    for index in range(0, len(deduped), chunk_size):
        chunk = deduped[index : index + chunk_size]
        response = await client.batch_papers(
            PaperBatchDetailsRequest(paper_ids=chunk, fields=",".join(RESOLVE_FIELDS)),
            api_key_override=api_key_override,
        )
        for item in response:
            if not isinstance(item, dict):
                continue
            paper_id = item.get("paperId")
            if not paper_id:
                continue
            hydrated_by_id[str(paper_id)] = PaperSummary.from_api_response(item)

    return tuple(hydrated_by_id[paper_id] for paper_id in deduped if paper_id in hydrated_by_id)


async def resolve_paper(
    client: S2Client,
    query: str,
    *,
    api_key_override: str | None = None,
    include_alternatives: bool = True,
    autocomplete_limit: int = 5,
) -> ResolvedPaper:
    normalized_query = normalize_paper_query(query)
    if not normalized_query:
        raise S2ValidationError(message="Query string cannot be empty", field="query")

    match_type = detect_query_kind(normalized_query)
    if match_type in {"doi", "paper_id"}:
        record = await client.get_paper(
            PaperDetailsRequest(paper_id=normalized_query, fields=_resolve_request_fields()),
            api_key_override=api_key_override,
        )
        return ResolvedPaper(
            query=query,
            normalized_query=normalized_query,
            match_type=match_type,
            source="direct",
            confidence=1.0,
            paper=PaperSummary.from_api_response(record),
        )

    primary_record: dict[str, object] | None = None
    primary_source: Literal["title_match", "autocomplete"] | None = None
    confidence = 0.0
    try:
        title_match = await client.match_paper_title(
            PaperTitleSearchRequest(query=normalized_query, fields=_resolve_request_fields()),
            api_key_override=api_key_override,
        )
    except S2NotFoundError:
        title_match = None
    except S2Error:
        raise

    if isinstance(title_match, dict) and title_match.get("paperId"):
        primary_record = title_match
        primary_source = "title_match"
        confidence = 0.95
        if not include_alternatives:
            return ResolvedPaper(
                query=query,
                normalized_query=normalized_query,
                match_type="title",
                source=primary_source,
                confidence=confidence,
                paper=PaperSummary.from_api_response(primary_record),
            )

    autocomplete = await client.autocomplete_papers(
        PaperAutocompleteRequest(query=normalized_query),
        api_key_override=api_key_override,
    )
    matches = autocomplete.get("matches", []) if isinstance(autocomplete, dict) else []
    autocomplete_ids = [
        match.get("paperId")
        for match in matches
        if isinstance(match, dict) and match.get("paperId")
    ][:autocomplete_limit]
    hydrated = await _hydrate_autocomplete_matches(
        client,
        autocomplete_ids,
        api_key_override=api_key_override,
    )

    if primary_record is not None:
        primary_paper = PaperSummary.from_api_response(primary_record)
        alternatives = tuple(paper for paper in hydrated if paper.paper_id != primary_paper.paper_id)
        return ResolvedPaper(
            query=query,
            normalized_query=normalized_query,
            match_type="title",
            source="title_match",
            confidence=confidence,
            paper=primary_paper,
            alternatives=alternatives,
        )

    if not matches or not hydrated:
        raise S2NotFoundError(
            message=f"Could not resolve paper query: {query}",
            details={},
            resource_type="paper",
            resource_id=query,
        )

    primary_paper = hydrated[0]
    return ResolvedPaper(
        query=query,
        normalized_query=normalized_query,
        match_type="title",
        source="autocomplete",
        confidence=0.70,
        paper=primary_paper,
        alternatives=tuple(hydrated[1:]),
    )


async def resolve_papers(
    client: S2Client,
    queries: Sequence[str],
    *,
    api_key_override: str | None = None,
    include_alternatives: bool = False,
    autocomplete_limit: int = 5,
) -> tuple[ResolvedPaper, ...]:
    cache: dict[str, asyncio.Task[ResolvedPaper]] = {}

    for query in queries:
        normalized_query = normalize_paper_query(query)
        if normalized_query not in cache:
            cache[normalized_query] = asyncio.create_task(
                resolve_paper(
                    client,
                    query,
                    api_key_override=api_key_override,
                    include_alternatives=include_alternatives,
                    autocomplete_limit=autocomplete_limit,
                )
            )

    results_by_query = {
        normalized_query: resolved
        for normalized_query, resolved in zip(cache, await asyncio.gather(*cache.values()))
    }
    return tuple(results_by_query[normalize_paper_query(query)] for query in queries)
