from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from ..config import Config, VALID_RECOMMENDATION_POOLS
from ..core.client import S2Client
from ..core.exceptions import S2ValidationError
from ..core.requests import PaperBatchDetailsRequest, PaperRecommendationsMultiRequest
from .models import ExpandReferencesResult, PaperSummary, ScoredPaper
from .resolve import resolve_papers
from .scoring import (
    CURRENT_YEAR,
    FOUNDATIONAL_AGE_GAP_YEARS,
    METHOD_KEYWORDS,
    SURVEY_BENCHMARK_KEYWORDS,
    citation_impact_score,
    jaccard_overlap,
    score_recommendation_candidate,
)

RECOMMENDATION_FIELDS_CSV = ",".join(
    (
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
)


def _lower_strings(values: Sequence[str]) -> set[str]:
    return {value.strip().lower() for value in values if value and value.strip()}


def _paper_feature_sets(paper: PaperSummary) -> tuple[set[str], set[str]]:
    return (
        _lower_strings(paper.fields_of_study),
        _lower_strings(paper.author_names()),
    )


def _explanation_from_components(components: dict[str, float]) -> tuple[str, ...]:
    reason_map = {
        "seed_similarity": "high field overlap with seeds",
        "impact": "high citation impact",
        "venue_novelty": "novel venue relative to seed set",
        "query_overlap": "strong textual overlap with seed titles",
        "negative_penalty": "penalized for overlap with negative seeds",
    }
    ordered = sorted(components.items(), key=lambda item: abs(item[1]), reverse=True)
    why: list[str] = []
    for name, value in ordered:
        if name == "negative_penalty":
            if value <= -0.10:
                why.append(reason_map[name])
        elif value > 0.10 and name in reason_map:
            why.append(reason_map[name])
        if len(why) == 3:
            break
    return tuple(why)


def _classify_candidate(
    paper: PaperSummary,
    *,
    score: float,
    seed_min_year: int | None,
    resolved_seeds,
) -> tuple[str | None, tuple[str, ...]]:
    text = f"{paper.title} {paper.abstract or ''}".lower()
    publication_types = {item.lower() for item in paper.publication_types}
    candidate_fields = _lower_strings(paper.fields_of_study)
    matched_seed_ids: list[str] = []
    for seed in resolved_seeds:
        seed_fields = _lower_strings(seed.paper.fields_of_study)
        if jaccard_overlap(candidate_fields, seed_fields) >= 0.20:
            matched_seed_ids.append(seed.paper.paper_id)

    if "review" in publication_types or "metaanalysis" in publication_types or any(
        keyword in text for keyword in SURVEY_BENCHMARK_KEYWORDS
    ):
        return "surveys_or_benchmarks", tuple(matched_seed_ids)
    if (
        paper.year is not None
        and seed_min_year is not None
        and paper.year <= seed_min_year - FOUNDATIONAL_AGE_GAP_YEARS
        and citation_impact_score(paper.citation_count, paper.influential_citation_count) >= 0.55
    ):
        return "foundational", tuple(matched_seed_ids)
    if paper.year is not None and paper.year >= CURRENT_YEAR - 3:
        return "recent", tuple(matched_seed_ids)
    if any(keyword in text for keyword in METHOD_KEYWORDS):
        return "methodological", tuple(matched_seed_ids)
    if len(resolved_seeds) >= 2 and len(set(matched_seed_ids)) >= 2 and score >= 0.45:
        return "bridge_papers", tuple(dict.fromkeys(matched_seed_ids))
    return None, tuple(dict.fromkeys(matched_seed_ids))


async def _hydrate_candidates(
    client: S2Client,
    candidates: Sequence[dict[str, Any]],
    *,
    api_key_override: str | None,
) -> list[dict[str, Any]]:
    candidate_by_id = {
        str(candidate["paperId"]): dict(candidate)
        for candidate in candidates
        if candidate.get("paperId")
    }
    if not candidate_by_id:
        return []

    chunk_size = min(Config.MAX_BATCH_SIZE, 500)
    ids = list(candidate_by_id)
    for index in range(0, len(ids), chunk_size):
        chunk_ids = ids[index : index + chunk_size]
        hydrated_chunk = await client.batch_papers(
            PaperBatchDetailsRequest(paper_ids=chunk_ids, fields=RECOMMENDATION_FIELDS_CSV),
            api_key_override=api_key_override,
        )
        for hydrated in hydrated_chunk:
            if not isinstance(hydrated, dict):
                continue
            paper_id = hydrated.get("paperId")
            if not paper_id:
                continue
            paper_id = str(paper_id)
            if paper_id not in candidate_by_id:
                continue
            candidate_by_id[paper_id] = {**candidate_by_id[paper_id], **hydrated}
    merged: list[dict[str, Any]] = []
    for paper_id in ids:
        candidate = candidate_by_id[paper_id]
        if candidate.get("paperId") and candidate.get("title"):
            merged.append(candidate)
    return merged


async def expand_references(
    client: S2Client,
    seeds: Sequence[str],
    *,
    negative_seeds: Sequence[str] = (),
    api_key_override: str | None = None,
    recommendation_pool: str = "all-cs",
    recommendation_limit: int = 60,
    per_bucket_limit: int = 5,
) -> ExpandReferencesResult:
    if not 1 <= len(seeds) <= 3:
        raise S2ValidationError(message="Must provide between 1 and 3 seed papers", field="seeds")
    if recommendation_pool not in VALID_RECOMMENDATION_POOLS:
        raise S2ValidationError(message="Invalid recommendation pool specified", field="recommendation_pool")
    if not 1 <= per_bucket_limit <= 10:
        raise S2ValidationError(message="Per-bucket limit must be between 1 and 10", field="per_bucket_limit")

    resolved_seeds = await resolve_papers(
        client,
        seeds,
        api_key_override=api_key_override,
        include_alternatives=False,
    )
    resolved_negative_seeds = await resolve_papers(
        client,
        negative_seeds,
        api_key_override=api_key_override,
        include_alternatives=False,
    )

    positive_ids = [seed.paper.paper_id for seed in resolved_seeds]
    negative_ids = [seed.paper.paper_id for seed in resolved_negative_seeds]
    if len(set(positive_ids)) != len(positive_ids):
        raise S2ValidationError(message="Resolved seed papers must be unique", field="seeds")
    if set(positive_ids) & set(negative_ids):
        raise S2ValidationError(message="Negative seeds cannot overlap with positive seeds", field="negative_seeds")

    response = await client.recommend_from_papers(
        PaperRecommendationsMultiRequest(
            positive_paper_ids=positive_ids,
            negative_paper_ids=negative_ids or None,
            fields=RECOMMENDATION_FIELDS_CSV,
            limit=recommendation_limit,
        ),
        api_key_override=api_key_override,
    )

    notes: list[str] = []
    raw_candidates = response.get("recommendedPapers", []) if isinstance(response, dict) else []
    filtered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    excluded_ids = set(positive_ids) | set(negative_ids)
    for item in raw_candidates:
        if not isinstance(item, dict):
            continue
        paper_id = item.get("paperId")
        if not paper_id:
            notes.append("Dropped recommendation without paperId")
            continue
        if paper_id in excluded_ids or paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        filtered.append(item)

    if not filtered:
        return ExpandReferencesResult(
            seeds=tuple(resolved_seeds),
            negative_seeds=tuple(resolved_negative_seeds),
            considered_candidates=0,
            notes=tuple(notes),
        )

    merged_records = await _hydrate_candidates(
        client,
        filtered,
        api_key_override=api_key_override,
    )
    if not merged_records:
        return ExpandReferencesResult(
            seeds=tuple(resolved_seeds),
            negative_seeds=tuple(resolved_negative_seeds),
            considered_candidates=0,
            notes=tuple(notes),
        )

    seed_titles = [seed.paper.title for seed in resolved_seeds]
    seed_fields = set().union(*(_lower_strings(seed.paper.fields_of_study) for seed in resolved_seeds))
    seed_author_names = set().union(*(_lower_strings(seed.paper.author_names()) for seed in resolved_seeds))
    negative_seed_fields = set().union(
        *(_lower_strings(seed.paper.fields_of_study) for seed in resolved_negative_seeds)
    )
    negative_seed_author_names = set().union(
        *(_lower_strings(seed.paper.author_names()) for seed in resolved_negative_seeds)
    )
    seen_venues = _lower_strings([seed.paper.venue or "" for seed in resolved_seeds])
    seed_min_year = min((seed.paper.year for seed in resolved_seeds if seed.paper.year is not None), default=None)

    scored: list[ScoredPaper] = []
    for record in merged_records:
        paper = PaperSummary.from_api_response(record)
        breakdown = score_recommendation_candidate(
            paper,
            seed_titles=seed_titles,
            seed_fields=seed_fields,
            seed_author_names=seed_author_names,
            negative_seed_fields=negative_seed_fields,
            negative_seed_author_names=negative_seed_author_names,
            seen_venues=seen_venues,
        )
        primary_category, matched_seed_ids = _classify_candidate(
            paper,
            score=breakdown.total,
            seed_min_year=seed_min_year,
            resolved_seeds=resolved_seeds,
        )
        scored.append(
            ScoredPaper(
                paper=paper,
                score=breakdown.total,
                score_breakdown=breakdown,
                primary_category=primary_category,
                matched_seed_ids=matched_seed_ids,
                why=_explanation_from_components(breakdown.component_dict()),
            )
        )

    scored.sort(
        key=lambda item: (
            -item.score,
            -(item.paper.citation_count or -1),
            item.paper.year is None,
            -(item.paper.year or -1),
        )
    )
    ranked = tuple(replace(item, rank=index) for index, item in enumerate(scored, start=1))

    def select_bucket(category: str) -> tuple[ScoredPaper, ...]:
        return tuple(item for item in ranked if item.primary_category == category)[:per_bucket_limit]

    return ExpandReferencesResult(
        seeds=tuple(resolved_seeds),
        negative_seeds=tuple(resolved_negative_seeds),
        closest_neighbors=ranked[:per_bucket_limit],
        bridge_papers=select_bucket("bridge_papers"),
        foundational=select_bucket("foundational"),
        methodological=select_bucket("methodological"),
        recent=select_bucket("recent"),
        surveys_or_benchmarks=select_bucket("surveys_or_benchmarks"),
        considered_candidates=len(ranked),
        notes=tuple(notes),
    )
