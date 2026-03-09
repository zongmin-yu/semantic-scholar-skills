from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from datetime import date
from math import log1p
import re

from .models import CitationEdge, PaperSummary, ScoreBreakdown, SnippetEvidence

CURRENT_YEAR = date.today().year
CITATION_CAP = 5000
INFLUENTIAL_CITATION_CAP = 500
RECENCY_HORIZON_YEARS = 15
RECENT_WINDOW_YEARS = 3
FOUNDATIONAL_AGE_GAP_YEARS = 3
MAX_CONTEXT_COUNT = 3
MAX_CONTEXT_WORDS = 40
METHOD_KEYWORDS = (
    "method",
    "approach",
    "framework",
    "model",
    "architecture",
    "algorithm",
    "technique",
    "system",
    "training",
    "retrieval",
)
SURVEY_BENCHMARK_KEYWORDS = ("survey", "review", "benchmark", "dataset", "leaderboard")
INTENT_WEIGHTS = {
    "background": 1.00,
    "motivation": 0.90,
    "method": 0.90,
    "result": 0.80,
    "uses": 0.70,
    "extends": 0.85,
    "compare": 0.65,
    "future": 0.40,
}
EXPAND_WEIGHTS = {
    "seed_similarity": 0.35,
    "impact": 0.25,
    "recency": 0.15,
    "venue_novelty": 0.10,
    "query_overlap": 0.15,
    "negative_penalty": -0.20,
}
TRACE_WEIGHTS = {
    "influence": 0.35,
    "context_richness": 0.25,
    "intent_signal": 0.20,
    "impact": 0.20,
}
TRIAGE_WEIGHTS = {
    "title_signal": 0.30,
    "snippet_signal": 0.30,
    "impact": 0.20,
    "recency": 0.10,
    "autocomplete_signal": 0.10,
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def tokenize(text: str | None) -> set[str]:
    if not text:
        return set()
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def log_normalize(value: int | None, *, cap: int) -> float:
    return min(log1p(max(value or 0, 0)) / log1p(cap), 1.0)


def citation_impact_score(citation_count: int | None, influential_citation_count: int | None) -> float:
    return (
        0.7 * log_normalize(citation_count, cap=CITATION_CAP)
        + 0.3 * log_normalize(influential_citation_count, cap=INFLUENTIAL_CITATION_CAP)
    )


def recency_score(
    year: int | None,
    *,
    current_year: int = CURRENT_YEAR,
    horizon_years: int = RECENCY_HORIZON_YEARS,
) -> float:
    if year is None:
        return 0.0
    age = min(max(current_year - year, 0), horizon_years)
    return clamp01(1.0 - age / horizon_years)


def jaccard_overlap(left: Collection[str], right: Collection[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def query_overlap_score(query: str, *texts: str | None) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens: set[str] = set()
    for text in texts:
        text_tokens.update(tokenize(text))
    return len(query_tokens & text_tokens) / len(query_tokens)


def venue_novelty_score(candidate_venue: str | None, seen_venues: Collection[str]) -> float:
    if not candidate_venue:
        return 0.5
    candidate_key = candidate_venue.strip().lower()
    seen = {venue.strip().lower() for venue in seen_venues if venue}
    if candidate_key not in seen:
        return 1.0
    return 0.2


def context_richness_score(contexts: Sequence[str]) -> float:
    non_empty = [ctx.strip() for ctx in contexts if ctx and ctx.strip()]
    count_component = min(len(non_empty), MAX_CONTEXT_COUNT) / MAX_CONTEXT_COUNT
    if non_empty:
        avg_words = sum(min(len(ctx.split()), MAX_CONTEXT_WORDS) for ctx in non_empty) / len(non_empty)
    else:
        avg_words = 0.0
    length_component = avg_words / MAX_CONTEXT_WORDS
    return 0.5 * count_component + 0.5 * length_component


def intent_signal_score(intents: Sequence[str]) -> float:
    if not intents:
        return 0.0
    scores: list[float] = []
    for intent in intents:
        normalized = intent.lower()
        score = 0.5
        for key, value in INTENT_WEIGHTS.items():
            if key in normalized:
                score = value
                break
        scores.append(score)
    return sum(scores) / len(scores)


def combine_weighted_scores(
    raw_components: Mapping[str, float],
    weights: Mapping[str, float],
    *,
    reasons: Sequence[str] = (),
) -> ScoreBreakdown:
    weighted_components = tuple(
        (name, raw_components[name] * weights[name])
        for name in raw_components
        if name in weights
    )
    total = sum(component for _, component in weighted_components)
    return ScoreBreakdown(total=clamp01(total), components=weighted_components, reasons=tuple(reasons))


def score_recommendation_candidate(
    paper: PaperSummary,
    *,
    seed_titles: Sequence[str],
    seed_fields: Collection[str],
    seed_author_names: Collection[str],
    negative_seed_fields: Collection[str],
    negative_seed_author_names: Collection[str],
    seen_venues: Collection[str],
) -> ScoreBreakdown:
    candidate_fields = {field.lower() for field in paper.fields_of_study if field}
    candidate_authors = {name.lower() for name in paper.author_names() if name}
    seed_title_overlap = max(
        (query_overlap_score(seed_title, paper.title, paper.abstract) for seed_title in seed_titles),
        default=0.0,
    )
    negative_penalty = 0.0
    if negative_seed_fields or negative_seed_author_names:
        negative_penalty = max(
            jaccard_overlap(candidate_fields, negative_seed_fields),
            jaccard_overlap(candidate_authors, negative_seed_author_names),
        )
    raw_components = {
        "seed_similarity": (
            0.5 * jaccard_overlap(candidate_fields, seed_fields)
            + 0.2 * jaccard_overlap(candidate_authors, seed_author_names)
            + 0.3 * seed_title_overlap
        ),
        "impact": citation_impact_score(paper.citation_count, paper.influential_citation_count),
        "recency": recency_score(paper.year),
        "venue_novelty": venue_novelty_score(paper.venue, seen_venues),
        "query_overlap": seed_title_overlap,
        "negative_penalty": negative_penalty,
    }
    return combine_weighted_scores(raw_components, EXPAND_WEIGHTS)


def score_citation_edge(edge: CitationEdge) -> ScoreBreakdown:
    raw_components = {
        "influence": 1.0 if edge.is_influential else 0.0,
        "context_richness": context_richness_score(edge.contexts),
        "intent_signal": intent_signal_score(edge.intents),
        "impact": citation_impact_score(edge.paper.citation_count, edge.paper.influential_citation_count),
    }
    return combine_weighted_scores(raw_components, TRACE_WEIGHTS)


def score_triage_candidate(
    paper: PaperSummary,
    *,
    query: str,
    snippets: Sequence[SnippetEvidence],
    title_match: bool,
    autocomplete_rank: int | None,
) -> ScoreBreakdown:
    raw_components = {
        "title_signal": 1.0 if title_match else query_overlap_score(query, paper.title, paper.abstract),
        "snippet_signal": max((query_overlap_score(query, snippet.text) for snippet in snippets), default=0.0),
        "impact": citation_impact_score(paper.citation_count, paper.influential_citation_count),
        "recency": recency_score(paper.year),
        "autocomplete_signal": (
            0.0
            if autocomplete_rank is None
            else clamp01(1.0 - (autocomplete_rank - 1) / 4)
        ),
    }
    return combine_weighted_scores(raw_components, TRIAGE_WEIGHTS)
