from __future__ import annotations

import pytest

from semantic_scholar_skills.engine.models import AuthorSummary, CitationEdge, PaperSummary, SnippetEvidence
from semantic_scholar_skills.engine.scoring import (
    CITATION_CAP,
    CURRENT_YEAR,
    INFLUENTIAL_CITATION_CAP,
    RECENCY_HORIZON_YEARS,
    TRACE_WEIGHTS,
    TRIAGE_WEIGHTS,
    clamp01,
    citation_impact_score,
    combine_weighted_scores,
    context_richness_score,
    intent_signal_score,
    jaccard_overlap,
    log_normalize,
    recency_score,
    score_citation_edge,
    score_recommendation_candidate,
    score_triage_candidate,
)


def make_paper(
    *,
    paper_id: str = "p1",
    title: str = "Retrieval-Augmented Language Models",
    abstract: str = "A retrieval method and training framework for language models.",
    year: int | None = 2022,
    venue: str | None = "NeurIPS",
    citation_count: int | None = 500,
    influential_citation_count: int | None = 50,
    fields_of_study: tuple[str, ...] = ("Computer Science", "Machine Learning"),
    author_names: tuple[str, ...] = ("Alice Example", "Bob Example"),
) -> PaperSummary:
    return PaperSummary(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        year=year,
        venue=venue,
        authors=tuple(
            AuthorSummary(author_id=f"a{index}", name=name)
            for index, name in enumerate(author_names, start=1)
        ),
        citation_count=citation_count,
        influential_citation_count=influential_citation_count,
        fields_of_study=fields_of_study,
    )


def test_log_normalize_caps_large_values_at_one() -> None:
    assert log_normalize(CITATION_CAP * 100, cap=CITATION_CAP) == 1.0


def test_citation_impact_score_weights_influential_citations_less_than_total_citations() -> None:
    high_total_only = citation_impact_score(CITATION_CAP, 0)
    high_influential_only = citation_impact_score(0, INFLUENTIAL_CITATION_CAP)

    assert high_total_only == pytest.approx(0.7)
    assert high_influential_only == pytest.approx(0.3)
    assert high_total_only > high_influential_only


def test_recency_score_returns_zero_for_missing_year() -> None:
    assert recency_score(None) == 0.0


def test_recency_score_linearly_decays_over_horizon() -> None:
    midpoint_year = CURRENT_YEAR - (RECENCY_HORIZON_YEARS // 2)

    assert recency_score(CURRENT_YEAR) == 1.0
    assert recency_score(midpoint_year) == pytest.approx(
        clamp01(1.0 - (CURRENT_YEAR - midpoint_year) / RECENCY_HORIZON_YEARS)
    )
    assert recency_score(CURRENT_YEAR - RECENCY_HORIZON_YEARS - 1) == 0.0


def test_jaccard_overlap_handles_empty_sets() -> None:
    assert jaccard_overlap(set(), set()) == 0.0
    assert jaccard_overlap({"a"}, {"a", "b"}) == pytest.approx(0.5)


def test_context_richness_score_rewards_multiple_nonempty_contexts() -> None:
    sparse = context_richness_score(["Short context"])
    rich = context_richness_score(
        [
            "This paper extends retrieval-augmented modeling with a stronger decoder objective.",
            "The framework uses dense retrieval during training and inference for factual grounding.",
            "Experiments compare the method against strong transformer baselines.",
        ]
    )

    assert rich > sparse


def test_intent_signal_score_averages_known_and_unknown_intents() -> None:
    assert intent_signal_score(["Background", "Mystery"]) == pytest.approx((1.0 + 0.5) / 2.0)


def test_score_recommendation_candidate_combines_expected_components() -> None:
    paper = make_paper(
        title="A Retrieval Framework for Language Model Training",
        abstract="This method describes a retrieval architecture and training system.",
        year=CURRENT_YEAR - 1,
        venue="ICLR",
        citation_count=1500,
        influential_citation_count=180,
        fields_of_study=("Computer Science", "Machine Learning"),
        author_names=("Alice Example", "Carol Example"),
    )

    breakdown = score_recommendation_candidate(
        paper,
        seed_titles=("Retrieval-Augmented Generation", "Dense Passage Retrieval"),
        seed_fields={"computer science", "machine learning", "information retrieval"},
        seed_author_names={"alice example", "patrick lewis"},
        negative_seed_fields={"biology"},
        negative_seed_author_names={"other person"},
        seen_venues={"NeurIPS", "ACL"},
    )

    component_names = dict(breakdown.components)
    assert breakdown.total > 0.0
    assert set(component_names) == {
        "seed_similarity",
        "impact",
        "recency",
        "venue_novelty",
        "query_overlap",
        "negative_penalty",
    }
    assert component_names["negative_penalty"] <= 0.0


def test_score_citation_edge_combines_influence_context_intent_and_impact() -> None:
    paper = make_paper(citation_count=2000, influential_citation_count=250)
    edge = CitationEdge(
        direction="citation",
        paper=paper,
        contexts=(
            "This paper extends the retrieval approach with stronger supervision.",
            "The method builds on dense retrieval and joint training.",
        ),
        intents=("Extends", "Method"),
        is_influential=True,
    )

    breakdown = score_citation_edge(edge)
    expected_influence = TRACE_WEIGHTS["influence"] * 1.0

    assert breakdown.total > 0.0
    assert dict(breakdown.components)["influence"] == pytest.approx(expected_influence)


def test_score_triage_candidate_rewards_exact_title_match_and_snippets() -> None:
    paper = make_paper(
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="A bidirectional transformer pre-training objective.",
        year=2019,
        citation_count=90000,
        influential_citation_count=5000,
    )
    snippets = (
        SnippetEvidence(
            text="This paper introduces a bidirectional transformer pre-training objective.",
            paper_id="p-bert",
            paper_title=paper.title,
        ),
    )

    breakdown = score_triage_candidate(
        paper,
        query="bert bidirectional transformer",
        snippets=snippets,
        title_match=True,
        autocomplete_rank=1,
    )

    components = dict(breakdown.components)
    assert components["title_signal"] == pytest.approx(TRIAGE_WEIGHTS["title_signal"] * 1.0)
    assert components["snippet_signal"] > 0.0
    assert breakdown.total > 0.5


def test_combine_weighted_scores_clamps_total_between_zero_and_one() -> None:
    breakdown = combine_weighted_scores(
        {"positive": 10.0, "negative": -5.0},
        {"positive": 1.0, "negative": -0.2},
        reasons=("combined",),
    )

    assert breakdown.total == 1.0
    assert breakdown.reasons == ("combined",)
    assert breakdown.to_dict()["components"]["positive"] == pytest.approx(10.0)
