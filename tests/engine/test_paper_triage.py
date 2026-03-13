from __future__ import annotations

import pytest

from semantic_scholar_skills.core import (
    PaperAutocompleteRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    SnippetSearchRequest,
)
from semantic_scholar_skills.engine.paper_triage import SNIPPET_FIELDS, TRIAGE_PAPER_FIELDS, paper_triage


@pytest.fixture
def title_match_payload() -> dict[str, object]:
    return {
        "data": [
            {
                "paperId": "p-bert",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2019,
                "citationCount": 90000,
            }
        ]
    }


@pytest.fixture
def autocomplete_payload() -> dict[str, object]:
    return {
        "matches": [
            {"id": "p-bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
            {"id": "p-roberta", "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach"},
        ]
    }


@pytest.fixture
def relevance_search_payload() -> dict[str, object]:
    return {
        "total": 3,
        "offset": 0,
        "data": [
            {
                "paperId": "p-bert",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2019,
                "citationCount": 90000,
            },
            {
                "paperId": "p-roberta",
                "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                "year": 2019,
                "citationCount": 30000,
            },
        ],
    }


@pytest.fixture
def bulk_search_payload() -> dict[str, object]:
    return {
        "token": "next-2",
        "data": [
            {
                "paperId": "p-elmo",
                "title": "Deep Contextualized Word Representations",
                "year": 2018,
                "citationCount": 40000,
            },
            {
                "paperId": "p-albert",
                "title": "ALBERT: A Lite BERT",
                "year": 2020,
                "citationCount": 12000,
            },
        ],
    }


@pytest.fixture
def hydrated_candidates() -> list[dict[str, object]]:
    return [
        {
            "paperId": "p-bert",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "A bidirectional transformer pre-training objective.",
            "year": 2019,
            "citationCount": 90000,
            "influentialCitationCount": 5000,
            "authors": [{"authorId": "a1", "name": "Jacob Devlin"}],
        },
        {
            "paperId": "p-roberta",
            "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
            "abstract": "A robust BERT pretraining approach.",
            "year": 2019,
            "citationCount": 30000,
            "influentialCitationCount": 2000,
            "authors": [{"authorId": "a2", "name": "Yinhan Liu"}],
        },
        {
            "paperId": "p-elmo",
            "title": "Deep Contextualized Word Representations",
            "abstract": "Contextualized word representations for NLP.",
            "year": 2018,
            "citationCount": 40000,
            "influentialCitationCount": 2500,
            "authors": [{"authorId": "a3", "name": "Matthew Peters"}],
        },
        {
            "paperId": "p-albert",
            "title": "ALBERT: A Lite BERT",
            "abstract": "A lightweight variant of BERT.",
            "year": 2020,
            "citationCount": 12000,
            "influentialCitationCount": 900,
            "authors": [{"authorId": "a4", "name": "Zhenzhong Lan"}],
        },
    ]


def queue_triage_basics(
    stub_s2_client,
    *,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
    snippet_payloads: dict[str, dict[str, object]] | None = None,
    include_second_bulk_page: bool = True,
) -> None:
    stub_s2_client.queue("match_paper_title", title_match_payload)
    stub_s2_client.queue("autocomplete_papers", autocomplete_payload)
    stub_s2_client.queue("search_papers", relevance_search_payload)
    stub_s2_client.queue("bulk_search_papers", bulk_search_payload)
    if include_second_bulk_page:
        stub_s2_client.queue("bulk_search_papers", {"data": []})
    stub_s2_client.queue("batch_papers", hydrated_candidates)
    for paper_id in ("p-bert", "p-roberta", "p-elmo", "p-albert"):
        payload = (snippet_payloads or {}).get(paper_id, {"data": []})
        stub_s2_client.queue("search_snippets", payload)


@pytest.mark.asyncio
async def test_paper_triage_uses_title_match_and_autocomplete_for_disambiguation(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
    )

    result = await paper_triage(stub_s2_client, "bert")

    request_types = {type(request) for _, request in stub_s2_client.calls[:4]}
    assert PaperTitleSearchRequest in request_types
    assert PaperAutocompleteRequest in request_types
    assert result.possible_interpretations[0].source == "title_match"
    assert any(item.source == "autocomplete" for item in result.possible_interpretations[1:])


@pytest.mark.asyncio
async def test_paper_triage_uses_bulk_search_to_expand_recall_after_relevance_search(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
    )

    await paper_triage(stub_s2_client, "bert", bulk_candidate_limit=10)

    bulk_requests = [request for name, request in stub_s2_client.calls if name == "bulk_search_papers"]
    assert len(bulk_requests) == 2
    assert isinstance(bulk_requests[0], PaperBulkSearchRequest)
    assert bulk_requests[0].query == "bert"
    assert bulk_requests[1].token == "next-2"


@pytest.mark.asyncio
async def test_paper_triage_hydrates_autocomplete_and_search_candidates_with_batch_papers(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
    )

    result = await paper_triage(stub_s2_client, "bert")

    batch_requests = [request for name, request in stub_s2_client.calls if name == "batch_papers"]
    assert len(batch_requests) == 1
    assert isinstance(batch_requests[0], PaperBatchDetailsRequest)
    assert batch_requests[0].fields == ",".join(TRIAGE_PAPER_FIELDS)
    assert result.shortlist[0].paper.abstract is not None


@pytest.mark.asyncio
async def test_paper_triage_matches_hydrated_records_by_paper_id_when_batch_response_is_out_of_order(
    stub_s2_client,
) -> None:
    stub_s2_client.queue(
        "match_paper_title",
        {
            "data": [
                {
                    "paperId": "p-bert",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "year": 2019,
                    "citationCount": 90000,
                }
            ]
        },
    )
    stub_s2_client.queue(
        "autocomplete_papers",
        {
            "matches": [
                {
                    "id": "p-roberta",
                    "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                }
            ]
        },
    )
    stub_s2_client.queue("search_papers", {"data": []})
    stub_s2_client.queue("bulk_search_papers", {"data": []})
    stub_s2_client.queue(
        "batch_papers",
        [
            {
                "paperId": "p-roberta",
                "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                "abstract": "A robust BERT pretraining approach.",
                "year": 2019,
                "citationCount": 30000,
                "influentialCitationCount": 2000,
                "authors": [{"authorId": "a2", "name": "Yinhan Liu"}],
            },
            {
                "paperId": "p-bert",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": "A bidirectional transformer pre-training objective.",
                "year": 2019,
                "citationCount": 90000,
                "influentialCitationCount": 5000,
                "authors": [{"authorId": "a1", "name": "Jacob Devlin"}],
            },
        ],
    )

    result = await paper_triage(stub_s2_client, "bert bidirectional transformer", snippet_candidate_limit=0)

    assert result.possible_interpretations[0].paper.paper_id == "p-bert"
    assert result.possible_interpretations[1].paper.paper_id == "p-roberta"
    assert result.shortlist[0].paper.paper_id == "p-bert"


@pytest.mark.asyncio
async def test_paper_triage_runs_snippet_search_only_for_top_preliminary_candidates(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
    )

    await paper_triage(stub_s2_client, "bert", snippet_candidate_limit=2)

    snippet_requests = [request for name, request in stub_s2_client.calls if name == "search_snippets"]
    assert len(snippet_requests) == 2
    assert all(isinstance(request, SnippetSearchRequest) for request in snippet_requests)


@pytest.mark.asyncio
async def test_paper_triage_extracts_snippet_text_from_nested_snippet_payloads(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
    sample_snippet_payload_for_candidate,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
        snippet_payloads={"p-bert": sample_snippet_payload_for_candidate},
    )

    result = await paper_triage(
        stub_s2_client,
        "bert bidirectional transformer",
        snippet_candidate_limit=1,
    )

    assert result.shortlist[0].snippet_evidence[0].text.startswith("This paper introduces")
    assert result.shortlist[0].snippet_evidence[0].paper_title == "BERT: Pre-training of Deep Bidirectional Transformers"


@pytest.mark.asyncio
async def test_paper_triage_final_ranking_rewards_title_signal_snippet_signal_impact_and_recency(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
    sample_snippet_payload_for_candidate,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
        snippet_payloads={"p-bert": sample_snippet_payload_for_candidate},
    )

    result = await paper_triage(
        stub_s2_client,
        "bert bidirectional transformer",
        snippet_candidate_limit=1,
    )

    assert result.shortlist[0].paper.paper_id == "p-bert"
    assert result.shortlist[0].score >= result.shortlist[-1].score
    assert "exact title match" in result.shortlist[0].why


@pytest.mark.asyncio
async def test_paper_triage_includes_follow_up_actions_in_result(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
    )

    result = await paper_triage(stub_s2_client, "bert")

    assert result.follow_up_actions == ("trace-citations", "expand-references")


@pytest.mark.asyncio
async def test_paper_triage_returns_interpretations_even_when_shortlist_is_empty(
    stub_s2_client,
    title_match_payload,
    autocomplete_payload,
    relevance_search_payload,
    bulk_search_payload,
    hydrated_candidates,
) -> None:
    queue_triage_basics(
        stub_s2_client,
        title_match_payload=title_match_payload,
        autocomplete_payload=autocomplete_payload,
        relevance_search_payload=relevance_search_payload,
        bulk_search_payload=bulk_search_payload,
        hydrated_candidates=hydrated_candidates,
    )

    result = await paper_triage(stub_s2_client, "bert", shortlist_size=0)

    assert result.shortlist == ()
    assert result.possible_interpretations
