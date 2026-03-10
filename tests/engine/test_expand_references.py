from __future__ import annotations

import pytest

from semantic_scholar_skills.core import PaperBatchDetailsRequest, PaperRecommendationsMultiRequest, S2ValidationError
from semantic_scholar_skills.engine.expand_references import RECOMMENDATION_FIELDS_CSV, expand_references
from semantic_scholar_skills.engine.scoring import CURRENT_YEAR


@pytest.fixture
def resolved_positive_seed_records() -> list[dict[str, object]]:
    return [
        {
            "paperId": "seed-a",
            "title": "Dense Passage Retrieval",
            "year": 2017,
            "authors": [{"authorId": "a1", "name": "Alice Seed"}],
            "fieldsOfStudy": ["Computer Science", "Information Retrieval"],
            "venue": "ACL",
        },
        {
            "paperId": "seed-b",
            "title": "Retrieval-Augmented Language Models",
            "year": 2019,
            "authors": [{"authorId": "a2", "name": "Bob Seed"}],
            "fieldsOfStudy": ["Computer Science", "Natural Language Processing"],
            "venue": "NeurIPS",
        },
    ]


@pytest.fixture
def resolved_negative_seed_record() -> dict[str, object]:
    return {
        "paperId": "seed-neg",
        "title": "Biomedical Entity Linking",
        "year": 2020,
        "authors": [{"authorId": "a3", "name": "Nina Negative"}],
        "fieldsOfStudy": ["Biology"],
        "venue": "BioNLP",
    }


@pytest.fixture
def hydrated_recommendation_batch() -> list[dict[str, object]]:
    return [
        {
            "paperId": "p-foundation",
            "title": "Distributed Representations for Retrieval",
            "abstract": "A seminal retrieval formulation.",
            "year": 2013,
            "venue": "ACL",
            "authors": [{"authorId": "f1", "name": "Found Author"}],
            "citationCount": 9000,
            "influentialCitationCount": 1400,
            "fieldsOfStudy": ["Computer Science", "Information Retrieval"],
            "publicationTypes": ["Conference"],
        },
        {
            "paperId": "p-recent",
            "title": "Adaptive Retrieval for Long-Context Language Models",
            "abstract": "Recent retrieval modeling paper.",
            "year": CURRENT_YEAR,
            "venue": "ICLR",
            "authors": [{"authorId": "r1", "name": "Recent Author"}],
            "citationCount": 600,
            "influentialCitationCount": 80,
            "fieldsOfStudy": ["Computer Science", "Natural Language Processing"],
            "publicationTypes": ["Conference"],
        },
        {
            "paperId": "p-method",
            "title": "A Retrieval Framework for Efficient Training",
            "abstract": "This method proposes a new training architecture.",
            "year": 2018,
            "venue": "EMNLP",
            "authors": [{"authorId": "m1", "name": "Method Author"}],
            "citationCount": 1200,
            "influentialCitationCount": 120,
            "fieldsOfStudy": ["Computer Science", "Natural Language Processing"],
            "publicationTypes": ["Conference"],
        },
        {
            "paperId": "p-survey",
            "title": "A Survey of Retrieval-Augmented Generation",
            "abstract": "Survey and benchmark coverage for retrieval-augmented models.",
            "year": 2021,
            "venue": "TACL",
            "authors": [{"authorId": "s1", "name": "Survey Author"}],
            "citationCount": 1500,
            "influentialCitationCount": 150,
            "fieldsOfStudy": ["Computer Science", "Natural Language Processing"],
            "publicationTypes": ["Review"],
        },
        {
            "paperId": "p-bridge",
            "title": "Connecting Evidence and Search Tasks",
            "abstract": "Connects question answering and search across tasks.",
            "year": 2016,
            "venue": "NAACL",
            "authors": [{"authorId": "b1", "name": "Bridge Author"}],
            "citationCount": 4000,
            "influentialCitationCount": 400,
            "fieldsOfStudy": ["Computer Science", "Information Retrieval", "Natural Language Processing"],
            "publicationTypes": ["Conference"],
        },
    ]


def queue_seed_resolution(stub_s2_client, *records: dict[str, object]) -> None:
    for record in records:
        stub_s2_client.queue("match_paper_title", record)


@pytest.mark.asyncio
async def test_expand_references_resolves_positive_and_negative_seeds_and_calls_multi_recommendations(
    stub_s2_client,
    resolved_positive_seed_records,
    resolved_negative_seed_record,
    sample_recommendations_payload,
    hydrated_recommendation_batch,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records, resolved_negative_seed_record)
    stub_s2_client.queue("recommend_from_papers", sample_recommendations_payload)
    stub_s2_client.queue("batch_papers", hydrated_recommendation_batch[:2])

    await expand_references(
        stub_s2_client,
        ["seed one", "seed two"],
        negative_seeds=["negative seed"],
    )

    recommend_calls = [request for name, request in stub_s2_client.calls if name == "recommend_from_papers"]
    assert len(recommend_calls) == 1
    assert recommend_calls[0] == PaperRecommendationsMultiRequest(
        positive_paper_ids=["seed-a", "seed-b"],
        negative_paper_ids=["seed-neg"],
        fields=RECOMMENDATION_FIELDS_CSV,
        limit=60,
    )


@pytest.mark.asyncio
async def test_expand_references_filters_out_positive_and_negative_seed_ids_from_candidates(
    stub_s2_client,
    resolved_positive_seed_records,
    resolved_negative_seed_record,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records, resolved_negative_seed_record)
    stub_s2_client.queue(
        "recommend_from_papers",
        {
            "recommendedPapers": [
                {"paperId": "seed-a", "title": "Dense Passage Retrieval"},
                {"paperId": "seed-neg", "title": "Biomedical Entity Linking"},
                {"paperId": "p-keep", "title": "A Kept Candidate"},
            ]
        },
    )
    stub_s2_client.queue(
        "batch_papers",
        [
            {
                "paperId": "p-keep",
                "title": "A Kept Candidate",
                "year": 2022,
                "authors": [{"authorId": "k1", "name": "Keep Author"}],
                "fieldsOfStudy": ["Computer Science"],
            }
        ],
    )

    result = await expand_references(
        stub_s2_client,
        ["seed one", "seed two"],
        negative_seeds=["negative seed"],
    )

    assert [paper.paper.paper_id for paper in result.closest_neighbors] == ["p-keep"]


@pytest.mark.asyncio
async def test_expand_references_hydrates_recommendation_candidates_with_batch_papers(
    stub_s2_client,
    resolved_positive_seed_records,
    sample_recommendations_payload,
    hydrated_recommendation_batch,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records)
    stub_s2_client.queue("recommend_from_papers", sample_recommendations_payload)
    stub_s2_client.queue("batch_papers", hydrated_recommendation_batch[:2])

    await expand_references(stub_s2_client, ["seed one", "seed two"])

    batch_calls = [request for name, request in stub_s2_client.calls if name == "batch_papers"]
    assert batch_calls == [
        PaperBatchDetailsRequest(
            paper_ids=["p-gpt", "p-bert"],
            fields=RECOMMENDATION_FIELDS_CSV,
        )
    ]


@pytest.mark.asyncio
async def test_expand_references_matches_hydrated_candidates_by_paper_id_when_batch_response_is_out_of_order(
    stub_s2_client,
    resolved_positive_seed_records,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records)
    stub_s2_client.queue(
        "recommend_from_papers",
        {
            "recommendedPapers": [
                {"paperId": "p-strong", "title": "Strong Candidate"},
                {"paperId": "p-weak", "title": "Weak Candidate"},
            ]
        },
    )
    stub_s2_client.queue(
        "batch_papers",
        [
            {
                "paperId": "p-weak",
                "title": "Weak Candidate",
                "abstract": "Weak candidate abstract.",
                "year": 2022,
                "venue": "Workshop",
                "authors": [{"authorId": "w1", "name": "Weak Author"}],
                "citationCount": 5,
                "influentialCitationCount": 0,
                "fieldsOfStudy": ["Computer Science"],
                "publicationTypes": ["Conference"],
            },
            {
                "paperId": "p-strong",
                "title": "Strong Candidate",
                "abstract": "Strong candidate abstract.",
                "year": 2018,
                "venue": "ACL",
                "authors": [{"authorId": "s1", "name": "Strong Author"}],
                "citationCount": 5000,
                "influentialCitationCount": 400,
                "fieldsOfStudy": ["Computer Science", "Information Retrieval"],
                "publicationTypes": ["Conference"],
            },
        ],
    )

    result = await expand_references(stub_s2_client, ["seed one", "seed two"])

    assert result.closest_neighbors[0].paper.paper_id == "p-strong"


@pytest.mark.asyncio
async def test_expand_references_assigns_foundational_recent_methodological_and_survey_categories(
    stub_s2_client,
    resolved_positive_seed_records,
    hydrated_recommendation_batch,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records)
    stub_s2_client.queue(
        "recommend_from_papers",
        {"recommendedPapers": [{"paperId": record["paperId"], "title": record["title"]} for record in hydrated_recommendation_batch]},
    )
    stub_s2_client.queue("batch_papers", hydrated_recommendation_batch)

    result = await expand_references(stub_s2_client, ["seed one", "seed two"])

    assert [paper.paper.paper_id for paper in result.foundational] == ["p-foundation"]
    assert [paper.paper.paper_id for paper in result.recent] == ["p-recent"]
    assert [paper.paper.paper_id for paper in result.methodological] == ["p-method"]
    assert [paper.paper.paper_id for paper in result.surveys_or_benchmarks] == ["p-survey"]


@pytest.mark.asyncio
async def test_expand_references_builds_bridge_bucket_for_cross_seed_connector(
    stub_s2_client,
    resolved_positive_seed_records,
    hydrated_recommendation_batch,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records)
    stub_s2_client.queue(
        "recommend_from_papers",
        {"recommendedPapers": [{"paperId": record["paperId"], "title": record["title"]} for record in hydrated_recommendation_batch]},
    )
    stub_s2_client.queue("batch_papers", hydrated_recommendation_batch)

    result = await expand_references(stub_s2_client, ["seed one", "seed two"])

    assert "p-bridge" in [paper.paper.paper_id for paper in result.bridge_papers]


@pytest.mark.asyncio
async def test_expand_references_rejects_duplicate_positive_seed_resolutions(
    stub_s2_client,
    resolved_positive_seed_records,
) -> None:
    duplicate = dict(resolved_positive_seed_records[0])
    queue_seed_resolution(stub_s2_client, resolved_positive_seed_records[0], duplicate)

    with pytest.raises(S2ValidationError, match="Resolved seed papers must be unique") as exc_info:
        await expand_references(stub_s2_client, ["seed one", "seed duplicate"])

    assert exc_info.value.field == "seeds"


@pytest.mark.asyncio
async def test_expand_references_returns_empty_buckets_when_recommendations_are_empty(
    stub_s2_client,
    resolved_positive_seed_records,
) -> None:
    queue_seed_resolution(stub_s2_client, *resolved_positive_seed_records)
    stub_s2_client.queue("recommend_from_papers", {"recommendedPapers": []})

    result = await expand_references(stub_s2_client, ["seed one", "seed two"])

    assert result.closest_neighbors == ()
    assert result.bridge_papers == ()
    assert result.considered_candidates == 0
