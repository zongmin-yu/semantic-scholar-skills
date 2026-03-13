from __future__ import annotations

import pytest

from semantic_scholar_skills.core import (
    PaperAutocompleteRequest,
    PaperBatchDetailsRequest,
    PaperDetailsRequest,
    PaperTitleSearchRequest,
    S2ApiError,
    S2NotFoundError,
)
from semantic_scholar_skills.engine.resolve import (
    RESOLVE_FIELDS,
    detect_query_kind,
    resolve_paper,
    resolve_papers,
)


def test_detect_query_kind_classifies_doi_corpusid_arxiv_ids_and_title() -> None:
    assert detect_query_kind("10.1145/3292500.3330672") == "doi"
    assert detect_query_kind("CorpusId:215416146") == "paper_id"
    assert detect_query_kind("ARXIV:2106.15928") == "paper_id"
    assert detect_query_kind("2106.15928") == "paper_id"
    assert detect_query_kind("hep-ph/9905221") == "paper_id"
    assert detect_query_kind("Attention Is All You Need") == "title"


@pytest.mark.asyncio
async def test_resolve_paper_uses_get_paper_for_doi(stub_s2_client, sample_paper_record) -> None:
    stub_s2_client.queue("get_paper", sample_paper_record)

    resolved = await resolve_paper(stub_s2_client, "10.5555/3295222.3295349")

    assert resolved.source == "direct"
    assert resolved.match_type == "doi"
    assert resolved.confidence == 1.0
    assert stub_s2_client.calls == [
        (
            "get_paper",
            PaperDetailsRequest(
                paper_id="DOI:10.5555/3295222.3295349",
                fields=list(RESOLVE_FIELDS),
            ),
        )
    ]


@pytest.mark.asyncio
async def test_resolve_paper_uses_get_paper_for_corpus_id(stub_s2_client, sample_paper_record) -> None:
    stub_s2_client.queue("get_paper", sample_paper_record)

    resolved = await resolve_paper(stub_s2_client, "CorpusId:215416146")

    assert resolved.source == "direct"
    assert resolved.match_type == "paper_id"
    assert resolved.paper.paper_id == "p-attn"
    assert stub_s2_client.calls == [
        (
            "get_paper",
            PaperDetailsRequest(
                paper_id="CorpusId:215416146",
                fields=list(RESOLVE_FIELDS),
            ),
        )
    ]


@pytest.mark.asyncio
async def test_resolve_paper_strips_doi_url_prefix_before_lookup(stub_s2_client, sample_paper_record) -> None:
    stub_s2_client.queue("get_paper", sample_paper_record)

    await resolve_paper(stub_s2_client, "https://doi.org/10.1145/3292500.3330672")

    [(method_name, request)] = stub_s2_client.calls
    assert method_name == "get_paper"
    assert request == PaperDetailsRequest(
        paper_id="DOI:10.1145/3292500.3330672",
        fields=list(RESOLVE_FIELDS),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_paper_id"),
    [
        ("2106.15928", "ARXIV:2106.15928"),
        ("hep-ph/9905221", "ARXIV:hep-ph/9905221"),
    ],
)
async def test_resolve_paper_normalizes_bare_arxiv_ids_before_direct_lookup(
    stub_s2_client,
    sample_paper_record,
    query,
    expected_paper_id,
) -> None:
    stub_s2_client.queue("get_paper", sample_paper_record)

    await resolve_paper(stub_s2_client, query)

    [(method_name, request)] = stub_s2_client.calls
    assert method_name == "get_paper"
    assert request == PaperDetailsRequest(
        paper_id=expected_paper_id,
        fields=list(RESOLVE_FIELDS),
    )


@pytest.mark.asyncio
async def test_resolve_paper_uses_title_match_before_autocomplete(stub_s2_client, sample_paper_record) -> None:
    stub_s2_client.queue("match_paper_title", {"data": [sample_paper_record]})

    resolved = await resolve_paper(
        stub_s2_client,
        "Attention Is All You Need",
        include_alternatives=False,
    )

    assert resolved.source == "title_match"
    assert resolved.confidence == 0.95
    assert stub_s2_client.calls == [
        (
            "match_paper_title",
            PaperTitleSearchRequest(
                query="Attention Is All You Need",
                fields=list(RESOLVE_FIELDS),
            ),
        )
    ]


@pytest.mark.asyncio
async def test_resolve_paper_returns_primary_title_match_when_autocomplete_enrichment_fails(
    stub_s2_client,
    sample_paper_record,
) -> None:
    stub_s2_client.queue("match_paper_title", {"data": [sample_paper_record]})
    stub_s2_client.queue("autocomplete_papers", S2ApiError(message="autocomplete unavailable"))

    resolved = await resolve_paper(stub_s2_client, "Attention Is All You Need")

    assert resolved.source == "title_match"
    assert resolved.paper.paper_id == "p-attn"
    assert resolved.alternatives == ()
    assert resolved.notes == (
        "Returning primary title match without alternatives because enrichment failed: autocomplete unavailable",
    )


@pytest.mark.asyncio
async def test_resolve_paper_returns_primary_title_match_when_batch_hydration_fails(
    stub_s2_client,
    sample_paper_record,
    sample_autocomplete_payload,
) -> None:
    stub_s2_client.queue("match_paper_title", {"data": [sample_paper_record]})
    stub_s2_client.queue("autocomplete_papers", sample_autocomplete_payload)
    stub_s2_client.queue("batch_papers", S2ApiError(message="batch unavailable"))

    resolved = await resolve_paper(stub_s2_client, "Attention Is All You Need")

    assert resolved.source == "title_match"
    assert resolved.paper.paper_id == "p-attn"
    assert resolved.alternatives == ()
    assert resolved.notes == (
        "Returning primary title match without alternatives because enrichment failed: batch unavailable",
    )


@pytest.mark.asyncio
async def test_resolve_paper_collects_alternatives_from_autocomplete_when_requested(
    stub_s2_client,
    sample_paper_record,
    sample_autocomplete_payload,
) -> None:
    stub_s2_client.queue("match_paper_title", {"data": [sample_paper_record]})
    stub_s2_client.queue("autocomplete_papers", sample_autocomplete_payload)
    stub_s2_client.queue(
        "batch_papers",
        [
            sample_paper_record,
            {
                "paperId": "p-bert",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2019,
                "authors": [{"authorId": "a4", "name": "Jacob Devlin"}],
            },
        ],
    )

    resolved = await resolve_paper(stub_s2_client, "Attention Is All You Need")

    assert resolved.source == "title_match"
    assert tuple(p.paper_id for p in resolved.alternatives) == ("p-bert",)
    assert stub_s2_client.calls == [
        (
            "match_paper_title",
            PaperTitleSearchRequest(
                query="Attention Is All You Need",
                fields=list(RESOLVE_FIELDS),
            ),
        ),
        ("autocomplete_papers", PaperAutocompleteRequest(query="Attention Is All You Need")),
        (
            "batch_papers",
            PaperBatchDetailsRequest(
                paper_ids=["p-attn", "p-bert"],
                fields=",".join(RESOLVE_FIELDS),
            ),
        ),
    ]


@pytest.mark.asyncio
async def test_resolve_paper_falls_back_to_autocomplete_and_batch_hydration_on_title_miss(
    stub_s2_client,
    sample_autocomplete_payload,
) -> None:
    stub_s2_client.queue(
        "match_paper_title",
        S2NotFoundError(message="not found", resource_type="paper", resource_id="Attention Is All You Need"),
    )
    stub_s2_client.queue("autocomplete_papers", sample_autocomplete_payload)
    stub_s2_client.queue(
        "batch_papers",
        [
            None,
            {
                "paperId": "p-bert",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "year": 2019,
                "authors": [{"authorId": "a4", "name": "Jacob Devlin"}],
            },
        ],
    )

    resolved = await resolve_paper(stub_s2_client, "BERT transformer")

    assert resolved.source == "autocomplete"
    assert resolved.confidence == 0.70
    assert resolved.paper.paper_id == "p-bert"
    assert resolved.alternatives == ()
    assert [name for name, _ in stub_s2_client.calls] == [
        "match_paper_title",
        "autocomplete_papers",
        "batch_papers",
    ]


@pytest.mark.asyncio
async def test_resolve_paper_preserves_autocomplete_order_when_batch_hydration_is_out_of_order(
    stub_s2_client,
) -> None:
    stub_s2_client.queue(
        "match_paper_title",
        S2NotFoundError(message="not found", resource_type="paper", resource_id="first suggestion"),
    )
    stub_s2_client.queue(
        "autocomplete_papers",
        {
            "matches": [
                {"id": "p-first", "title": "First suggestion"},
                {"id": "p-second", "title": "Second suggestion"},
            ]
        },
    )
    stub_s2_client.queue(
        "batch_papers",
        [
            {
                "paperId": "p-second",
                "title": "Second suggestion",
                "year": 2020,
                "authors": [{"authorId": "a2", "name": "Author Two"}],
            },
            {
                "paperId": "p-first",
                "title": "First suggestion",
                "year": 2021,
                "authors": [{"authorId": "a1", "name": "Author One"}],
            },
        ],
    )

    resolved = await resolve_paper(stub_s2_client, "first suggestion")

    assert resolved.paper.paper_id == "p-first"
    assert tuple(paper.paper_id for paper in resolved.alternatives) == ("p-second",)


@pytest.mark.asyncio
async def test_resolve_paper_raises_not_found_when_autocomplete_has_no_matches(stub_s2_client) -> None:
    stub_s2_client.queue(
        "match_paper_title",
        S2NotFoundError(message="not found", resource_type="paper", resource_id="unknown"),
    )
    stub_s2_client.queue("autocomplete_papers", {"matches": []})

    with pytest.raises(S2NotFoundError, match="Could not resolve paper query: Unknown paper"):
        await resolve_paper(stub_s2_client, "Unknown paper")


@pytest.mark.asyncio
async def test_resolve_papers_preserves_input_order_and_reuses_cached_resolution_for_duplicate_queries(
    stub_s2_client,
    sample_paper_record,
) -> None:
    bert_record = {
        "paperId": "p-bert",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "year": 2019,
        "authors": [{"authorId": "a4", "name": "Jacob Devlin"}],
    }
    stub_s2_client.queue("get_paper", sample_paper_record)
    stub_s2_client.queue("get_paper", bert_record)

    resolved = await resolve_papers(
        stub_s2_client,
        ["10.5555/3295222.3295349", "CorpusId:215416146", "10.5555/3295222.3295349"],
    )

    assert tuple(item.paper.paper_id for item in resolved) == ("p-attn", "p-bert", "p-attn")
    assert [name for name, _ in stub_s2_client.calls] == ["get_paper", "get_paper"]
