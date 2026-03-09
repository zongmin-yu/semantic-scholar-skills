from __future__ import annotations

from semantic_scholar_skills.engine.models import (
    AuthorSummary,
    CitationEdge,
    PaperSummary,
    ResolvedPaper,
    ScoreBreakdown,
    SnippetEvidence,
    TriageResult,
)


def test_author_summary_from_api_response_tolerates_missing_keys() -> None:
    author = AuthorSummary.from_api_response({"name": "Ashish Vaswani"})

    assert author.author_id is None
    assert author.name == "Ashish Vaswani"
    assert author.url is None
    assert author.affiliations == ()
    assert author.to_dict() == {
        "author_id": None,
        "name": "Ashish Vaswani",
        "url": None,
        "affiliations": [],
    }


def test_paper_summary_from_api_response_synthesizes_corpus_identifier_and_recursively_serializes() -> None:
    paper = PaperSummary.from_api_response(
        {
            "corpusId": 123,
            "title": "Corpus-only paper",
            "authors": [{"authorId": "a1", "name": "Author One"}],
            "externalIds": {"DOI": "10.1/example"},
        }
    )

    assert paper.paper_id == "CorpusId:123"
    assert paper.author_names() == ("Author One",)
    assert paper.to_dict() == {
        "paper_id": "CorpusId:123",
        "title": "Corpus-only paper",
        "abstract": None,
        "year": None,
        "venue": None,
        "url": None,
        "authors": [
            {"author_id": "a1", "name": "Author One", "url": None, "affiliations": []}
        ],
        "citation_count": None,
        "influential_citation_count": None,
        "reference_count": None,
        "fields_of_study": [],
        "publication_types": [],
        "external_ids": [["DOI", "10.1/example"]],
        "corpus_id": 123,
        "publication_date": None,
        "tldr": None,
    }


def test_citation_edge_from_api_response_supports_nested_payload_shapes() -> None:
    edge = CitationEdge.from_api_response(
        {
            "contexts": ["Builds on prior retrieval work."],
            "intents": ["Extends"],
            "isInfluential": True,
            "citedPaper": {
                "paperId": "p-rag",
                "title": "Retrieval-Augmented Generation",
                "authors": [{"authorId": "a3", "name": "Patrick Lewis"}],
            },
        },
        direction="reference",
        depth=2,
    )

    assert edge.direction == "reference"
    assert edge.paper.paper_id == "p-rag"
    assert edge.contexts == ("Builds on prior retrieval work.",)
    assert edge.intents == ("Extends",)
    assert edge.is_influential is True
    assert edge.depth == 2


def test_snippet_evidence_and_nested_result_to_dict_are_json_compatible(sample_paper_record) -> None:
    paper = PaperSummary.from_api_response(sample_paper_record)
    resolved = ResolvedPaper(
        query="Attention Is All You Need",
        normalized_query="Attention Is All You Need",
        match_type="title",
        source="title_match",
        confidence=0.95,
        paper=paper,
    )
    result = TriageResult(
        query="attention is all you need",
        normalized_query="attention is all you need",
        possible_interpretations=(resolved,),
        notes=("resolved from title match",),
    )
    snippet = SnippetEvidence.from_api_response(
        {
            "snippet": {"text": "A transformer pre-training objective."},
            "paper": {"paperId": "p-bert", "title": "BERT"},
        }
    )

    assert snippet.to_dict() == {
        "text": "A transformer pre-training objective.",
        "paper_id": "p-bert",
        "paper_title": "BERT",
        "field_path": "snippet.text",
    }
    assert result.to_dict()["possible_interpretations"][0]["paper"]["paper_id"] == "p-attn"
    assert result.to_dict()["notes"] == ["resolved from title match"]


def test_score_breakdown_component_dict_and_to_dict_preserve_components() -> None:
    breakdown = ScoreBreakdown(
        total=0.6,
        components=(("impact", 0.25), ("recency", 0.35)),
        reasons=("high citation impact",),
    )

    assert breakdown.component_dict() == {"impact": 0.25, "recency": 0.35}
    assert breakdown.to_dict() == {
        "total": 0.6,
        "components": {"impact": 0.25, "recency": 0.35},
        "reasons": ["high citation impact"],
    }
