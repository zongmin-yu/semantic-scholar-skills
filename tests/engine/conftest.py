from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class StubS2Client:
    responses: dict[str, deque[Any]] = field(default_factory=dict)
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def queue(self, method_name: str, *items: Any) -> None:
        self.responses.setdefault(method_name, deque()).extend(items)

    def _pop(self, method_name: str) -> Any:
        queue = self.responses.get(method_name)
        if not queue:
            raise AssertionError(f"No queued response for {method_name}")
        item = queue.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    async def get_paper(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("get_paper", request))
        return self._pop("get_paper")

    async def match_paper_title(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("match_paper_title", request))
        return self._pop("match_paper_title")

    async def autocomplete_papers(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("autocomplete_papers", request))
        return self._pop("autocomplete_papers")

    async def batch_papers(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("batch_papers", request))
        return self._pop("batch_papers")

    async def search_papers(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("search_papers", request))
        return self._pop("search_papers")

    async def bulk_search_papers(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("bulk_search_papers", request))
        return self._pop("bulk_search_papers")

    async def search_snippets(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("search_snippets", request))
        return self._pop("search_snippets")

    async def get_paper_references(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("get_paper_references", request))
        return self._pop("get_paper_references")

    async def get_paper_citations(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("get_paper_citations", request))
        return self._pop("get_paper_citations")

    async def recommend_from_papers(self, request, *, api_key_override: str | None = None) -> Any:
        self.calls.append(("recommend_from_papers", request))
        return self._pop("recommend_from_papers")


@pytest.fixture
def stub_client() -> StubS2Client:
    return StubS2Client()


@pytest.fixture
def stub_s2_client(stub_client: StubS2Client) -> StubS2Client:
    return stub_client


@pytest.fixture
def sample_paper_record() -> dict[str, Any]:
    return {
        "paperId": "p-attn",
        "corpusId": 215416146,
        "externalIds": {"DOI": "10.5555/3295222.3295349"},
        "title": "Attention Is All You Need",
        "abstract": "Transformer sequence modeling paper.",
        "year": 2017,
        "venue": "NeurIPS",
        "url": "https://www.semanticscholar.org/paper/p-attn",
        "authors": [{"authorId": "a1", "name": "Ashish Vaswani"}],
        "citationCount": 12000,
        "influentialCitationCount": 3500,
        "referenceCount": 38,
        "fieldsOfStudy": ["Computer Science"],
        "publicationTypes": ["Conference"],
    }


@pytest.fixture
def sample_autocomplete_payload() -> dict[str, Any]:
    return {
        "matches": [
            {"id": "p-attn", "title": "Attention Is All You Need"},
            {"id": "p-bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
        ]
    }


@pytest.fixture
def sample_recommendations_payload() -> dict[str, Any]:
    return {
        "recommendedPapers": [
            {"paperId": "p-gpt", "title": "Improving Language Understanding by Generative Pre-Training"},
            {"paperId": "p-bert", "title": "BERT: Pre-training of Deep Bidirectional Transformers"},
        ]
    }


@pytest.fixture
def sample_flat_edge_payload() -> dict[str, Any]:
    return {
        "offset": 0,
        "next": 2,
        "data": [
            {
                "paperId": "p-ref-1",
                "title": "Sequence to Sequence Learning with Neural Networks",
                "year": 2014,
                "citationCount": 9000,
                "influentialCitationCount": 1400,
                "contexts": ["We follow the encoder-decoder setup."],
                "intents": ["Background"],
                "isInfluential": True,
                "authors": [{"authorId": "a2", "name": "Ilya Sutskever"}],
                "venue": "NeurIPS",
            }
        ],
    }


@pytest.fixture
def sample_nested_edge_payload() -> dict[str, Any]:
    return {
        "offset": 0,
        "next": 1,
        "data": [
            {
                "contexts": ["The proposed method extends BERT with retrieval."],
                "intents": ["Extends"],
                "isInfluential": True,
                "citingPaper": {
                    "paperId": "p-rag",
                    "title": "Retrieval-Augmented Generation",
                    "year": 2020,
                    "citationCount": 1800,
                    "influentialCitationCount": 210,
                    "authors": [{"authorId": "a3", "name": "Patrick Lewis"}],
                    "venue": "NeurIPS",
                },
            }
        ],
    }


@pytest.fixture
def sample_snippet_payload_for_candidate() -> dict[str, Any]:
    return {
        "data": [
            {
                "snippet": {
                    "text": "This paper introduces a bidirectional transformer pre-training objective."
                },
                "paper": {"title": "BERT: Pre-training of Deep Bidirectional Transformers"},
            }
        ]
    }
