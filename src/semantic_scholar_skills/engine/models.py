from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, Mapping


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if item is not None and str(item))


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


@dataclass(slots=True, frozen=True)
class AuthorSummary:
    """Immutable author projection used by all engine outputs."""

    author_id: str | None = None
    name: str | None = None
    url: str | None = None
    affiliations: tuple[str, ...] = ()

    @classmethod
    def from_api_response(cls, data: Mapping[str, Any] | None) -> AuthorSummary:
        payload = data or {}
        return cls(
            author_id=payload.get("authorId"),
            name=payload.get("name"),
            url=payload.get("url"),
            affiliations=_tuple_of_strings(payload.get("affiliations")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "author_id": self.author_id,
            "name": self.name,
            "url": self.url,
            "affiliations": list(self.affiliations),
        }


@dataclass(slots=True, frozen=True)
class PaperSummary:
    """Immutable normalized paper payload built from paper, citation, reference, search, and recommendation responses."""

    paper_id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    authors: tuple[AuthorSummary, ...] = ()
    citation_count: int | None = None
    influential_citation_count: int | None = None
    reference_count: int | None = None
    fields_of_study: tuple[str, ...] = ()
    publication_types: tuple[str, ...] = ()
    external_ids: tuple[tuple[str, str], ...] = ()
    corpus_id: int | None = None
    publication_date: str | None = None
    tldr: str | None = None

    @classmethod
    def from_api_response(cls, data: Mapping[str, Any] | None) -> PaperSummary:
        payload = data or {}
        corpus_id = payload.get("corpusId")
        paper_id = payload.get("paperId") or (f"CorpusId:{corpus_id}" if corpus_id is not None else "")
        authors = payload.get("authors")
        raw_external_ids = payload.get("externalIds")
        external_ids: tuple[tuple[str, str], ...] = ()
        if isinstance(raw_external_ids, Mapping):
            external_ids = tuple(
                (str(key), str(value))
                for key, value in raw_external_ids.items()
                if value is not None and str(value)
            )
        raw_tldr = payload.get("tldr")
        if isinstance(raw_tldr, Mapping):
            tldr = raw_tldr.get("text")
        else:
            tldr = raw_tldr
        return cls(
            paper_id=str(paper_id),
            title=str(payload.get("title") or ""),
            abstract=payload.get("abstract"),
            year=payload.get("year"),
            venue=payload.get("venue"),
            url=payload.get("url"),
            authors=tuple(
                AuthorSummary.from_api_response(author)
                for author in authors
                if isinstance(author, Mapping)
            )
            if isinstance(authors, (list, tuple))
            else (),
            citation_count=payload.get("citationCount"),
            influential_citation_count=payload.get("influentialCitationCount"),
            reference_count=payload.get("referenceCount"),
            fields_of_study=_tuple_of_strings(payload.get("fieldsOfStudy")),
            publication_types=_tuple_of_strings(payload.get("publicationTypes")),
            external_ids=external_ids,
            corpus_id=corpus_id,
            publication_date=payload.get("publicationDate"),
            tldr=tldr,
        )

    def author_names(self) -> tuple[str, ...]:
        return tuple(author.name for author in self.authors if author.name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "authors": [_serialize(author) for author in self.authors],
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "reference_count": self.reference_count,
            "fields_of_study": list(self.fields_of_study),
            "publication_types": list(self.publication_types),
            "external_ids": [[key, value] for key, value in self.external_ids],
            "corpus_id": self.corpus_id,
            "publication_date": self.publication_date,
            "tldr": self.tldr,
        }


@dataclass(slots=True, frozen=True)
class ResolvedPaper:
    """A user query resolved to a concrete paper plus provenance and fallback candidates."""

    query: str
    normalized_query: str
    match_type: Literal["doi", "paper_id", "title"]
    source: Literal["direct", "title_match", "autocomplete"]
    confidence: float
    paper: PaperSummary
    alternatives: tuple[PaperSummary, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "match_type": self.match_type,
            "source": self.source,
            "confidence": self.confidence,
            "paper": _serialize(self.paper),
            "alternatives": [_serialize(item) for item in self.alternatives],
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class ScoreBreakdown:
    """Named score components plus normalized total in the range [0.0, 1.0]."""

    total: float
    components: tuple[tuple[str, float], ...] = ()
    reasons: tuple[str, ...] = ()

    def component_dict(self) -> dict[str, float]:
        return dict(self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "components": self.component_dict(),
            "reasons": list(self.reasons),
        }


@dataclass(slots=True, frozen=True)
class ScoredPaper:
    """Paper plus workflow-specific score, ranking metadata, and human-readable explanation hooks."""

    paper: PaperSummary
    score: float
    rank: int | None = None
    score_breakdown: ScoreBreakdown | None = None
    primary_category: str | None = None
    matched_seed_ids: tuple[str, ...] = ()
    why: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper": _serialize(self.paper),
            "score": self.score,
            "rank": self.rank,
            "score_breakdown": _serialize(self.score_breakdown),
            "primary_category": self.primary_category,
            "matched_seed_ids": list(self.matched_seed_ids),
            "why": list(self.why),
        }


@dataclass(slots=True, frozen=True)
class SnippetEvidence:
    """A snippet-search hit attributed to exactly one candidate paper."""

    text: str
    paper_id: str | None = None
    paper_title: str | None = None
    field_path: str = "snippet.text"

    @classmethod
    def from_api_response(
        cls,
        data: Mapping[str, Any] | None,
        *,
        fallback_paper_id: str | None = None,
        fallback_paper_title: str | None = None,
    ) -> SnippetEvidence:
        payload = data or {}
        snippet = payload.get("snippet")
        paper = payload.get("paper")
        text = ""
        if isinstance(snippet, Mapping):
            text = str(snippet.get("text") or "")
        elif isinstance(payload.get("snippet.text"), str):
            text = str(payload.get("snippet.text"))
        paper_id = fallback_paper_id
        paper_title = fallback_paper_title
        if isinstance(paper, Mapping):
            paper_id = paper.get("paperId") or fallback_paper_id
            paper_title = paper.get("title") or fallback_paper_title
        return cls(
            text=text,
            paper_id=paper_id,
            paper_title=paper_title,
            field_path="snippet.text",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "field_path": self.field_path,
        }


@dataclass(slots=True, frozen=True)
class TriageCandidate:
    """One shortlisted paper in paper triage, including evidence used to rank it."""

    paper: PaperSummary
    score: float
    rank: int | None = None
    score_breakdown: ScoreBreakdown | None = None
    autocomplete_rank: int | None = None
    title_match: bool = False
    snippet_evidence: tuple[SnippetEvidence, ...] = ()
    why: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper": _serialize(self.paper),
            "score": self.score,
            "rank": self.rank,
            "score_breakdown": _serialize(self.score_breakdown),
            "autocomplete_rank": self.autocomplete_rank,
            "title_match": self.title_match,
            "snippet_evidence": [_serialize(item) for item in self.snippet_evidence],
            "why": list(self.why),
        }


@dataclass(slots=True, frozen=True)
class TriageResult:
    """Top-level output of paper_triage()."""

    query: str
    normalized_query: str
    possible_interpretations: tuple[ResolvedPaper, ...] = ()
    shortlist: tuple[TriageCandidate, ...] = ()
    follow_up_actions: tuple[str, ...] = ("trace-citations", "expand-references")
    considered_candidates: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "possible_interpretations": [_serialize(item) for item in self.possible_interpretations],
            "shortlist": [_serialize(item) for item in self.shortlist],
            "follow_up_actions": list(self.follow_up_actions),
            "considered_candidates": self.considered_candidates,
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class CitationEdge:
    """A single citation/reference relationship around a focal paper."""

    direction: Literal["reference", "citation"]
    paper: PaperSummary
    score: float = 0.0
    score_breakdown: ScoreBreakdown | None = None
    contexts: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    is_influential: bool | None = None
    depth: int = 1
    why: tuple[str, ...] = ()

    @classmethod
    def from_api_response(
        cls,
        data: Mapping[str, Any],
        *,
        direction: Literal["reference", "citation"],
        depth: int = 1,
    ) -> CitationEdge:
        paper_payload = (
            data.get("paper")
            or data.get("citingPaper")
            or data.get("citedPaper")
            or data
        )
        mapping = paper_payload if isinstance(paper_payload, Mapping) else {}
        return cls(
            direction=direction,
            paper=PaperSummary.from_api_response(mapping),
            contexts=tuple(ctx.strip() for ctx in data.get("contexts", []) if ctx and str(ctx).strip()),
            intents=tuple(str(intent) for intent in data.get("intents", []) if intent),
            is_influential=data.get("isInfluential"),
            depth=depth,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "paper": _serialize(self.paper),
            "score": self.score,
            "score_breakdown": _serialize(self.score_breakdown),
            "contexts": list(self.contexts),
            "intents": list(self.intents),
            "is_influential": self.is_influential,
            "depth": self.depth,
            "why": list(self.why),
        }


@dataclass(slots=True, frozen=True)
class CitationTraceResult:
    """Top-level output of trace_citations()."""

    focal: ResolvedPaper
    foundations: tuple[CitationEdge, ...] = ()
    direct_descendants: tuple[CitationEdge, ...] = ()
    bridge_nodes: tuple[CitationEdge, ...] = ()
    weak_edges: tuple[CitationEdge, ...] = ()
    second_hop: tuple[CitationEdge, ...] = ()
    reference_count_examined: int = 0
    citation_count_examined: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal": _serialize(self.focal),
            "foundations": [_serialize(item) for item in self.foundations],
            "direct_descendants": [_serialize(item) for item in self.direct_descendants],
            "bridge_nodes": [_serialize(item) for item in self.bridge_nodes],
            "weak_edges": [_serialize(item) for item in self.weak_edges],
            "second_hop": [_serialize(item) for item in self.second_hop],
            "reference_count_examined": self.reference_count_examined,
            "citation_count_examined": self.citation_count_examined,
            "notes": list(self.notes),
        }


@dataclass(slots=True, frozen=True)
class ExpandReferencesResult:
    """Top-level output of expand_references()."""

    seeds: tuple[ResolvedPaper, ...]
    negative_seeds: tuple[ResolvedPaper, ...] = ()
    closest_neighbors: tuple[ScoredPaper, ...] = ()
    bridge_papers: tuple[ScoredPaper, ...] = ()
    foundational: tuple[ScoredPaper, ...] = ()
    methodological: tuple[ScoredPaper, ...] = ()
    recent: tuple[ScoredPaper, ...] = ()
    surveys_or_benchmarks: tuple[ScoredPaper, ...] = ()
    considered_candidates: int = 0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": [_serialize(item) for item in self.seeds],
            "negative_seeds": [_serialize(item) for item in self.negative_seeds],
            "closest_neighbors": [_serialize(item) for item in self.closest_neighbors],
            "bridge_papers": [_serialize(item) for item in self.bridge_papers],
            "foundational": [_serialize(item) for item in self.foundational],
            "methodological": [_serialize(item) for item in self.methodological],
            "recent": [_serialize(item) for item in self.recent],
            "surveys_or_benchmarks": [_serialize(item) for item in self.surveys_or_benchmarks],
            "considered_candidates": self.considered_candidates,
            "notes": list(self.notes),
        }
