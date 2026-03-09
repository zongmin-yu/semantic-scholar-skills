from .expand_references import expand_references
from .models import (
    AuthorSummary,
    CitationEdge,
    CitationTraceResult,
    ExpandReferencesResult,
    PaperSummary,
    ResolvedPaper,
    ScoreBreakdown,
    ScoredPaper,
    SnippetEvidence,
    TriageCandidate,
    TriageResult,
)
from .paper_triage import paper_triage
from .resolve import detect_query_kind, normalize_paper_query, resolve_paper, resolve_papers
from .trace_citations import trace_citations

__all__ = [
    "AuthorSummary",
    "PaperSummary",
    "ResolvedPaper",
    "ScoreBreakdown",
    "ScoredPaper",
    "SnippetEvidence",
    "TriageCandidate",
    "TriageResult",
    "CitationEdge",
    "CitationTraceResult",
    "ExpandReferencesResult",
    "detect_query_kind",
    "normalize_paper_query",
    "resolve_paper",
    "resolve_papers",
    "expand_references",
    "trace_citations",
    "paper_triage",
]
