from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

from .transport_stdlib import StdlibTransport

if TYPE_CHECKING:
    from ..core.client import S2Client, SupportsRequestJson
    from ..engine.models import CitationTraceResult, ExpandReferencesResult, TriageResult


def _import_absolute_runtime():
    from semantic_scholar_skills.core.client import S2Client
    from semantic_scholar_skills.engine.expand_references import expand_references
    from semantic_scholar_skills.engine.paper_triage import paper_triage
    from semantic_scholar_skills.engine.trace_citations import trace_citations

    return S2Client, expand_references, trace_citations, paper_triage


def _import_relative_runtime():
    from ..core.client import S2Client
    from ..engine.expand_references import expand_references
    from ..engine.paper_triage import paper_triage
    from ..engine.trace_citations import trace_citations

    return S2Client, expand_references, trace_citations, paper_triage


def _load_runtime():
    try:
        return _import_absolute_runtime()
    except (ImportError, ModuleNotFoundError) as absolute_exc:
        try:
            return _import_relative_runtime()
        except (ImportError, ModuleNotFoundError) as relative_exc:
            combined = f"{absolute_exc} {relative_exc}".lower()
            if "httpx" in combined:
                raise RuntimeError(
                    "Standalone imports failed because core/client.py still pulls in the httpx-based transport "
                    "at import time. Move the transport imports inside get_default_client() and "
                    "make_compat_client() before using the standalone fallback."
                ) from relative_exc
            raise


def create_client(*, transport: SupportsRequestJson | None = None) -> S2Client:
    s2_client_cls, _, _, _ = _load_runtime()
    return s2_client_cls(transport or StdlibTransport())


async def run_expand_references(
    seeds: Sequence[str],
    *,
    negative_seeds=(),
    api_key_override: str | None = None,
    recommendation_pool: str = "all-cs",
    recommendation_limit: int = 60,
    per_bucket_limit: int = 5,
) -> ExpandReferencesResult:
    _, expand_references, _, _ = _load_runtime()
    client = create_client()
    return await expand_references(
        client,
        seeds,
        negative_seeds=negative_seeds,
        api_key_override=api_key_override,
        recommendation_pool=recommendation_pool,
        recommendation_limit=recommendation_limit,
        per_bucket_limit=per_bucket_limit,
    )


async def run_trace_citations(
    focal_query: str,
    *,
    api_key_override: str | None = None,
    depth: int = 1,
    max_references: int = 50,
    max_citations: int = 50,
    second_hop_limit: int = 10,
) -> CitationTraceResult:
    _, _, trace_citations, _ = _load_runtime()
    client = create_client()
    return await trace_citations(
        client,
        focal_query,
        api_key_override=api_key_override,
        depth=depth,
        max_references=max_references,
        max_citations=max_citations,
        second_hop_limit=second_hop_limit,
    )


async def run_paper_triage(
    query: str,
    *,
    api_key_override: str | None = None,
    shortlist_size: int = 7,
    relevance_limit: int = 10,
    bulk_candidate_limit: int = 20,
    snippet_candidate_limit: int = 5,
    snippet_limit_per_paper: int = 3,
) -> TriageResult:
    _, _, _, paper_triage = _load_runtime()
    client = create_client()
    return await paper_triage(
        client,
        query,
        api_key_override=api_key_override,
        shortlist_size=shortlist_size,
        relevance_limit=relevance_limit,
        bulk_candidate_limit=bulk_candidate_limit,
        snippet_candidate_limit=snippet_candidate_limit,
        snippet_limit_per_paper=snippet_limit_per_paper,
    )


async def run_workflow(
    workflow: Literal["expand-references", "trace-citations", "paper-triage"],
    /,
    **kwargs: Any,
) -> ExpandReferencesResult | CitationTraceResult | TriageResult:
    dispatch = {
        "expand-references": run_expand_references,
        "trace-citations": run_trace_citations,
        "paper-triage": run_paper_triage,
    }
    return await dispatch[workflow](**kwargs)
