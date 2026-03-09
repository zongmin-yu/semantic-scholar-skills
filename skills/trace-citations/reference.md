# Trace Citations Reference

## Underlying Runtime

- Installed entrypoint: `semantic_scholar_skills.standalone.run_trace_citations`
- Installed dispatcher: `semantic_scholar_skills.standalone.run_workflow`
- Engine workflow: `semantic_scholar_skills.engine.trace_citations.trace_citations`
- Result model: `semantic_scholar_skills.engine.models.CitationTraceResult`

## CLI Arguments

- `focal_query`
  - Title, DOI, or paper ID for the focal paper.
- `--api-key`
  - Optional API key override.
- `--depth`
  - Maps to `depth`.
  - Valid values: `1`, `2`.
- `--max-references`
  - Maps to `max_references`.
- `--max-citations`
  - Maps to `max_citations`.
- `--second-hop-limit`
  - Maps to `second_hop_limit`.

## Result Fields

- `result.focal`
  - The resolved focal paper as `ResolvedPaper`.
- `result.foundations`
  - Strong references behind the focal paper.
- `result.direct_descendants`
  - Strong citations that build on the focal paper.
- `result.bridge_nodes`
  - Mid-confidence edges with useful context or intent signal.
- `result.weak_edges`
  - Low-signal edges with little context and no strong influence markers.
- `result.second_hop`
  - Optional depth-two edges, present only when `depth=2`.
- `result.reference_count_examined`
  - Number of first-hop reference edges scored.
- `result.citation_count_examined`
  - Number of first-hop citation edges scored.
- `result.notes`
  - Execution notes.

## Nested Object Highlights

- Each edge is a `CitationEdge`.
- `CitationEdge.direction` is `reference` or `citation`.
- `CitationEdge.contexts` and `CitationEdge.intents` come from Semantic Scholar citation metadata.
- `CitationEdge.score_breakdown.components` carries the weighted influence, context, intent, and impact signals.
- `CitationEdge.why` is a short explanation list that can be shown directly to a human.
