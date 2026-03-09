# Expand References Reference

## Underlying Runtime

- Installed entrypoint: `semantic_scholar_skills.standalone.run_expand_references`
- Installed dispatcher: `semantic_scholar_skills.standalone.run_workflow`
- Engine workflow: `semantic_scholar_skills.engine.expand_references.expand_references`
- Result model: `semantic_scholar_skills.engine.models.ExpandReferencesResult`

The standalone layer already creates its own `S2Client` with `StdlibTransport`, so the skill launcher only needs to import the standalone module and wrap it with `asyncio.run()`.

## CLI Arguments

- `seeds`
  - One to three seed paper queries.
  - Each seed may be a title, DOI, or paper ID.
- `--negative`
  - Optional repeated negative seed paper query.
- `--api-key`
  - Optional API key override passed through to the workflow.
- `--pool`
  - Maps to `recommendation_pool`.
  - Valid values: `all-cs`, `recent`.
- `--limit`
  - Maps to `recommendation_limit`.
- `--per-bucket-limit`
  - Maps to `per_bucket_limit`.

## Result Fields

- `result.seeds`
  - Resolved positive seeds as `ResolvedPaper` objects.
- `result.negative_seeds`
  - Resolved negative seeds.
- `result.closest_neighbors`
  - Top scored nearby papers regardless of bucket.
- `result.bridge_papers`
  - Candidates that connect multiple seeds.
- `result.foundational`
  - Older, high-impact work that predates the seed set.
- `result.methodological`
  - Method or architecture-oriented papers.
- `result.recent`
  - Recent work near the seed cluster.
- `result.surveys_or_benchmarks`
  - Surveys, reviews, benchmarks, and dataset-style anchors.
- `result.considered_candidates`
  - Count of reranked recommendation candidates.
- `result.notes`
  - Execution notes such as dropped malformed recommendations.

## Nested Object Highlights

- Each item in the scored buckets is a `ScoredPaper`.
- `ScoredPaper.paper` is a normalized `PaperSummary`.
- `ScoredPaper.score_breakdown.components` carries the weighted scoring components.
- `ScoredPaper.why` is the short human-readable explanation list.
- `ScoredPaper.matched_seed_ids` shows which seeds the candidate aligned with.
