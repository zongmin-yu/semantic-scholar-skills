# Paper Triage Reference

## Underlying Runtime

- Installed entrypoint: `semantic_scholar_skills.standalone.run_paper_triage`
- Installed dispatcher: `semantic_scholar_skills.standalone.run_workflow`
- Engine workflow: `semantic_scholar_skills.engine.paper_triage.paper_triage`
- Result model: `semantic_scholar_skills.engine.models.TriageResult`

## CLI Arguments

- `query`
  - The title fragment or paper query to triage.
- `--api-key`
  - Optional API key override.
- `--shortlist-size`
  - Maps to `shortlist_size`.
- `--relevance-limit`
  - Maps to `relevance_limit`.
- `--bulk-candidate-limit`
  - Maps to `bulk_candidate_limit`.
- `--snippet-candidate-limit`
  - Maps to `snippet_candidate_limit`.
- `--snippet-limit-per-paper`
  - Maps to `snippet_limit_per_paper`.

## Result Fields

- `result.query`
  - Original query string.
- `result.normalized_query`
  - Normalized form used internally for resolution and search.
- `result.possible_interpretations`
  - Candidate `ResolvedPaper` interpretations of the query.
- `result.shortlist`
  - Ranked `TriageCandidate` shortlist.
- `result.follow_up_actions`
  - Workflow names suggested after the right paper is identified.
- `result.considered_candidates`
  - Number of candidates scored across title match, autocomplete, relevance search, bulk search, and snippet evidence.
- `result.notes`
  - Execution notes.

## Nested Object Highlights

- `ResolvedPaper` records how the query resolved and where the match came from.
- `TriageCandidate.paper` is the normalized `PaperSummary`.
- `TriageCandidate.snippet_evidence` contains `SnippetEvidence` records from snippet search.
- `TriageCandidate.score_breakdown.components` shows title, snippet, impact, recency, and autocomplete contributions.
- `TriageCandidate.why` is the short human-readable explanation list.
