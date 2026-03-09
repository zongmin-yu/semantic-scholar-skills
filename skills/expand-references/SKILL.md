---
name: expand-references
description: Expand one to three seed papers into nearby, bridge, foundational, methodological, recent, and survey follow-ups
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash, Read
argument-hint: "<seed-1> [seed-2] [seed-3] [--negative <paper>] [--pool all-cs|recent] [--limit <n>] [--per-bucket-limit <n>]"
---

# Expand References

Turn one to three seed papers into a structured follow-up reading list.
Use this when the human already has anchor papers and wants the next papers to read.

## Arguments

- Positional arguments are the seed papers. Quote multi-word titles.
- `--negative <paper>` may be repeated to push the workflow away from an unwanted cluster.
- `--pool all-cs|recent` selects the Semantic Scholar recommendation pool.
- `--limit <n>` controls how many raw recommendations are requested before reranking.
- `--per-bucket-limit <n>` caps each curated bucket after scoring.

## Workflow

1. Run `python scripts/run.py ...`.
2. Read `result.closest_neighbors` for the immediate next reads.
3. Read `result.bridge_papers` for papers that connect multiple seeds.
4. Read `result.foundational`, `result.methodological`, `result.recent`, and `result.surveys_or_benchmarks` for curated slices of the neighborhood.
5. If the result is sparse or off-topic, adjust the seed set or add `--negative` papers and rerun.

## Output

- The script prints the unified JSON envelope described in `output_contract.md`.
- The underlying workflow result is `ExpandReferencesResult.to_dict()`.
- `result.notes` captures dropped records and other execution notes.

## When To Escalate

- Fewer than one clear seed paper is available.
- The resolved seeds are obviously duplicates or wrong papers.
- The output is empty even after trying better seeds or a different recommendation pool.
