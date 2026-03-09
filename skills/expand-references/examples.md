# Expand References Examples

## Two Related Seeds

```bash
python scripts/run.py \
  "Dense Passage Retrieval for Open-Domain Question Answering" \
  "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
```

Use this when the human already knows two papers in the same neighborhood and wants a structured expansion around them.

## Push Away From an Unwanted Cluster

```bash
python scripts/run.py \
  "Attention Is All You Need" \
  --negative "Biomedical Entity Linking" \
  --pool recent \
  --limit 40 \
  --per-bucket-limit 4
```

Use `--negative` when the seed title is overloaded and you want to bias away from a different field.

## How To Read The Output

- Start with `result.closest_neighbors` for the shortest path to more papers.
- Read `result.bridge_papers` when you want cross-seed connectors.
- Read `result.surveys_or_benchmarks` when you need a fast overview before going deeper.
- If the top bucket looks wrong, inspect `result.seeds` first to confirm the seed resolution step.
