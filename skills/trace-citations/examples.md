# Trace Citations Examples

## First-Hop Lineage

```bash
python scripts/run.py "Attention Is All You Need"
```

This is the default lineage view: one focal paper, one hop, scored and bucketed.

## Add A Second Hop

```bash
python scripts/run.py \
  "Attention Is All You Need" \
  --depth 2 \
  --max-references 30 \
  --max-citations 30 \
  --second-hop-limit 5
```

Use this only after the first-hop output already looks correct.

## How To Read The Output

- Start with `result.foundations` to see the prior work feeding into the focal paper.
- Use `result.direct_descendants` to identify papers that clearly extend or rely on it.
- Use `result.bridge_nodes` when you care about connectors across adjacent clusters.
- Treat `result.weak_edges` as low-priority unless the human specifically wants exhaustive coverage.
