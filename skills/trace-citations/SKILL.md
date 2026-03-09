---
name: trace-citations
description: Trace the citation neighborhood around one focal paper into foundations, descendants, bridges, weak edges, and optional second-hop links
disable-model-invocation: true
user-invocable: true
allowed-tools: Bash, Read
argument-hint: "<focal-query> [--depth 1|2] [--max-references <n>] [--max-citations <n>] [--second-hop-limit <n>]"
---

# Trace Citations

Map the citation graph around one focal paper into useful buckets.
Use this when the human wants lineage, influence, and strong versus weak citation edges around a paper.

## Arguments

- The positional argument is the focal paper query. Quote multi-word titles.
- `--depth 1|2` controls whether to expand a second hop from the strongest first-hop edges.
- `--max-references <n>` and `--max-citations <n>` cap the first-hop fetch sizes.
- `--second-hop-limit <n>` caps how many first-hop anchors get expanded at depth two.

## Workflow

1. Run `python scripts/run.py ...`.
2. Read `result.foundations` for strong references behind the focal paper.
3. Read `result.direct_descendants` for strong citing descendants.
4. Read `result.bridge_nodes` for medium-confidence connectors with rich context or intent signal.
5. Read `result.weak_edges` for low-signal edges that are probably less useful.
6. If `depth=2`, read `result.second_hop` only after the first-hop picture looks sensible.

## Output

- The script prints the unified JSON envelope described in `output_contract.md`.
- The underlying workflow result is `CitationTraceResult.to_dict()`.
- `result.reference_count_examined` and `result.citation_count_examined` show the first-hop search breadth.

## When To Escalate

- The focal paper resolves incorrectly.
- The API returns very sparse context and intent data, making edge interpretation weak.
- The first-hop graph is too noisy and needs a tighter focal paper choice before going to depth two.
