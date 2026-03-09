# Paper Triage Examples

## Simple Title Fragment

```bash
python scripts/run.py bert
```

Use this when the human only remembers a short fragment.

## Wider Recall And Smaller Shortlist

```bash
python scripts/run.py \
  "retrieval augmented generation" \
  --shortlist-size 5 \
  --relevance-limit 12 \
  --bulk-candidate-limit 20 \
  --snippet-candidate-limit 4 \
  --snippet-limit-per-paper 2
```

Use this when the query is overloaded and you want broader recall before reranking.

## How To Read The Output

- Inspect `result.possible_interpretations` first if the query is ambiguous.
- Then read `result.shortlist[0]` and `result.shortlist[1]` before trusting the rest of the list.
- Use `result.follow_up_actions` to move into `trace-citations` or `expand-references` once the right paper is confirmed.
