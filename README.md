# semantic-scholar-skills

[![CI](https://github.com/zongmin-yu/semantic-scholar-skills/actions/workflows/ci.yml/badge.svg)](https://github.com/zongmin-yu/semantic-scholar-skills/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**S2-first discovery engine for AI research workflows.**

Three research workflows built on the Semantic Scholar API — available as a Python library, Claude Code skills, or a 16-tool MCP server.

## Workflows

### `/expand-references` — Discover related work from seed papers

Feed 1–3 seed papers. Get back curated buckets: foundational, methodological, recent, survey, bridge papers, and closest neighbors. Powered by S2's Recommendations API with positive/negative seeds.

### `/trace-citations` — Map a paper's citation neighborhood

Pick a focal paper. Trace its citations and references with context snippets, intent labels, and influence flags. Returns foundations, descendants, bridge connections, and optional second-hop expansion.

### `/paper-triage` — Triage an ambiguous query into a shortlist

Type a vague query. The engine runs autocomplete, relevance search, bulk search, and snippet extraction in parallel, then scores and ranks into a shortlist with explanations.

## Quick Start

### As Claude Code skills (copy & use)

```bash
# Copy any skill to your personal skills directory
cp -r skills/expand-references ~/.claude/skills/

# Then in Claude Code:
/expand-references Attention Is All You Need
```

Each skill is **fully self-contained** — no `pip install` required. The bundled standalone runtime uses only Python stdlib.

### As a Python library

```bash
pip install semantic-scholar-skills
```

core library only: installs `core/`, `engine/`, and the stdlib-backed `standalone/` runtime.

```python
import asyncio
from semantic_scholar_skills.core import get_default_client, cleanup_client
from semantic_scholar_skills.engine import paper_triage

async def main():
    client = get_default_client()
    try:
        result = await paper_triage(client, "retrieval augmented generation")
        for paper in result.shortlist[:5]:
            print(f"{paper.score:.2f}  {paper.paper.title}")
    finally:
        await cleanup_client()

asyncio.run(main())
```

### As an MCP server

```bash
pip install "semantic-scholar-skills[mcp]"
semantic-scholar-skills-mcp
```

full MCP stack: installs `fastmcp`, `fastapi`, and `uvicorn` in addition to the core library.

16 tools covering papers, authors, citations, recommendations, and snippets. Works with Claude Desktop, Cursor, and any MCP client.

## Architecture

```
              Claude Code Skills          MCP Server (16 tools)
                    │                           │
                    ▼                           ▼
            ┌─────────────┐             ┌─────────────┐
            │  skills-src/ │             │    mcp/     │
            │  (SKILL.md)  │             │  (FastMCP)  │
            └──────┬───────┘             └──────┬──────┘
                   │                            │
                   ▼                            ▼
            ┌──────────────────────────────────────┐
            │         engine/ — Workflow Logic      │
            │  expand_references · trace_citations  │
            │  paper_triage · resolve · scoring     │
            └──────────────────┬───────────────────┘
                               │
                   ┌───────────┴───────────┐
                   ▼                       ▼
            ┌─────────────┐         ┌─────────────┐
            │   core/     │         │ standalone/  │
            │  (httpx)    │         │  (urllib)    │
            └─────────────┘         └─────────────┘
                   │                       │
                   └───────────┬───────────┘
                               ▼
                    Semantic Scholar API
```

**Source is shared, artifacts are self-contained.** The `engine/` and `standalone/` layers are vendored into each skill bundle by `bundle_skills.py`, so users can copy a single folder and it just works.

## API Key (Optional)

```bash
export SEMANTIC_SCHOLAR_API_KEY=your-key-here
```

Without a key: 100 requests per 5 minutes. With a key: up to 10 req/s. Get one free at [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api).

## Skill Bundles

The `skills/` directory contains pre-built, self-contained bundles for Claude Code:

| Skill | What it does | Example |
|-------|-------------|---------|
| `expand-references` | Seed papers → curated reading buckets | `/expand-references "Attention Is All You Need"` |
| `trace-citations` | Focal paper → citation lineage map | `/trace-citations "BERT: Pre-training of Deep Bidirectional Transformers"` |
| `paper-triage` | Vague query → ranked shortlist | `/paper-triage retrieval augmented generation` |

**Install**: Copy a skill folder to `~/.claude/skills/` (personal) or `.claude/skills/` (project-level).

**No dependencies**: Each bundle includes a vendored runtime that uses only Python stdlib. If the full package is installed, it uses that instead for better performance.

> Skill bundles are not included in the published wheel. To use them, clone the repository or copy a generated bundle from `skills/`.

## MCP Server

16 tools organized by domain:

| Domain | Tools |
|--------|-------|
| **Papers** | `paper_relevance_search`, `paper_bulk_search`, `paper_title_search`, `paper_details`, `paper_batch_details`, `paper_authors`, `paper_citations`, `paper_references`, `paper_autocomplete`, `snippet_search` |
| **Authors** | `author_search`, `author_details`, `author_papers`, `author_batch_details` |
| **Recommendations** | `get_paper_recommendations_single`, `get_paper_recommendations_multi` |

Configure for Claude Desktop (`~/.config/claude-desktop/config.json`):

```json
{
  "mcpServers": {
    "semantic-scholar": {
      "command": "semantic-scholar-skills-mcp",
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Python API

### Core client — direct S2 API access

```python
from semantic_scholar_skills.core import get_default_client, PaperDetailsRequest

client = get_default_client()
paper = await client.get_paper(
    PaperDetailsRequest(paper_id="CorpusId:215416146", fields=["title", "year", "authors"])
)
```

### Engine — higher-level workflows

```python
from semantic_scholar_skills.engine import expand_references, trace_citations, paper_triage

# Expand from seeds
result = await expand_references(client, ["Attention Is All You Need"], per_bucket_limit=3)

# Trace citation neighborhood
result = await trace_citations(client, "BERT", max_references=50, max_citations=50)

# Triage a query
result = await paper_triage(client, "graph neural networks for molecules")
```

All engine functions return frozen dataclasses with `.to_dict()` for serialization.

## Development

```bash
git clone https://github.com/zongmin-yu/semantic-scholar-skills.git
cd semantic-scholar-skills
pip install -e '.[test]'

# Run tests (offline, no API key needed)
pytest -m "not live" -q

# Regenerate skill bundles
python scripts/bundle_skills.py

# Check bundle drift
python scripts/check_bundle_drift.py

# Audit S2 API field drift
python scripts/spec_audit.py
```

## Background

This project is the successor to [`semantic-scholar-fastmcp-mcp-server`](https://github.com/zongmin-yu/semantic-scholar-fastmcp-mcp-server), which remains the MCP-server-only implementation (~100 stars). This repo adds the workflow engine and Claude Code skills while keeping full MCP parity.

## Semantic Scholar API Terms

This project uses the [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) provided by the Allen Institute for AI. Please review the [API License Agreement](https://api.semanticscholar.org/license/) before use.

## License

MIT
