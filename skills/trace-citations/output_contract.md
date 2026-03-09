# Output Contract

Every bundled skill prints exactly one JSON object to stdout.

- Exit code `0` means `status: "ok"`.
- Exit code `1` means `status: "error"`.
- The payload is stable across installed-package and vendored fallback execution.

## Success Shape

```json
{
  "schema_version": "1.0",
  "workflow": "paper-triage",
  "status": "ok",
  "runtime": {
    "mode": "installed",
    "module": "semantic_scholar_skills.standalone",
    "path": "/abs/path/to/semantic_scholar_skills/standalone/__init__.py"
  },
  "arguments": {
    "query": "bert",
    "api_key_override": null,
    "shortlist_size": 7,
    "relevance_limit": 10,
    "bulk_candidate_limit": 20,
    "snippet_candidate_limit": 5,
    "snippet_limit_per_paper": 3
  },
  "result": {
    "...": "top-level workflow result from result.to_dict()"
  }
}
```

## Error Shape

```json
{
  "schema_version": "1.0",
  "workflow": "paper-triage",
  "status": "error",
  "runtime": {
    "mode": "vendored",
    "module": "semantic_scholar_skills.standalone",
    "path": "/abs/path/to/scripts/_vendor/semantic_scholar_skills/standalone/__init__.py"
  },
  "arguments": {
    "query": ""
  },
  "error": {
    "type": "S2ValidationError",
    "message": "Query string cannot be empty",
    "details": {},
    "field": "query"
  }
}
```

## Top-Level Fields

- `schema_version`
  - Starts at `"1.0"`.
- `workflow`
  - One of `expand-references`, `trace-citations`, or `paper-triage`.
- `status`
  - `"ok"` or `"error"`.
- `runtime`
  - `mode` is `installed`, `vendored`, or `unavailable`.
  - `module` is the imported Python module name.
  - `path` is the resolved module file path when available.
- `arguments`
  - The exact parsed CLI arguments passed into the workflow.
- `result`
  - Present only when `status` is `"ok"`.
  - This is the direct `to_dict()` serialization of the engine result object.
- `error`
  - Present only when `status` is `"error"`.
  - Structured pass-through from the raised exception when possible.

## Consumer Rules

- Always branch on `status`, not just the process exit code.
- Treat `runtime.mode` as observability metadata, not business logic.
- Read workflow-specific detail from `result`, especially each result object's `notes` field.
- Do not scrape stderr for machine-readable information. The JSON payload is the contract.
