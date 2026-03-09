from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from semantic_scholar_skills.core.exceptions import S2ValidationError
import semantic_scholar_skills.standalone as standalone

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "bundle_skills.py"


def build_bundle(output_dir: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(BUNDLE_SCRIPT), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def load_module(module_path: Path, module_name: str):
    for name in (module_name, "_shared", "_shared.launcher"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeResult:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


@pytest.mark.parametrize(
    ("skill_name", "argv", "expected_workflow", "expected_kwargs"),
    [
        (
            "expand-references",
            [
                "Dense Passage Retrieval",
                "Retrieval-Augmented Generation",
                "--negative",
                "Biomedical Entity Linking",
                "--pool",
                "recent",
                "--limit",
                "12",
                "--per-bucket-limit",
                "3",
            ],
            "expand-references",
            {
                "seeds": ["Dense Passage Retrieval", "Retrieval-Augmented Generation"],
                "negative_seeds": ["Biomedical Entity Linking"],
                "api_key_override": None,
                "recommendation_pool": "recent",
                "recommendation_limit": 12,
                "per_bucket_limit": 3,
            },
        ),
        (
            "trace-citations",
            [
                "Attention",
                "Is",
                "All",
                "You",
                "Need",
                "--depth",
                "2",
                "--max-references",
                "7",
                "--max-citations",
                "8",
                "--second-hop-limit",
                "4",
            ],
            "trace-citations",
            {
                "focal_query": "Attention Is All You Need",
                "api_key_override": None,
                "depth": 2,
                "max_references": 7,
                "max_citations": 8,
                "second_hop_limit": 4,
            },
        ),
        (
            "paper-triage",
            [
                "retrieval",
                "augmented",
                "generation",
                "--shortlist-size",
                "5",
                "--relevance-limit",
                "11",
                "--bulk-candidate-limit",
                "13",
                "--snippet-candidate-limit",
                "3",
                "--snippet-limit-per-paper",
                "2",
            ],
            "paper-triage",
            {
                "query": "retrieval augmented generation",
                "api_key_override": None,
                "shortlist_size": 5,
                "relevance_limit": 11,
                "bulk_candidate_limit": 13,
                "snippet_candidate_limit": 3,
                "snippet_limit_per_paper": 2,
            },
        ),
    ],
)
def test_generated_run_scripts_emit_success_contract(
    monkeypatch,
    tmp_path,
    capsys,
    skill_name: str,
    argv: list[str],
    expected_workflow: str,
    expected_kwargs: dict[str, object],
) -> None:
    output_dir = tmp_path / "skills"
    build_bundle(output_dir)
    run_module = load_module(
        output_dir / skill_name / "scripts" / "run.py",
        f"{skill_name.replace('-', '_')}_run_success",
    )

    recorded: dict[str, object] = {}

    async def fake_run_workflow(workflow: str, **kwargs):
        recorded["workflow"] = workflow
        recorded["kwargs"] = kwargs
        return FakeResult({"echo": workflow, "kwargs": kwargs})

    monkeypatch.setattr(standalone, "run_workflow", fake_run_workflow)

    exit_code = run_module.main(argv)
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert recorded["workflow"] == expected_workflow
    assert recorded["kwargs"] == expected_kwargs
    assert payload["status"] == "ok"
    assert payload["workflow"] == expected_workflow
    assert payload["runtime"]["mode"] == "installed"
    assert payload["arguments"] == expected_kwargs
    assert payload["result"] == {"echo": expected_workflow, "kwargs": expected_kwargs}


def test_generated_run_script_emits_error_contract(monkeypatch, tmp_path, capsys) -> None:
    output_dir = tmp_path / "skills"
    build_bundle(output_dir)
    run_module = load_module(
        output_dir / "paper-triage" / "scripts" / "run.py",
        "paper_triage_run_error",
    )

    async def fake_run_workflow(workflow: str, **kwargs):
        raise S2ValidationError(
            message="Query string cannot be empty",
            details={"hint": "Provide a non-empty paper query"},
            field="query",
        )

    monkeypatch.setattr(standalone, "run_workflow", fake_run_workflow)

    exit_code = run_module.main(["bert"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["status"] == "error"
    assert payload["workflow"] == "paper-triage"
    assert payload["runtime"]["mode"] == "installed"
    assert payload["error"]["type"] == "S2ValidationError"
    assert payload["error"]["field"] == "query"
    assert payload["error"]["details"]["hint"] == "Provide a non-empty paper query"
