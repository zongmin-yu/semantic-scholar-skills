from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ci_workflow_tests_all_supported_python_versions() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert re.search(r'python-version:\s*\["3\.10",\s*"3\.11",\s*"3\.12"\]', workflow)
    assert "Set up Python ${{ matrix.python-version }}" in workflow
    assert "python-version: ${{ matrix.python-version }}" in workflow


def test_pyproject_splits_core_mcp_and_test_dependencies() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert re.search(
        r'dependencies = \[\s+"httpx>=0\.24\.0",\s+"python-dotenv>=1\.0\.0",\s*\]',
        pyproject,
    )
    assert re.search(
        r'mcp = \[\s+"fastmcp>=2\.0\.0,<3\.0\.0",\s+"fastapi>=0\.115\.0",\s+"uvicorn>=0\.32\.0",\s*\]',
        pyproject,
    )
    assert re.search(
        r'test = \[\s+"fastmcp>=2\.0\.0,<3\.0\.0",\s+"fastapi>=0\.115\.0",\s+"uvicorn>=0\.32\.0",\s+"pytest>=7\.3\.1",\s+"pytest-asyncio>=0\.21\.0",\s*\]',
        pyproject,
    )


def test_readme_documents_core_and_mcp_install_commands() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "pip install semantic-scholar-skills" in readme
    assert 'pip install "semantic-scholar-skills[mcp]"' in readme
    assert "core library only" in readme
    assert "full MCP stack" in readme
