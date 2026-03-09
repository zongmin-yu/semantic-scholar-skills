from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "bundle_skills.py"
DRIFT_SCRIPT = REPO_ROOT / "scripts" / "check_bundle_drift.py"
SKILL_NAMES = ("expand-references", "trace-citations", "paper-triage")


def run_bundle(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BUNDLE_SCRIPT), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def run_drift_check(output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DRIFT_SCRIPT), "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bundle_script_materializes_expected_skill_tree(tmp_path) -> None:
    output_dir = tmp_path / "skills"
    completed = run_bundle(output_dir)

    assert completed.returncode == 0, completed.stderr
    for skill_name in SKILL_NAMES:
        skill_dir = output_dir / skill_name
        assert (skill_dir / "SKILL.md").is_file()
        assert (skill_dir / "reference.md").is_file()
        assert (skill_dir / "examples.md").is_file()
        assert (skill_dir / "output_contract.md").is_file()
        assert (skill_dir / "scripts" / "run.py").is_file()
        assert (skill_dir / "scripts" / "_shared" / "launcher.py").is_file()
        assert (
            skill_dir
            / "scripts"
            / "_vendor"
            / "semantic_scholar_skills"
            / "standalone"
            / "entrypoint.py"
        ).is_file()


def test_bundle_patches_vendored_core_init_to_avoid_transport_exports(tmp_path) -> None:
    output_dir = tmp_path / "skills"
    completed = run_bundle(output_dir)

    assert completed.returncode == 0, completed.stderr
    vendored_core_init = (
        output_dir
        / "expand-references"
        / "scripts"
        / "_vendor"
        / "semantic_scholar_skills"
        / "core"
        / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "transport" not in vendored_core_init
    assert "SupportsRequestJson" in vendored_core_init
    assert "PaperRecommendationsMultiRequest" in vendored_core_init


def test_check_bundle_drift_reports_manual_edits(tmp_path) -> None:
    output_dir = tmp_path / "skills"
    completed = run_bundle(output_dir)

    assert completed.returncode == 0, completed.stderr

    clean = run_drift_check(output_dir)
    assert clean.returncode == 0, clean.stdout + clean.stderr

    edited_file = output_dir / "paper-triage" / "reference.md"
    edited_file.write_text(edited_file.read_text(encoding="utf-8") + "\nManual drift.\n", encoding="utf-8")

    dirty = run_drift_check(output_dir)
    assert dirty.returncode == 1
    assert "paper-triage/reference.md" in (dirty.stdout + dirty.stderr)
