from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
import hashlib
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SKILLS_SRC_DIR = ROOT / "skills-src"
PACKAGE_DIR = ROOT / "src" / "semantic_scholar_skills"
DEFAULT_OUTPUT_DIR = ROOT / "skills"
SKILL_NAMES = ("expand-references", "trace-citations", "paper-triage")
SKILL_DOC_FILES = ("SKILL.md", "reference.md", "examples.md")

VENDORED_FILES = (
    Path("__init__.py"),
    Path("config.py"),
    Path("core/client.py"),
    Path("core/exceptions.py"),
    Path("core/requests.py"),
    Path("engine/__init__.py"),
    Path("engine/models.py"),
    Path("engine/resolve.py"),
    Path("engine/scoring.py"),
    Path("engine/expand_references.py"),
    Path("engine/trace_citations.py"),
    Path("engine/paper_triage.py"),
    Path("standalone/__init__.py"),
    Path("standalone/entrypoint.py"),
    Path("standalone/transport_stdlib.py"),
)

VENDORED_CORE_INIT = """\
\"\"\"Minimal core exports for vendored standalone skills.\"\"\"

from .client import S2Client, SupportsRequestJson
from .exceptions import (
    S2ApiError,
    S2Error,
    S2NotFoundError,
    S2RateLimitError,
    S2TimeoutError,
    S2ValidationError,
)
from .requests import (
    AuthorBatchDetailsRequest,
    AuthorDetailsRequest,
    AuthorPapersRequest,
    AuthorSearchRequest,
    PaperAutocompleteRequest,
    PaperAuthorsRequest,
    PaperBatchDetailsRequest,
    PaperBulkSearchRequest,
    PaperCitationsRequest,
    PaperDetailsRequest,
    PaperRecommendationsMultiRequest,
    PaperRecommendationsSingleRequest,
    PaperReferencesRequest,
    PaperRelevanceSearchRequest,
    PaperTitleSearchRequest,
    RequestModel,
    SnippetSearchRequest,
)

__all__ = [
    "AuthorBatchDetailsRequest",
    "AuthorDetailsRequest",
    "AuthorPapersRequest",
    "AuthorSearchRequest",
    "PaperAutocompleteRequest",
    "PaperAuthorsRequest",
    "PaperBatchDetailsRequest",
    "PaperBulkSearchRequest",
    "PaperCitationsRequest",
    "PaperDetailsRequest",
    "PaperRecommendationsMultiRequest",
    "PaperRecommendationsSingleRequest",
    "PaperReferencesRequest",
    "PaperRelevanceSearchRequest",
    "PaperTitleSearchRequest",
    "RequestModel",
    "S2ApiError",
    "S2Client",
    "S2Error",
    "S2NotFoundError",
    "S2RateLimitError",
    "S2TimeoutError",
    "S2ValidationError",
    "SnippetSearchRequest",
    "SupportsRequestJson",
]
"""


@dataclass(frozen=True)
class DriftReport:
    missing_from_target: tuple[str, ...] = ()
    extra_in_target: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()

    @property
    def is_clean(self) -> bool:
        return not (self.missing_from_target or self.extra_in_target or self.changed)


def resolve_output_dir(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_text(destination: Path, contents: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(contents, encoding="utf-8")


def _vendor_package(vendor_root: Path) -> None:
    package_root = vendor_root / "semantic_scholar_skills"
    for relative_path in VENDORED_FILES:
        _copy_file(PACKAGE_DIR / relative_path, package_root / relative_path)
    _write_text(package_root / "core" / "__init__.py", VENDORED_CORE_INIT)


def build_skill_bundle(skill_name: str, destination: Path) -> None:
    source_dir = SKILLS_SRC_DIR / skill_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing skill source directory: {source_dir}")

    for filename in SKILL_DOC_FILES:
        _copy_file(source_dir / filename, destination / filename)

    _copy_file(SKILLS_SRC_DIR / "_shared" / "output_contract.md", destination / "output_contract.md")

    scripts_dir = destination / "scripts"
    _copy_file(source_dir / "run.py", scripts_dir / "run.py")
    _copy_file(SKILLS_SRC_DIR / "_shared" / "launcher.py", scripts_dir / "_shared" / "launcher.py")
    _vendor_package(scripts_dir / "_vendor")


def build_bundle(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for skill_name in SKILL_NAMES:
        build_skill_bundle(skill_name, output_dir / skill_name)
    return output_dir


def _collect_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _diff_directories(expected_dir: Path, target_dir: Path) -> DriftReport:
    expected = _collect_hashes(expected_dir)
    actual = _collect_hashes(target_dir)
    expected_paths = set(expected)
    actual_paths = set(actual)
    return DriftReport(
        missing_from_target=tuple(sorted(expected_paths - actual_paths)),
        extra_in_target=tuple(sorted(actual_paths - expected_paths)),
        changed=tuple(sorted(path for path in expected_paths & actual_paths if expected[path] != actual[path])),
    )


def _print_text_diff(expected_path: Path, actual_path: Path) -> None:
    expected_lines = expected_path.read_text(encoding="utf-8").splitlines()
    actual_lines = actual_path.read_text(encoding="utf-8").splitlines()
    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile=str(expected_path),
        tofile=str(actual_path),
        lineterm="",
    )
    for line in diff:
        print(line)


def check_bundle_drift(target_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    target_dir = Path(target_dir)
    with tempfile.TemporaryDirectory() as temp_dir_name:
        generated_dir = Path(temp_dir_name) / "skills"
        build_bundle(generated_dir)
        report = _diff_directories(generated_dir, target_dir)
        if report.is_clean:
            print(f"Skills bundle is up to date: {target_dir}")
            return True

        print(f"Skills bundle drift detected: {target_dir}")
        if report.missing_from_target:
            print("Missing from target:")
            for relative_path in report.missing_from_target:
                print(f"  {relative_path}")
        if report.extra_in_target:
            print("Extra in target:")
            for relative_path in report.extra_in_target:
                print(f"  {relative_path}")
        if report.changed:
            print("Changed files:")
            for relative_path in report.changed:
                print(f"  {relative_path}")
            for relative_path in report.changed:
                expected_path = generated_dir / relative_path
                actual_path = target_dir / relative_path
                if expected_path.suffix in {".md", ".py"}:
                    _print_text_diff(expected_path, actual_path)
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or check the tracked Semantic Scholar skills bundle.")
    parser.add_argument(
        "--output-dir",
        default="skills",
        help="Bundle destination directory. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare generated output against an existing bundle instead of writing files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    if args.check:
        return 0 if check_bundle_drift(output_dir) else 1
    build_bundle(output_dir)
    print(f"Built skills bundle at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
