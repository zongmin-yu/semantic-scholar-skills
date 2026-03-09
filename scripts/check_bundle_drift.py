from __future__ import annotations

import argparse

from bundle_skills import check_bundle_drift as check_skills_bundle_drift
from bundle_skills import resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the tracked skills bundle has drifted from skills-src/.")
    parser.add_argument(
        "--output-dir",
        default="skills",
        help="Bundle directory to compare against generated output. Relative paths are resolved from the repo root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    return 0 if check_skills_bundle_drift(output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
