from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _configure_path() -> None:
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir, script_dir.parent):
        if (candidate / "_shared" / "launcher.py").is_file():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return
    raise RuntimeError("Could not locate _shared/launcher.py")


_configure_path()

from _shared.launcher import dumps_payload, launch

WORKFLOW = "expand-references"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Semantic Scholar expand-references workflow.",
    )
    parser.add_argument(
        "seeds",
        nargs="+",
        help="One to three quoted seed papers (title, DOI, or paper ID).",
    )
    parser.add_argument(
        "--negative",
        action="append",
        default=[],
        dest="negative_seeds",
        help="Optional negative seed paper to push the neighborhood away from.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key_override",
        help="Semantic Scholar API key override.",
    )
    parser.add_argument(
        "--pool",
        choices=["all-cs", "recent"],
        default="all-cs",
        dest="recommendation_pool",
        help="Semantic Scholar recommendation pool.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        dest="recommendation_limit",
        help="Maximum number of raw recommendations to request.",
    )
    parser.add_argument(
        "--per-bucket-limit",
        type=int,
        default=5,
        dest="per_bucket_limit",
        help="Maximum items retained per curated output bucket.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = launch(
        WORKFLOW,
        seeds=args.seeds,
        negative_seeds=args.negative_seeds,
        api_key_override=args.api_key_override,
        recommendation_pool=args.recommendation_pool,
        recommendation_limit=args.recommendation_limit,
        per_bucket_limit=args.per_bucket_limit,
    )
    print(dumps_payload(outcome.payload))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
