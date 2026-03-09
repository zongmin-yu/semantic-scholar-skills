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

WORKFLOW = "paper-triage"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Semantic Scholar paper-triage workflow.",
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Quoted paper query or title fragment.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key_override",
        help="Semantic Scholar API key override.",
    )
    parser.add_argument(
        "--shortlist-size",
        type=int,
        default=7,
        dest="shortlist_size",
        help="Maximum number of shortlisted papers to return.",
    )
    parser.add_argument(
        "--relevance-limit",
        type=int,
        default=10,
        dest="relevance_limit",
        help="Maximum number of first-pass relevance candidates.",
    )
    parser.add_argument(
        "--bulk-candidate-limit",
        type=int,
        default=20,
        dest="bulk_candidate_limit",
        help="Maximum number of broader recall candidates to keep.",
    )
    parser.add_argument(
        "--snippet-candidate-limit",
        type=int,
        default=5,
        dest="snippet_candidate_limit",
        help="How many preliminary candidates receive snippet search.",
    )
    parser.add_argument(
        "--snippet-limit-per-paper",
        type=int,
        default=3,
        dest="snippet_limit_per_paper",
        help="Maximum number of snippets collected for each snippet target.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = launch(
        WORKFLOW,
        query=" ".join(args.query),
        api_key_override=args.api_key_override,
        shortlist_size=args.shortlist_size,
        relevance_limit=args.relevance_limit,
        bulk_candidate_limit=args.bulk_candidate_limit,
        snippet_candidate_limit=args.snippet_candidate_limit,
        snippet_limit_per_paper=args.snippet_limit_per_paper,
    )
    print(dumps_payload(outcome.payload))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
