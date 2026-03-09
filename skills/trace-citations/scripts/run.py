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

WORKFLOW = "trace-citations"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Semantic Scholar trace-citations workflow.",
    )
    parser.add_argument(
        "focal_query",
        nargs="+",
        help="Quoted focal paper query (title, DOI, or paper ID).",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key_override",
        help="Semantic Scholar API key override.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        choices=[1, 2],
        default=1,
        help="Citation trace depth.",
    )
    parser.add_argument(
        "--max-references",
        type=int,
        default=50,
        dest="max_references",
        help="Maximum number of first-hop references to request.",
    )
    parser.add_argument(
        "--max-citations",
        type=int,
        default=50,
        dest="max_citations",
        help="Maximum number of first-hop citations to request.",
    )
    parser.add_argument(
        "--second-hop-limit",
        type=int,
        default=10,
        dest="second_hop_limit",
        help="Maximum number of first-hop anchors expanded when depth=2.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outcome = launch(
        WORKFLOW,
        focal_query=" ".join(args.focal_query),
        api_key_override=args.api_key_override,
        depth=args.depth,
        max_references=args.max_references,
        max_citations=args.max_citations,
        second_hop_limit=args.second_hop_limit,
    )
    print(dumps_payload(outcome.payload))
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
