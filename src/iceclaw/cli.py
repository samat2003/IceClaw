"""IceClaw CLI shell (P0): all commands fail with NOT_IMPLEMENTED until their phase."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from iceclaw.config import load_dotenv_file

NOT_IMPLEMENTED_EXIT = 2

_PHASES = {
    "list-regions": "P2",
    "investigate": "P5",
    "verify": "P5",
    "benchmark": "P7",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iceclaw",
        description="IceClaw lunar ice-prospectivity audit workbench "
        "(prospectivity index, not detected ice).",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.add_parser("list-regions", help="List validated regions (P2)")
    inv = sub.add_parser("investigate", help="Run an investigation (P5)")
    inv.add_argument("--region", required=True, metavar="REGION_ID")
    ver = sub.add_parser("verify", help="Verify a saved run bundle (P5)")
    ver.add_argument("--run", required=True, metavar="RUN_ID")
    ben = sub.add_parser("benchmark", help="Run a benchmark config (P7)")
    ben.add_argument("--config", required=True, metavar="PATH")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    load_dotenv_file()  # values are never printed
    print(
        f"NOT_IMPLEMENTED: '{args.command}' is scheduled for phase {_PHASES[args.command]}.",
        file=sys.stderr,
    )
    return NOT_IMPLEMENTED_EXIT


if __name__ == "__main__":
    sys.exit(main())
