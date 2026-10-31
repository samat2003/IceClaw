"""IceClaw CLI shell (P0): all commands fail with NOT_IMPLEMENTED until their phase."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from iceclaw.config import load_dotenv_file, load_settings

NOT_IMPLEMENTED_EXIT = 2

_PHASES = {
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
    lr = sub.add_parser("list-regions", help="List validated regions from the real catalog")
    lr.add_argument("--json", action="store_true", help="machine-readable output")
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
    if args.command == "list-regions":
        return _list_regions(as_json=args.json)
    print(
        f"NOT_IMPLEMENTED: '{args.command}' is scheduled for phase {_PHASES[args.command]}.",
        file=sys.stderr,
    )
    return NOT_IMPLEMENTED_EXIT


def _list_regions(as_json: bool) -> int:
    import json

    from iceclaw import services

    catalog = services.load_catalog(load_settings())
    rows = services.describe_regions(catalog)
    for rej in catalog.rejections:
        print(
            f"REJECTED {rej.sector_id or '<export>'} [{rej.code}] "
            f"at {rej.export_root}: {rej.message}",
            file=sys.stderr,
        )
    if as_json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            lat, lon = r["center_lat_lon"]
            print(
                f"{r['region_id']}  pole={r['pole']}  readiness={r['readiness']}  "
                f"split={r['source_split']}  source={r['source_mode']}"
            )
            print(
                f"    centre=({lat:.4f}, {lon:.4f})  grid={r['grid'][0]}x{r['grid'][1]} "
                f"@ {r['spatial_resolution_m']} m/px  dataset_rev={r['source_revision'][:12]}"
            )
            print(f"    checkpoint={r['checkpoint'][:60]}...")
            print(f"    pixelwise: {', '.join(r['pixelwise_modalities'])}")
            for m in r["other_metadata"]:
                print(f"    {m['kind']}: {m['name']} = {m['value']} {m['unit'] or ''}".rstrip())
            print(f"    UNAVAILABLE: {', '.join(r['unavailable_modalities']) or 'none'}")
            if r["readiness_reason"]:
                print(f"    note: {r['readiness_reason']}")
    if not rows:
        print("NO_VALIDATED_REGIONS: no usable MOONATLAS export found.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
