"""Build (or reuse) the frozen multimodal suite ``p4-multimodal-v1`` and score the baseline.

Offline. Needs the verified SomBench files (``scripts/prepare_sombench.py``). The P3 suite is
never touched. The report contains sealed labels, so it is written under the evaluator root only.

    python scripts/p4_build_multimodal.py \
        --scenarios-root data/scenarios --eval-root data/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iceclaw.config import load_dotenv_file, load_settings
from iceclaw.data.catalog import discover
from iceclaw.evaluation import multimodal as mm
from iceclaw.evaluation import score
from iceclaw.evaluation.fixtures import load_manifest, sha256_bytes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios-root", type=Path, default=Path("data/scenarios"))
    ap.add_argument("--eval-root", type=Path, default=Path("data/evaluation"))
    ap.add_argument("--suite-id", default=mm.SUITE_ID)
    ap.add_argument("--seed", type=int, default=mm.DEFAULT_SUITE_SEED)
    args = ap.parse_args(argv)

    load_dotenv_file()  # never printed
    settings = load_settings()
    if settings.sombench_root is None:
        print("ICECLAW_SOMBENCH_ROOT is not set", file=sys.stderr)
        return 1
    if (args.scenarios_root / args.suite_id).exists():
        print(f"reusing frozen suite at {args.scenarios_root / args.suite_id}")
    else:
        tiles = mm.load_observation_tiles(discover(settings))
        dm = (settings.sombench_root / "download_manifest.json").read_bytes()
        m = mm.build_multimodal_suite(
            tiles,
            args.scenarios_root,
            args.eval_root,
            suite_seed=args.seed,
            suite_id=args.suite_id,
            dataset_provenance={"download_manifest_sha256": sha256_bytes(dm)},
        )
        print(f"built {m.suite_id}: {len(m.cases)} cases, manifest sha256 {m.body_sha256()}")

    problems = mm.verify_multimodal_suite(args.scenarios_root, args.eval_root, args.suite_id)
    if problems:
        print("SUITE INTEGRITY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    manifest = load_manifest(args.scenarios_root, args.suite_id)
    cfg = mm.ObsBaselineConfig()
    results = {
        cid: mm.run_multimodal_baseline(
            mm.load_multimodal_scenario(args.scenarios_root, args.suite_id, cid), cfg
        )
        for cid in manifest.detection_case_ids
    }
    frozen = score.freeze_detections(manifest, {c: r.flagged for c, r in results.items()})
    report = score.score_detections(frozen, args.scenarios_root, args.eval_root, args.suite_id)
    for row in report["per_case"]:
        row["triggered_rules"] = results[row["case_id"]].triggered_rules
    report.update(
        manifest_sha256=manifest.body_sha256(),
        baseline_config_sha256=cfg.sha256(),
        suite_id=manifest.suite_id,
    )
    out = args.eval_root / args.suite_id / "baseline_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(
        json.dumps({k: v for k, v in report.items() if k != "per_case"}, indent=2, sort_keys=True)
    )
    for r in report["per_case"]:
        print(r["case_id"], r["fault_id"], r["outcome"], r["triggered_rules"])
    print(f"full report (contains sealed labels): {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
