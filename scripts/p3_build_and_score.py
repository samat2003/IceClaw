"""Build (or reuse) the frozen P3 prediction-only suite, run the deterministic baseline, score it.

Offline: reads the real catalog (published MOONATLAS example) and writes under two SEPARATE roots.
The report contains sealed labels, so it is written under the evaluator root only.

    python scripts/p3_build_and_score.py --scenarios-root data/scenarios --eval-root data/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from iceclaw.config import load_dotenv_file, load_settings
from iceclaw.evaluation import baselines, fixtures, score


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenarios-root", type=Path, default=Path("data/scenarios"))
    ap.add_argument("--eval-root", type=Path, default=Path("data/evaluation"))
    ap.add_argument("--suite-id", default=fixtures.SUITE_ID)
    ap.add_argument("--region", default="ice-s-0001-0002")
    ap.add_argument("--seed", type=int, default=fixtures.DEFAULT_SUITE_SEED)
    args = ap.parse_args(argv)

    load_dotenv_file()  # never printed
    scen = args.scenarios_root / args.suite_id
    if scen.exists():
        print(f"reusing frozen suite at {scen}")
    else:
        source = fixtures.load_real_source_bundle(load_settings(), args.region)
        m = fixtures.build_suite(
            source,
            args.scenarios_root,
            args.eval_root,
            suite_seed=args.seed,
            suite_id=args.suite_id,
        )
        print(f"built suite {m.suite_id}, manifest sha256 {m.body_sha256()}")

    problems = fixtures.verify_suite(args.scenarios_root, args.eval_root, args.suite_id)
    if problems:
        print("SUITE INTEGRITY FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    manifest = fixtures.load_manifest(args.scenarios_root, args.suite_id)
    cfg = baselines.BaselineConfig()

    results = {
        cid: baselines.run_baseline(
            fixtures.load_scenario(args.scenarios_root, args.suite_id, cid), cfg
        )
        for cid in manifest.detection_case_ids
    }
    frozen = score.freeze_detections(manifest, {c: r.flagged for c, r in results.items()})
    report = score.score_detections(frozen, args.scenarios_root, args.eval_root, args.suite_id)
    for row in report["per_case"]:
        row["triggered_rules"] = results[row["case_id"]].triggered_rules

    sec = manifest.secondary_task
    ranking = baselines.rank_windows_baseline(
        fixtures.load_scenario(args.scenarios_root, args.suite_id, sec["case_id"]),
        sec["windows"],
        sec["top_k"],
    )
    report["secondary_ranking"] = score.score_ranking(
        score.freeze_ranking(manifest, ranking), args.scenarios_root, args.eval_root, args.suite_id
    )
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
    print(f"full report (contains sealed labels): {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
