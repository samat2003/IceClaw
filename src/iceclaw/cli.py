"""IceClaw CLI. Every command calls the same services as the Streamlit UI."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from iceclaw.config import load_dotenv_file, load_settings

NOT_IMPLEMENTED_EXIT = 2  # kept for compatibility; every command is implemented since P7


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iceclaw",
        description="IceClaw lunar ice-prospectivity audit workbench "
        "(prospectivity index, not detected ice).",
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    lr = sub.add_parser("list-regions", help="List validated regions from the real catalog")
    lr.add_argument("--json", action="store_true", help="machine-readable output")
    lr.add_argument(
        "--observation-only",
        action="store_true",
        help="list tiles that have verified observations but NO prediction (not investigable)",
    )
    inv = sub.add_parser(
        "investigate",
        help="Run the four specialists (live model calls) and export a verified bundle",
    )
    inv.add_argument("--region", required=True, metavar="REGION_ID")
    inv.add_argument("--max-events", type=int, default=40)
    inv.add_argument("--max-hops", type=int, default=5)
    inv.add_argument("--max-tool-calls", type=int, default=16)
    inv.add_argument("--max-llm-calls", type=int, default=28)
    inv.add_argument("--timeout-s", type=float, default=420.0)
    inv.add_argument("--max-rounds", type=int, default=3)
    inv.add_argument("--max-input-tokens", type=int, default=150000)
    inv.add_argument(
        "--replay", metavar="RUN_ID", help="replay a saved run instead (no model calls)"
    )
    ver = sub.add_parser("verify", help="Verify a bundle offline (no API key, no GPU)")
    ver.add_argument("--run", required=True, metavar="RUN_ID_OR_PATH")
    ver.add_argument(
        "--source-package", metavar="PATH", help="external content-addressed source package"
    )
    ver.add_argument(
        "--local-source", action="store_true", help="rebuild the package from local data"
    )
    ver.add_argument("--json", action="store_true")
    ver.add_argument(
        "--write",
        action="store_true",
        help="refresh the bundle's verification.json with this result (not an integrity file)",
    )
    exp = sub.add_parser("export", help="Export a RECORDED run into a bundle (no model call)")
    exp.add_argument("--run", required=True, metavar="RUN_ID_OR_PATH")
    exp.add_argument("--out", metavar="NEW_RUN_ID")
    pub = sub.add_parser(
        "publish", help="Create a public-redacted bundle (no opaque provider data); no model call"
    )
    pub.add_argument("--run", required=True, metavar="RUN_ID_OR_PATH")
    pub.add_argument("--out", metavar="NEW_BUNDLE_ID")
    rep = sub.add_parser("replay", help="Replay a saved run from its events (no model call)")
    rep.add_argument("--run", required=True, metavar="RUN_ID_OR_PATH")
    tri = sub.add_parser(
        "triage",
        help="Deterministic research-triage screening of a bundle's source package (no model call)",
    )
    tri.add_argument("--run", required=True, metavar="RUN_ID_OR_PATH")
    tri.add_argument("--out", metavar="DIR", help="default: <runs>/_derived/<bundle>/triage")
    tri.add_argument("--no-write", action="store_true", help="print only")
    vt = sub.add_parser(
        "verify-triage", help="Recompute a triage artifact from the bundle's source (offline)"
    )
    vt.add_argument("--artifact", required=True, metavar="DIR_OR_FILE")
    vt.add_argument("--run", metavar="RUN_ID_OR_PATH", help="default: the bundle named inside")
    rv = sub.add_parser(
        "review", help="Targeted collaborative review of ONE screening window (LIVE, paid)"
    )
    rv.add_argument("--run", required=True, metavar="RUN_ID_OR_PATH")
    rv.add_argument("--window", type=int, required=True, metavar="WINDOW_ID")
    rv.add_argument(
        "--arm",
        choices=["single_generalist", "isolated_specialists", "communicating_specialists"],
        default="communicating_specialists",
    )
    rv.add_argument("--out", metavar="DIR", help="default: <runs>/_reviews/<new id>")
    rv.add_argument("--confirm-paid", action="store_true", help="required: real, paid OpenAI calls")
    ben = sub.add_parser(
        "benchmark", help="P7 paired-tile benchmark: build-suite, run, freeze, score, estimate"
    )
    ben.add_argument("--config", required=True, metavar="PATH")
    ben.add_argument(
        "--action",
        choices=["build-suite", "run", "freeze", "score", "estimate"],
        default="run",
    )
    ben.add_argument("--arms", nargs="*", metavar="ARM", help="subset of the config's arms")
    ben.add_argument("--eval-root", metavar="DIR", help="evaluator-only root (build/score)")
    ben.add_argument(
        "--confirm-paid",
        action="store_true",
        help="required to run LLM arms (real, paid OpenAI calls)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    if args.command in ("list-regions", "investigate", "review") or (
        args.command == "benchmark" and args.action in ("run", "build-suite")
    ):
        load_dotenv_file()  # IceClaw's own loader; values are never printed
    if args.command == "list-regions":
        return _list_regions(as_json=args.json, obs_only=args.observation_only)
    if args.command == "investigate":
        return _investigate(args)
    if args.command == "verify":
        return _verify(args)
    if args.command == "export":
        return _export(args)
    if args.command == "publish":
        return _publish(args)
    if args.command == "replay":
        return _replay(args.run)
    if args.command == "triage":
        return _triage(args)
    if args.command == "verify-triage":
        return _verify_triage(args)
    if args.command == "review":
        return _review(args)
    if args.command == "benchmark":
        return _benchmark(args)
    print(f"UNKNOWN_COMMAND: {args.command!r}", file=sys.stderr)  # argparse normally prevents this
    return 1


def _service():
    from iceclaw.services import InvestigationService

    return InvestigationService(load_settings())


def _investigate(args) -> int:
    from iceclaw.config import ConfigError
    from iceclaw.contracts import RunBudget

    if args.replay:
        return _replay(args.replay)
    svc = _service()
    budget = RunBudget(
        max_events=args.max_events, max_hops=args.max_hops, max_tool_calls=args.max_tool_calls,
        max_llm_calls=args.max_llm_calls, timeout_s=args.timeout_s,
    )  # fmt: skip
    s = svc.settings
    print(f"settings: {s.redacted()}")
    print(f"budget: {budget.model_dump()}; max input tokens {args.max_input_tokens}")
    try:
        res = svc.investigate(
            args.region, budget=budget, max_rounds_per_agent=args.max_rounds,
            max_input_tokens=args.max_input_tokens,
        )  # fmt: skip
    except ConfigError as e:
        print(f"LIVE MODE UNAVAILABLE: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # unknown / observation-only / prediction-only region etc.
        print(f"{getattr(e, 'code', type(e).__name__)}: {e}", file=sys.stderr)
        return 1
    sm = res.summary
    tokens = f"{sm['input_tokens']} in / {sm['output_tokens']} out / {sm['total_tokens']} total"
    print(
        f"run {res.run_id}: {sm['ended_reason']} in {sm['elapsed_s']}s; events {sm['events']}, "
        f"tool calls {sm['tool_calls']}, model calls {sm['llm_calls']}, tokens {tokens}; "
        f"cost {sm['cost']['status']}"
    )
    print(res.verification.render())
    print(f"bundle: {res.run_dir}")
    return 0 if res.ok else 3


def _verify(args) -> int:
    try:
        rep = _service().verify(
            args.run, source_package=args.source_package, local_source=args.local_source
        )
    except FileNotFoundError as e:
        print(f"RUN_NOT_FOUND: {e}", file=sys.stderr)
        return 2
    if args.json:
        import json

        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(rep.render())
    if args.write:
        import json

        target = _service().resolve_run(args.run) / "verification.json"
        target.write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8", newline="\n")
        print(f"wrote {target}")
    return 0 if rep.ok else 1


def _export(args) -> int:
    try:
        res = _service().export_recorded(args.run, out_id=args.out)
    except FileNotFoundError as e:
        print(f"RUN_NOT_FOUND: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"{getattr(e, 'code', type(e).__name__)}: {e}", file=sys.stderr)
        return 1
    print(res.verification.render())
    print(f"bundle: {res.run_dir}")
    return 0 if res.ok else 3


def _publish(args) -> int:
    import json

    try:
        res = _service().publish(args.run, out_id=args.out)
    except FileNotFoundError as e:
        print(f"RUN_NOT_FOUND: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"{getattr(e, 'code', type(e).__name__)}: {e}", file=sys.stderr)
        return 1
    m = json.loads((res.run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    red = m["transcript_redactions"]
    before, after = (
        red["private_transcript_bytes"],
        (res.run_dir / "llm_calls.jsonl").stat().st_size,
    )
    print(
        f"published {res.run_dir.name} from {red['source_bundle_dir']}: profile "
        f"{m['transcript_profile']}; {red['fields_redacted']} opaque field(s) "
        f"({red['characters_removed']} chars) redacted; transcript {before} -> {after} bytes"
    )
    print(res.verification.render())
    print(f"bundle: {res.run_dir}")
    return 0 if res.ok else 3


def _replay(run: str) -> int:
    try:
        r = _service().replay(run)
    except (FileNotFoundError, ValueError) as e:
        print(f"RUN_NOT_FOUND: {e}", file=sys.stderr)
        return 2
    print(r.header)
    for line in r.lines:
        print(line)
    print(f"-- end of recorded run ({r.n_events} events, {r.model_calls_made} model calls made) --")
    return 0


def _list_regions(as_json: bool, obs_only: bool = False) -> int:
    import json

    from iceclaw import services

    catalog = services.load_catalog(load_settings())
    rows = services.describe_regions(catalog)
    obs_rows = services.describe_observation_only(catalog)
    if obs_only:
        if as_json:
            print(json.dumps(obs_rows, indent=2))
        else:
            for o in obs_rows:
                print(
                    f"{o['region_id']}  status={o['status']}  split={o['source_split']}  "
                    f"tile={o['original_tile_id']}  prediction_ready=False"
                )
        return 0
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
                f"{r['region_id']}  status={r['status']}  pole={r['pole']}  "
                f"readiness={r['readiness']}  split={r['source_split']}  "
                f"source={r['source_mode']}"
            )
            print(
                f"    prediction_ready={r['prediction_ready']}  "
                f"observation_ready={r['observation_ready']}"
            )
            print(
                f"    centre=({lat:.4f}, {lon:.4f})  grid={r['grid'][0]}x{r['grid'][1]} "
                f"@ {r['spatial_resolution_m']} m/px  dataset_rev={r['source_revision'][:12]}"
            )
            print(f"    checkpoint={r['checkpoint'][:60]}...")
            units = r["modality_units"]
            print("    pixelwise: " + ", ".join(f"{k} [{units[k]}]" for k in units))
            if r["unit_unknown_layers_not_exposed"]:
                print(
                    "    downloaded but NOT exposed (unit unknown): "
                    + ", ".join(r["unit_unknown_layers_not_exposed"])
                )
            if r["pairing_error"]:
                print(f"    PAIRING NOT APPLIED: {r['pairing_error']}")
            for m in r["other_metadata"]:
                print(f"    {m['kind']}: {m['name']} = {m['value']} {m['unit'] or ''}".rstrip())
            print(f"    UNAVAILABLE: {', '.join(r['unavailable_modalities']) or 'none'}")
            if r["readiness_reason"]:
                print(f"    note: {r['readiness_reason']}")
    if obs_rows and not as_json:
        print(
            f"\n{len(obs_rows)} observation-only tile(s): verified SomBench inputs, NO NASA-IBM "
            "prediction (not investigable; use --observation-only to list)"
        )
    if not rows:
        print("NO_VALIDATED_REGIONS: no usable MOONATLAS export found.", file=sys.stderr)
        return 1
    return 0


def _benchmark(args) -> int:
    import json as _json
    from pathlib import Path

    from iceclaw import benchmark as bm

    try:
        cfg = bm.load_config(Path(args.config))
    except FileNotFoundError:
        print(f"CONFIG_NOT_FOUND: {args.config}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"BAD_CONFIG: {e}", file=sys.stderr)
        return 1
    arms = args.arms or cfg.arms
    print(f"config {args.config} sha256 {cfg.sha256()[:16]}; outputs {cfg.run_root}")
    if args.action == "build-suite":
        from iceclaw.evaluation import paired

        if not args.eval_root:
            print("--eval-root is required to build a suite", file=sys.stderr)
            return 1
        src = paired.load_real_paired_source(load_settings())
        extra = {}
        if bm.is_v2(cfg):  # the case list was pre-registered in the v2 config
            extra = {"fault_ids": bm.planned_fault_ids(cfg), "suite_version": "2"}
        m = paired.build_paired_suite(src, Path(cfg.scenarios_root), Path(args.eval_root),
                                      suite_id=cfg.suite_id, **extra)  # fmt: skip
        print(f"frozen suite {m.suite_id}: {len(m.cases)} cases; manifest {m.body_sha256()}")
        return 0
    if args.action == "estimate":
        print(_json.dumps(bm.budget_estimate(cfg, bm.planned_case_count(cfg)), indent=1))
        return 0
    if args.action == "run":
        llm = [a for a in arms if a != bm.BASELINE]
        if llm and not args.confirm_paid:
            est = bm.budget_estimate(cfg, len(cfg.resolved_cases()))
            print(
                "REFUSED: LLM arms make real, paid model calls. Upper bound for this run: "
                f"{_json.dumps(est['upper_bound'])}. Re-run with --confirm-paid.",
                file=sys.stderr,
            )
            return 2
        from iceclaw.config import ConfigError

        svc = _service()
        try:
            for a in arms:
                bm.run_arm(cfg, a, svc)
        except ConfigError as e:
            print(f"LIVE MODE UNAVAILABLE: {e}", file=sys.stderr)
            return 2
        return 0
    if args.action == "freeze":
        for a in arms:
            rec = bm.freeze_arm(cfg, a)
            print(f"frozen {a}: {rec['frozen']['sha256']}")
        return 0
    if not args.eval_root:
        print("--eval-root is required to score", file=sys.stderr)
        return 1
    out = bm.score(cfg, Path(args.eval_root))
    for a, r in out["arms"].items():
        print(
            f"{a}: n={r['n_cases']} TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']} "
            f"precision={r['precision']} recall={r['recall']} f1={r['f1']} "
            f"fpr={r['false_positive_rate']} missing={r['decision_missing']} "
            f"tokens={r['usage_total']['total_tokens']:.0f} "
            f"calls={r['usage_total']['llm_calls']:.0f}"
        )
    print(f"scores: {cfg.run_root / 'scores'}")
    return 0


def _embedded_source(run_dir):
    import json

    m = json.loads((run_dir / "input_manifest.json").read_text(encoding="utf-8"))
    return run_dir / m["source_package"]["embedded_path"]


def _triage(args) -> int:
    from pathlib import Path

    from iceclaw.reports import triage_artifact as ta
    from iceclaw.reports.package import PackageError
    from iceclaw.science.triage import TriageError

    svc = _service()
    try:
        run_dir = svc.resolve_run(args.run)
        art = ta.build_artifact(_embedded_source(run_dir), run_dir.name)
    except FileNotFoundError as e:
        print(f"RUN_NOT_FOUND: {e}", file=sys.stderr)
        return 2
    except (PackageError, TriageError, KeyError, OSError) as e:
        print(f"TRIAGE_FAILED: {getattr(e, 'code', type(e).__name__)}: {e}", file=sys.stderr)
        return 1
    t = art["triage"]
    c, sel = t["counts"], t["selection"]
    print(ta.LABEL)
    print(f"source: {art['source']['bundle']} (package {art['source']['package_sha256'][:16]}...)")
    print(
        f"windows {c['n_windows']} ({t['grid']['window_px']}x{t['grid']['window_px']} px); "
        f"index threshold (P{t['config']['index_percentile']:g}) {t['index_threshold_raw']}"
    )
    print(
        f"follow-up screening windows: {c['n_followup_windows']} ({c['n_joint_pixels']} of "
        f"{c['n_valid_all_layer_pixels']} pixels meet all four conditions); "
        f"model-observation review windows: {c['n_review_windows']}"
    )
    by_id = {w["id"]: w for w in t["windows"]}
    for i in sel["followup_ids"][:8]:
        w = by_id[i]
        print(
            f"  follow-up {w['label']} rows {w['window'][0]}-{w['window'][1]} cols "
            f"{w['window'][2]}-{w['window'][3]} ({w['centre_lat']:.3f}, {w['centre_lon']:.3f}) "
            f"joint pixels {w['n_joint']}"
        )
    for i in sel["review_ids"][:8]:
        w = by_id[i]
        why = ",".join(sel["review_reasons"][str(i)])
        print(
            f"  review    {w['label']} ({w['centre_lat']:.3f}, {w['centre_lon']:.3f}) weak: {why}"
        )
    for m in t["messages"]:
        print(f"NOTE: {m}")
    if not args.no_write:
        out = Path(args.out) if args.out else svc.settings.runs_root / "_derived" / run_dir.name
        out = out if args.out else out / "triage"
        print(f"wrote {ta.write_artifact(art, out)}")
    return 0


def _verify_triage(args) -> int:
    from iceclaw.reports import triage_artifact as ta

    try:
        art = ta.load_artifact(args.artifact)
        svc = _service()
        run_dir = svc.resolve_run(args.run or art["source"]["bundle"])
        rep = ta.verify_artifact(art, _embedded_source(run_dir))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"NOT_FOUND_OR_MALFORMED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(rep.render())
    return 0 if rep.ok else 1


def _review(args) -> int:
    from pathlib import Path

    from iceclaw import services
    from iceclaw.config import ConfigError

    svc = _service()
    budget = services.DEFAULT_REVIEW_BUDGET
    try:
        run_dir = svc.resolve_run(args.run)
        source = _embedded_source(run_dir)
    except FileNotFoundError as e:
        print(f"RUN_NOT_FOUND: {e}", file=sys.stderr)
        return 2
    print(
        f"review budget (TOTAL for this run): {budget.max_llm_calls} model calls, "
        f"{budget.max_tool_calls} tool calls, "
        f"{services.DEFAULT_REVIEW_MAX_INPUT_TOKENS} input tokens"
    )
    if not args.confirm_paid:
        print(
            "REFUSED: a targeted review makes real, paid model calls. Re-run with --confirm-paid.",
            file=sys.stderr,
        )
        return 2
    from datetime import UTC, datetime

    out = (
        Path(args.out)
        if args.out
        else svc.settings.runs_root / "_reviews" / f"rev-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    try:
        res = services.investigate_window(
            svc,
            source_dir=source,
            window_id=args.window,
            out_dir=out,
            source_label=run_dir.name,
            arm=args.arm,
        )
    except ConfigError as e:
        print(f"LIVE MODE UNAVAILABLE: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # unknown window, tampered source etc.
        print(f"{getattr(e, 'code', type(e).__name__)}: {e}", file=sys.stderr)
        return 1
    r, m = res.record, res.record["metrics"]
    print(
        f"review {r['review_id']}: {r['status']} ({m['n_reviews_filed']}/{m['n_agents']} reviews), "
        f"agreement {r['agreement']}; requests {m['n_test_requests']} "
        f"(answered {m['n_test_requests_answered']}), cross-agent citations "
        f"{m['n_cross_agent_citations']}; verification {m['n_measurements_reproduced']}/"
        f"{m['n_measurements_checked']}; calls {m['usage']['llm_calls']}, tokens "
        f"{m['usage']['total_tokens']}"
    )
    for reason in r["incomplete_reasons"]:
        print(f"INCOMPLETE: {reason}")
    print(f"wrote {out / 'review.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
