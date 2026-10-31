"""P7 composition runner: four arms on the frozen paired-tile suite, freeze, then score.

This module sits OUTSIDE ``iceclaw.evaluation`` (the evaluator never imports agents/swarm). It

1. loads each opaque case with ``evaluation.paired.load_paired_scenario`` from the scenarios root
   only (no evaluator root is needed or opened while systems run),
2. runs the deterministic baseline or one LLM arm per case through the services-level scenario
   adapter (fresh runtime, event store and model clients for every case and arm),
3. writes each case's original logs and its machine-readable decision,
4. freezes every arm's decisions and the SHA256 of every output file, and only then
5. scores the frozen outputs against the sealed key (``evaluation.score``).

Decision rule (pre-registered, identical for all LLM arms; see ``DECISION_RULE``).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from iceclaw.agents.benchmark_arms import COMMUNICATING, ISOLATED, LLM_ARMS, SINGLE
from iceclaw.contracts import RunBudget
from iceclaw.evaluation import paired
from iceclaw.evaluation.fixtures import canonical_json, load_manifest
from iceclaw.evaluation.score import FrozenDetections, freeze_detections, score_detections

FORMAT = "iceclaw-p7-benchmark/1"
BASELINE = "deterministic_rules"
ALL_ARMS = (BASELINE, SINGLE, ISOLATED, COMMUNICATING)
DECISION_RULE = (
    "A case is FLAGGED iff at least one agent's LATEST submit_assessment verdict is "
    "'fault_suspected' (every assessment cites >= 1 measurement event). If no agent filed an "
    "assessment the case is NOT flagged and is reported as decision_missing. The deterministic "
    "baseline flags a case iff any pre-registered rule triggers (PairedBaselineConfig)."
)
CASE_FILES = (
    "decision.json",
    "events.jsonl",
    "llm_calls.jsonl",
    "run_summary.json",
    "tool_runs.json",
)


class BenchmarkConfig(BaseModel):
    """Frozen benchmark plan. Its SHA256 names the output directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    format: str = FORMAT
    suite_id: str = paired.SUITE_ID
    scenarios_root: str
    output_root: str
    arms: list[str]
    case_ids: list[str]  # empty = every detection case of the suite
    budget: RunBudget  # TOTAL per case and arm (never multiplied by the number of agents)
    max_input_tokens: int
    provider: str = "openai"
    model: str
    decision_rule: str = DECISION_RULE
    baseline_config: dict[str, Any] = asdict(paired.PairedBaselineConfig())
    note: str = ""

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.model_dump(mode="json")).encode()).hexdigest()

    @property
    def run_root(self) -> Path:
        return Path(self.output_root) / f"{self.suite_id}-{self.sha256()[:12]}"

    def resolved_cases(self) -> list[str]:
        manifest = load_manifest(Path(self.scenarios_root), self.suite_id)
        ids = self.case_ids or list(manifest.detection_case_ids)
        unknown = sorted(set(ids) - set(manifest.detection_case_ids))
        if unknown:
            raise ValueError(f"unknown case ids {unknown}")
        return ids


# =========================================================================== P7 protocol v2
FORMAT_V2 = "iceclaw-p7-benchmark/2"
FLAGGED, NOT_FLAGGED, INCOMPLETE = "FLAGGED", "NOT_FLAGGED", "INCOMPLETE"
DECISION_RULE_V2 = (
    "Individual verdict per agent of the arm: 'fault_suspected' or 'no_fault_found' = that agent's "
    "LATEST submit_assessment (citing >= 1 measurement); 'incomplete' = the agent filed none. "
    "Team decision: FLAGGED iff at least one individual verdict is fault_suspected; NOT_FLAGGED iff "
    "every agent of the arm filed no_fault_found; otherwise INCOMPLETE (no alarm raised and at "
    "least one agent incomplete). PRIMARY metrics count INCOMPLETE as not flagged (an arm gets no "
    "credit for a verdict it did not produce); incomplete cases and incomplete individual verdicts "
    "are reported separately, and SECONDARY metrics are computed over complete cases only. The "
    "deterministic baseline is FLAGGED iff any pre-registered rule triggers (never incomplete)."
)


class ReservePlan(BaseModel):
    """Per-agent guaranteed minimum inside the SAME total budget (multi-agent arms only)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    llm_calls_per_agent: int
    tool_calls_per_agent: int
    input_tokens_per_agent: int
    assessment_events_per_agent: int = 1
    applies_to: list[str] = [ISOLATED, COMMUNICATING]
    release_rule: str = (
        "an agent's unused reserve is withheld from the other agents until it files its first "
        "assessment or exits, then returns to the common pool; an agent that has not assessed "
        "receives one final turn before the run may end"
    )

    def runtime_reserve(self):
        from iceclaw.swarm.runtime import AgentReserve

        return AgentReserve(
            llm_calls=self.llm_calls_per_agent,
            tool_calls=self.tool_calls_per_agent,
            input_tokens=self.input_tokens_per_agent,
            assessment_events=self.assessment_events_per_agent,
        )


class BenchmarkConfigV2(BenchmarkConfig):
    """Protocol v2 (new class, so v1 config hashes and output folders are unchanged)."""

    format: str = FORMAT_V2
    suite_id: str = "p7-paired-v2"
    decision_rule: str = DECISION_RULE_V2
    reserve: ReservePlan
    preregistration: dict[str, Any] = {}


def _validate_v2(cfg: BenchmarkConfigV2) -> None:
    r, b, n = cfg.reserve, cfg.budget, 4  # four specialists in the multi-agent arms
    checks = [
        (r.llm_calls_per_agent * n, b.max_llm_calls, "model calls"),
        (r.tool_calls_per_agent * n, b.max_tool_calls, "tool calls"),
        (r.input_tokens_per_agent * n, cfg.max_input_tokens, "input tokens"),
        (r.assessment_events_per_agent * n, b.max_events, "assessment events"),
    ]
    for reserved, total, name in checks:
        if reserved > total:
            raise ValueError(f"reserved {name} ({reserved}) exceed the shared total ({total})")
    if not set(r.applies_to) <= {ISOLATED, COMMUNICATING}:
        raise ValueError("reserves apply only to the four-agent arms")


def is_v2(cfg: BenchmarkConfig) -> bool:
    return isinstance(cfg, BenchmarkConfigV2)


def decide_v2(assessments: dict[str, dict[str, Any]], agent_ids: list[str]) -> dict[str, Any]:
    """Apply ``DECISION_RULE_V2`` over ALL agents of the arm (missing = incomplete)."""
    individual = {a: assessments[a]["verdict"] if a in assessments else "incomplete"
                  for a in sorted(agent_ids)}  # fmt: skip
    vals = list(individual.values())
    if "fault_suspected" in vals:
        team = FLAGGED
    elif all(v == "no_fault_found" for v in vals):
        team = NOT_FLAGGED
    else:
        team = INCOMPLETE
    layers = sorted({x for v in assessments.values() if v["verdict"] == "fault_suspected"
                     for x in v.get("suspected_layers", [])})  # fmt: skip
    return {
        "team_decision": team,
        "flagged": team == FLAGGED,
        "decision_missing": team == INCOMPLETE,
        "individual_verdicts": individual,
        "n_incomplete_verdicts": vals.count("incomplete"),
        "n_agents": len(vals),
        "verdicts": {a: v for a, v in individual.items() if v != "incomplete"},
        "suspected_layers": layers,
    }


def load_config(path: Path) -> BenchmarkConfig:
    raw = Path(path).read_text(encoding="utf-8")
    if json.loads(raw).get("format") == FORMAT_V2:
        cfg: BenchmarkConfig = BenchmarkConfigV2.model_validate_json(raw)
        _validate_v2(cfg)  # type: ignore[arg-type]
    else:
        cfg = BenchmarkConfig.model_validate_json(raw)
    bad = sorted(set(cfg.arms) - set(ALL_ARMS))
    if bad:
        raise ValueError(f"unknown arms {bad}; choose from {ALL_ARMS}")
    if cfg.baseline_config != asdict(paired.PairedBaselineConfig()):
        raise ValueError("baseline_config differs from the pre-registered PairedBaselineConfig")
    return cfg


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------ decisions
def decide(assessments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply ``DECISION_RULE`` to the latest assessment of each agent."""
    verdicts = {a: v["verdict"] for a, v in sorted(assessments.items())}
    layers = sorted({x for v in assessments.values() for x in v.get("suspected_layers", [])
                     if v["verdict"] == "fault_suspected"})  # fmt: skip
    return {
        "flagged": any(v == "fault_suspected" for v in verdicts.values()),
        "decision_missing": not verdicts,
        "n_agents_assessed": len(verdicts),
        "verdicts": verdicts,
        "suspected_layers": layers,
    }


def _write(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


def run_baseline_case(cfg: BenchmarkConfig, case_id: str, out_dir: Path) -> dict[str, Any]:
    scen = paired.load_paired_scenario(Path(cfg.scenarios_root), cfg.suite_id, case_id)
    t0 = time.perf_counter()
    res = paired.run_paired_baseline(scen, paired.PairedBaselineConfig())
    elapsed = time.perf_counter() - t0
    out_dir.mkdir(parents=True, exist_ok=False)
    rules = [asdict(r) for r in res.rules]
    _write(out_dir / "tool_runs.json", list(res.tool_run_ids))
    (out_dir / "events.jsonl").write_text("", encoding="utf-8")
    (out_dir / "llm_calls.jsonl").write_text("", encoding="utf-8")
    summary = {"ended_reason": "DETERMINISTIC", "elapsed_s": round(elapsed, 4),
               "tool_calls": len(res.tool_run_ids), "llm_calls": 0, "input_tokens": 0,
               "output_tokens": 0, "total_tokens": 0, "events": 0, "rules": rules}  # fmt: skip
    _write(out_dir / "run_summary.json", summary)
    decision = {
        "arm": BASELINE,
        "case_id": case_id,
        "flagged": res.flagged,
        "decision_missing": False,
        **(
            {"team_decision": FLAGGED if res.flagged else NOT_FLAGGED, "n_incomplete_verdicts": 0}
            if is_v2(cfg)
            else {}
        ),
        "triggered_rules": res.triggered_rules,
        "usage": {
            k: summary[k]
            for k in (
                "elapsed_s",
                "tool_calls",
                "llm_calls",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "events",
            )
        },  # fmt: skip
        "ended_reason": "DETERMINISTIC",
    }
    _write(out_dir / "decision.json", decision)
    return decision


def run_llm_case(service: Any, cfg: BenchmarkConfig, arm: str, case_id: str, out_dir: Path):
    from iceclaw.services import investigate_scenario

    scen = paired.load_paired_scenario(Path(cfg.scenarios_root), cfg.suite_id, case_id)
    reserve = None
    if is_v2(cfg) and arm in cfg.reserve.applies_to:  # type: ignore[attr-defined]
        reserve = cfg.reserve.runtime_reserve()  # type: ignore[attr-defined]
    res = investigate_scenario(
        service, scen, arm=arm, budget=cfg.budget, max_input_tokens=cfg.max_input_tokens,
        out_dir=out_dir, reserve=reserve,
    )  # fmt: skip
    s = res.summary
    if is_v2(cfg):
        from iceclaw.agents.benchmark_arms import arm_plan

        agent_ids = [sp.agent_id for sp in arm_plan(arm, cfg.budget.max_llm_calls).specs]
        verdict = decide_v2(res.assessments, agent_ids)
    else:
        verdict = decide(res.assessments)
    decision = {
        "arm": arm,
        "case_id": case_id,
        "investigation_id": res.investigation_id,
        **verdict,
        "reserve": None if reserve is None else asdict(reserve),
        "assessments": res.assessments,
        "assessment_history": res.assessment_history,
        "usage": {
            "elapsed_s": s["elapsed_s"],
            "tool_calls": s["tool_calls"],
            "llm_calls": s["llm_calls"],
            "input_tokens": s["input_tokens"],
            "output_tokens": s["output_tokens"],
            "total_tokens": s["total_tokens"],
            "events": s["events"],
        },
        "ended_reason": s["ended_reason"],
        "exit_reasons": s["exit_reasons"],
        "tokens_by_agent": s.get("tokens_by_agent", {}),
        "provider": s.get("provider"),
        "model": s.get("model"),
    }
    _write(out_dir / "decision.json", decision)
    return decision


def run_arm(cfg: BenchmarkConfig, arm: str, service: Any = None, log=print) -> list[dict[str, Any]]:
    """Run ``arm`` on every configured case (completed cases are kept, never re-run)."""
    if arm not in cfg.arms:
        raise ValueError(f"{arm!r} is not an arm of this config")
    root = cfg.run_root
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "config.json"
    if not cfg_path.exists():
        cfg_path.write_text(canonical_json(cfg.model_dump(mode="json")), encoding="utf-8")
    if (root / "frozen" / f"{arm}.json").exists():
        raise RuntimeError(f"{arm} is already frozen; outputs are immutable")
    if arm != BASELINE:
        if service is None:
            raise ValueError("an InvestigationService is required for LLM arms")
        if service.client_factory is None:
            st = service.settings
            st.require_live_llm()
            if (st.llm_provider, st.llm_model) != (cfg.provider, cfg.model):
                raise RuntimeError(
                    f"configured model {st.llm_provider}/{st.llm_model} differs from the "
                    f"benchmark config {cfg.provider}/{cfg.model}"
                )
    out = []
    for cid in cfg.resolved_cases():
        d = root / arm / cid
        if (d / "decision.json").exists():
            out.append(json.loads((d / "decision.json").read_text()))
            continue
        if d.exists():
            raise RuntimeError(f"{d} exists without a decision (interrupted run): inspect it")
        log(f"[{arm}] {cid} ...")
        dec = (
            run_baseline_case(cfg, cid, d)
            if arm == BASELINE
            else run_llm_case(service, cfg, arm, cid, d)
        )
        u = dec["usage"]
        log(
            f"[{arm}] {cid}: flagged={dec['flagged']} missing={dec['decision_missing']} "
            f"calls={u['llm_calls']} tokens={u['total_tokens']} tools={u['tool_calls']} "
            f"{u['elapsed_s']}s"
        )
        out.append(dec)
    return out


# ------------------------------------------------------------------------------- freezing
def freeze_arm(cfg: BenchmarkConfig, arm: str) -> dict[str, Any]:
    """Freeze decisions AND the hash of every original output file before any scoring."""
    root = cfg.run_root
    path = root / "frozen" / f"{arm}.json"
    if path.exists():
        raise RuntimeError(f"{arm} already frozen")
    manifest = load_manifest(Path(cfg.scenarios_root), cfg.suite_id)
    cases = cfg.resolved_cases()
    flags, files = {}, {}
    for cid in cases:
        d = root / arm / cid
        dec = json.loads((d / "decision.json").read_text())
        flags[cid] = bool(dec["flagged"])
        files[cid] = {n: sha256_file(d / n) for n in CASE_FILES}
    subset = None if set(cases) == set(manifest.detection_case_ids) else cases
    frozen = freeze_detections(manifest, flags, subset)
    record = {
        "arm": arm,
        "config_sha256": cfg.sha256(),
        "frozen_utc": datetime.now(UTC).isoformat(),
        "frozen": {
            "suite_manifest_sha256": frozen.suite_manifest_sha256,
            "flags": [list(x) for x in frozen.flags],
            "sha256": frozen.sha256,
            "declared_subset": None
            if frozen.declared_subset is None
            else list(frozen.declared_subset),
        },
        "output_file_sha256": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, record)
    return record


def _load_frozen(cfg: BenchmarkConfig, arm: str) -> tuple[FrozenDetections, dict[str, Any]]:
    rec = json.loads((cfg.run_root / "frozen" / f"{arm}.json").read_text())
    if rec["config_sha256"] != cfg.sha256():
        raise RuntimeError("frozen record belongs to a different config")
    for cid, names in rec["output_file_sha256"].items():  # outputs unchanged since freezing?
        for n, h in names.items():
            if sha256_file(cfg.run_root / arm / cid / n) != h:
                raise RuntimeError(f"{arm}/{cid}/{n} changed after freezing")
    f = rec["frozen"]
    frozen = FrozenDetections(
        f["suite_manifest_sha256"],
        tuple((c, bool(v)) for c, v in f["flags"]),
        f["sha256"],
        None if f["declared_subset"] is None else tuple(f["declared_subset"]),
    )
    return frozen, rec


# -------------------------------------------------------------------------------- scoring
def score(cfg: BenchmarkConfig, eval_root: Path) -> dict[str, Any]:
    """Score every FROZEN arm (the sealed key is opened here, never earlier)."""
    root = cfg.run_root
    arms, rows = {}, []
    for arm in cfg.arms:
        if not (root / "frozen" / f"{arm}.json").exists():
            continue
        frozen, _ = _load_frozen(cfg, arm)
        res = score_detections(frozen, Path(cfg.scenarios_root), Path(eval_root), cfg.suite_id)
        usage = {k: 0.0 for k in ("elapsed_s", "tool_calls", "llm_calls", "input_tokens",
                                  "output_tokens", "total_tokens", "events")}  # fmt: skip
        missing = 0
        incomplete_verdicts = 0
        complete_flags: dict[str, bool] = {}
        for pc in res["per_case"]:
            dec = json.loads((root / arm / pc["case_id"] / "decision.json").read_text())
            missing += bool(dec.get("decision_missing"))
            incomplete_verdicts += int(dec.get("n_incomplete_verdicts", 0))
            if dec.get("team_decision", FLAGGED) != INCOMPLETE:
                complete_flags[pc["case_id"]] = bool(dec["flagged"])
            for k in usage:
                usage[k] += float(dec["usage"][k])
            rows.append({
                "arm": arm, "case_id": pc["case_id"], "fault_id": pc["fault_id"],
                "is_fault": pc["is_fault"], "flagged": pc["flagged"], "outcome": pc["outcome"],
                "decision_missing": bool(dec.get("decision_missing")),
                **({"team_decision": dec.get("team_decision"),
                    "individual_verdicts": json.dumps(dec.get("individual_verdicts", {}), sort_keys=True),
                    "n_incomplete_verdicts": dec.get("n_incomplete_verdicts", 0)} if is_v2(cfg) else {}),
                "verdicts": json.dumps(dec.get("verdicts", {}), sort_keys=True),
                "suspected_layers": ";".join(dec.get("suspected_layers", [])),
                "triggered_rules": ";".join(dec.get("triggered_rules", [])),
                "ended_reason": dec.get("ended_reason"),
                **{k: dec["usage"][k] for k in usage},
            })  # fmt: skip
        n = len(res["per_case"])
        arms[arm] = {
            **{k: res[k] for k in ("n_cases", "n_fault_cases", "n_clean_cases", "tp", "fp",
                                   "fn", "tn", "precision", "recall", "f1",
                                   "false_positive_rate", "declared_subset", "frozen_sha256")},
            "decision_missing": missing,
            "usage_total": usage,
            "usage_mean_per_case": {k: (v / n if n else None) for k, v in usage.items()},
            "expectation_check": res["expectation_check"] if arm == BASELINE else None,
        }  # fmt: skip
        if is_v2(cfg):
            from iceclaw.evaluation.score import detection_metrics

            labels = {pc["case_id"]: bool(pc["is_fault"]) for pc in res["per_case"]}
            arms[arm]["incomplete_cases"] = missing
            arms[arm]["incomplete_individual_verdicts"] = incomplete_verdicts
            arms[arm]["secondary_complete_cases_only"] = (
                detection_metrics(complete_flags, {c: labels[c] for c in complete_flags})
                if complete_flags
                else None
            )
    out = {
        "format": cfg.format,
        "config_sha256": cfg.sha256(),
        "suite_id": cfg.suite_id,
        "scored_utc": datetime.now(UTC).isoformat(),
        "decision_rule": cfg.decision_rule,
        "budget_per_case_and_arm": {
            **cfg.budget.model_dump(),
            "max_input_tokens": cfg.max_input_tokens,
        },
        "arms": arms,
        "caveats": CAVEATS,
    }
    (root / "scores").mkdir(exist_ok=True)
    _write(root / "scores" / "scores.json", out)
    (root / "scores" / "case_outcomes.csv").write_text(_csv(rows), encoding="utf-8")
    summary_rows = [
        {"arm": a, **{k: v for k, v in s.items() if k not in ("usage_total", "usage_mean_per_case",
                                                              "expectation_check", "declared_subset",
                                                              "secondary_complete_cases_only")},
         **{f"total_{k}": v for k, v in s["usage_total"].items()}}
        for a, s in arms.items()
    ]  # fmt: skip
    (root / "scores" / "arm_summary.csv").write_text(_csv(summary_rows), encoding="utf-8")
    return out


CAVEATS = [
    "All cases are copies of ONE genuine paired tile (ice-s-0001-0002); cases share pixels and are "
    "not independent samples. Results do not generalize to other regions or to the Moon.",
    "Faults are synthetic corruptions of genuine data, not real lunar anomalies.",
    "The statistical unit is the case, never the pixel; no confidence intervals are reported.",
    "Metrics over a declared subset (e.g. a one-case pilot) are feasibility checks, not a "
    "performance comparison.",
    "Token counts are provider-reported; no price is applied (unpriced).",
    "Fault expectations were pre-registered by the author of the deterministic baseline; they are "
    "a plumbing check, not an independent difficulty estimate.",
]


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    keys = list(rows[0])
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in keys})
    return buf.getvalue()


# ------------------------------------------------------------------------- budget estimate
def budget_estimate(cfg: BenchmarkConfig, n_cases: int, observed: dict[str, dict] | None = None):
    """Upper bound (caps) and, if a pilot was scored, a measured extrapolation per LLM arm."""
    llm_arms = [a for a in cfg.arms if a in LLM_ARMS]
    cap = {
        "model_calls": cfg.budget.max_llm_calls * n_cases * len(llm_arms),
        "input_tokens_cap": cfg.max_input_tokens * n_cases * len(llm_arms),
        "note": "input cap is enforced per case by pre-call reservation (estimate-based); output "
        "tokens are additional (soft)",
    }
    measured = {}
    for a in llm_arms:
        m = (observed or {}).get(a)
        if m:
            per = m["usage_mean_per_case"]
            measured[a] = {
                "per_case": per,
                "projected_model_calls": per["llm_calls"] * n_cases,
                "projected_total_tokens": per["total_tokens"] * n_cases,
                "projected_elapsed_s": per["elapsed_s"] * n_cases,
            }
    return {
        "n_cases": n_cases,
        "llm_arms": llm_arms,
        "upper_bound": cap,
        "pilot_projection": measured,
    }


def planned_fault_ids(cfg: BenchmarkConfig) -> tuple[str, ...]:
    """The pre-registered case list of a v2 config (validated against the code registry)."""
    ids = tuple((getattr(cfg, "preregistration", {}) or {}).get("fault_ids", ()))
    if ids != paired.PAIRED_V2_FAULT_IDS:
        raise ValueError("v2 pre-registered fault_ids differ from paired.PAIRED_V2_FAULT_IDS")
    return ids


def planned_case_count(cfg: BenchmarkConfig) -> int:
    """Number of cases a config will run; usable BEFORE the suite exists (v2 pre-registration)."""
    if cfg.case_ids:
        return len(cfg.case_ids)
    if is_v2(cfg):
        return len(planned_fault_ids(cfg))
    return len(cfg.resolved_cases())
