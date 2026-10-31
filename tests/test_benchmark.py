"""P7: paired-tile suite, scenario adapter, isolation, equal budgets, freeze-then-score.

Offline only: SYNTHETIC 4-layer arrays and scripted fake models (no API key, no model call).
These tests prove the plumbing; they are not benchmark results. A real-data test at the end
builds a throwaway copy of the paired suite from the genuine tile and skips if it is absent.
"""

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from agent_helpers import (
    AUDITOR,
    GEOLOGY,
    PRED,
    SHADOW,
    SKEPTIC,
    TEMP,
    TERRAIN,
    THERMAL,
    Script,
    call,
    first_event_id,
    synthetic_arrays,
)

from iceclaw import benchmark as bm
from iceclaw.agents.benchmark_arms import COMMUNICATING, GENERALIST, ISOLATED, SINGLE, arm_plan
from iceclaw.agents.provider import ScriptedChatClient
from iceclaw.cli import main
from iceclaw.config import Settings
from iceclaw.contracts import RunBudget
from iceclaw.evaluation import fixtures
from iceclaw.evaluation import paired as pr
from iceclaw.evaluation.score import freeze_detections, score_detections
from iceclaw.services import InvestigationService, ScenarioRejected, check_scenario
from iceclaw.swarm.runtime import RunLimits, SwarmRuntime
from iceclaw.swarm.store import EvidenceStore

SALT = b"\x01" * 16
REPO = Path(__file__).resolve().parents[1]
BUDGET = RunBudget(max_events=48, max_hops=5, max_tool_calls=16, max_llm_calls=16, timeout_s=60)


def synthetic_source(n: int = 64) -> pr.PairedSource:
    return pr.paired_source_from_arrays(
        synthetic_arrays(n), "south", {"source_mode": "upstream_published_example",
                                        "source_dataset": "SYNTHETIC", "source_revision": "0"}
    )  # fmt: skip


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("p7")
    scen, ev = root / "scen", root / "eval"
    m = pr.build_paired_suite(synthetic_source(), scen, ev, suite_seed=7, salt=SALT)
    return m, scen, ev


def cfg_for(scen: Path, out: Path, arms, cases=(), **kw) -> bm.BenchmarkConfig:
    return bm.BenchmarkConfig(
        scenarios_root=str(scen), output_root=str(out), arms=list(arms), case_ids=list(cases),
        budget=kw.pop("budget", BUDGET), max_input_tokens=kw.pop("max_input_tokens", 200_000),
        provider="scripted", model="scripted-v0", **kw,
    )  # fmt: skip


# ======================================================================= suite + firewall
def test_suite_has_four_clean_controls_and_eight_faults_on_one_tile(suite):
    m, scen, ev = suite
    sealed = fixtures.load_sealed(ev, m)
    assert len(m.cases) == 12 and m.detection_case_ids == [f"case-{i:02d}" for i in range(1, 13)]
    labels = sealed.labels
    assert sum(labels.values()) == 8 and sum(not v for v in labels.values()) == 4
    assert {c["fault_id"] for c in sealed.cases.values()} == set(pr.PAIRED_V1_FAULT_IDS)
    assert m.independence["independent_source_tiles"] == 1
    assert pr.verify_paired_suite(scen, ev) == []


def test_agent_visible_files_carry_no_truth_seed_salt_or_pro(suite):
    m, scen, ev = suite
    sealed = fixtures.load_sealed(ev, m)
    secrets_ = {c["fault_id"] for c in sealed.cases.values()} | {str(ev), "synthetic-region"}
    text = (scen / m.suite_id / "manifest.json").read_text()
    for tok in sorted(secrets_) + ["is_fault", "expected_detectable", "suite_seed", "salt"]:
        assert tok not in text, tok
    blob = []
    for c in m.cases:
        d = scen / m.suite_id / c.case_id
        assert sorted(p.name for p in d.iterdir()) == sorted(c.files)
        blob.append((d / "meta.json").read_text())
        s = pr.load_paired_scenario(scen, m.suite_id, c.case_id)
        blob += [json.dumps(s.public_dict()), s.context.model_dump_json(), repr(s.arrays.layers)]
        assert set(s.arrays.layers) == set(pr.LAYER_ORDER)
    joined = "\n".join(blob).lower()
    for tok in sorted(secrets_) + ["reference", "sealed", "salt", "seed", "fault", "expected_"]:
        assert tok.lower() not in joined, tok
    assert not re.search(r"\bPRO\b", "\n".join(blob))
    for root in (scen, ev):
        assert not [p for p in Path(root).rglob("*") if "PRO" in re.split(r"[._]", p.name)]


def test_suite_is_frozen_reproducible_and_tamper_evident(tmp_path):
    src = synthetic_source()
    a = pr.build_paired_suite(src, tmp_path / "s1", tmp_path / "e1", suite_seed=3, salt=SALT)
    b = pr.build_paired_suite(src, tmp_path / "s2", tmp_path / "e2", suite_seed=3, salt=SALT)
    assert a.body_sha256() == b.body_sha256()
    with pytest.raises(fixtures.SuiteAlreadyFrozen):
        pr.build_paired_suite(src, tmp_path / "s1", tmp_path / "e1", suite_seed=3, salt=SALT)
    f = tmp_path / "s1" / pr.SUITE_ID / "case-03" / f"{TEMP}.values.npy"
    raw = bytearray(f.read_bytes())
    raw[-1] ^= 1
    f.write_bytes(bytes(raw))
    assert any("case-03" in p for p in pr.verify_paired_suite(tmp_path / "s1", tmp_path / "e1"))
    # default seed is private and random: two default builds differ
    c = pr.build_paired_suite(src, tmp_path / "s3", tmp_path / "e3", salt=SALT)
    d = pr.build_paired_suite(src, tmp_path / "s4", tmp_path / "e4", salt=SALT)
    assert c.sealed_commitment != d.sealed_commitment


def test_source_arrays_are_read_only_and_faults_act_on_copies():
    src = synthetic_source()
    before = {n: v.copy() for n, (v, _) in src.case.layers.items()}
    for fid in pr.PAIRED_V1_FAULT_IDS:
        pr.apply_paired_fault(fid, src.case, 11)
    for n, (v, _) in src.case.layers.items():
        assert np.array_equal(v, before[n], equal_nan=True)
        assert not v.flags.writeable


def test_p3_and_p4_suites_are_untouched_by_p7():
    assert fixtures.SUITE_ID == "p3-prediction-only-v1" and pr.SUITE_ID == "p7-paired-v1"
    from iceclaw.evaluation import multimodal

    assert multimodal.SUITE_ID == "p4-multimodal-v1"
    assert multimodal.MULTIMODAL_V1_FAULT_IDS[0] == "clean_identical"


# ============================================================================ baseline
def test_baseline_is_preregistered_and_flags_obvious_faults_on_synthetic_data(suite, tmp_path):
    m, scen, ev = suite
    cfg = cfg_for(scen, tmp_path / "out", [bm.BASELINE])
    bm.run_arm(cfg, bm.BASELINE)
    bm.freeze_arm(cfg, bm.BASELINE)
    out = bm.score(cfg, ev)
    r = out["arms"][bm.BASELINE]
    assert r["n_cases"] == 12 and r["declared_subset"] is None
    sealed = fixtures.load_sealed(ev, m)
    by_fault = {
        sealed.cases[c]["fault_id"]: json.loads((cfg.run_root / bm.BASELINE / c / "decision.json").read_text())["flagged"]
        for c in m.detection_case_ids
    }  # fmt: skip
    for fid in ("prediction_percent_scale", "prediction_nan_block", "temperature_unit_celsius",
                "shadow_mask_inverted", "slope_unit_radians"):  # fmt: skip
        assert by_fault[fid], fid
    for fid in ("clean_identical", "clean_identical_repeat", "clean_float32_roundtrip",
                "clean_subthreshold_noise", "prediction_offset_small", "prediction_misregistered"):  # fmt: skip
        assert not by_fault[fid], fid
    assert r["usage_total"]["llm_calls"] == 0 and r["usage_total"]["tool_calls"] > 0
    assert (cfg.run_root / "scores" / "case_outcomes.csv").read_text().count("\n") == 13
    assert "ONE genuine paired tile" in out["caveats"][0]


# ======================================================================= scenario adapter
def test_adapter_accepts_only_opaque_well_formed_cases(suite):
    m, scen, _ = suite
    s = pr.load_paired_scenario(scen, m.suite_id, "case-01")
    ctx, arr = check_scenario(s)
    assert ctx.region_id == arr.region_id == "case-01"

    class Bad:
        def __init__(self, **kw):
            self.case_id, self.context, self.arrays = s.case_id, s.context, s.arrays
            self.__dict__.update(kw)

    with pytest.raises(ScenarioRejected):
        check_scenario(Bad(case_id="ice-s-0001-0002"))  # real tile ids are not opaque
    with pytest.raises(ScenarioRejected):
        check_scenario(Bad(case_id="case-02"))  # id mismatch with context/arrays
    with pytest.raises(ScenarioRejected):
        check_scenario(Bad(context=s.context.model_copy(update={"source_ids": ["/data/x.tif"]})))
    lay = s.arrays.layers[TEMP]
    from dataclasses import replace

    bad_layers = {**s.arrays.layers, TEMP: replace(lay, file_id="/abs/path/TMAX.tif")}
    with pytest.raises(ScenarioRejected):
        check_scenario(Bad(arrays=replace(s.arrays, layers=bad_layers)))
    bad_unit = {**s.arrays.layers, TEMP: replace(lay, unit="C")}
    with pytest.raises(ScenarioRejected):
        check_scenario(Bad(arrays=replace(s.arrays, layers=bad_unit)))
    with pytest.raises(ScenarioRejected):
        check_scenario(Bad(context={"region_id": "case-01"}))  # not a validated context


# ============================================================ runtime isolation primitives
def iso_runtime(communication: bool) -> SwarmRuntime:
    from agent_helpers import context_for

    arrays = synthetic_arrays()
    plan = arm_plan(ISOLATED if not communication else COMMUNICATING, 16)
    return SwarmRuntime(
        investigation_id="inv-iso", context=context_for(arrays, BUDGET), arrays=arrays,
        store=EvidenceStore(), specs=list(plan.specs), limits=RunLimits(budget=BUDGET),
        communication=communication, require_assessment=True,
    )  # fmt: skip


def test_isolated_agents_cannot_see_cite_or_contact_each_other():
    rt = iso_runtime(False)
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    assert m["ok"]
    assert rt.state[GEOLOGY].inbox == [] and rt.state[SKEPTIC].inbox == []  # nothing delivered
    assert rt.list_events(GEOLOGY) == [] and rt.list_events(THERMAL)
    assert rt.get_event_detail(m["event_id"], GEOLOGY)["error"] == "UNKNOWN_EVENT"
    assert rt.publish_claim(GEOLOGY, "x", [m["event_id"]])["error"] == "UNKNOWN_EVENT"
    assert (
        rt.request_test(SKEPTIC, THERMAL, "quantiles", "why", TEMP)["error"]
        == "COMMUNICATION_DISABLED"
    )
    c = rt.publish_claim(THERMAL, "PSR colder", [m["event_id"]])
    assert rt.publish_challenge(SKEPTIC, c["event_id"], "no")["error"] == "COMMUNICATION_DISABLED"
    assert rt.recompute_measurement(SKEPTIC, m["tool_run_id"])["error"] == "UNKNOWN_TOOL_RUN"
    # an identical measurement made earlier by another agent is NOT free in isolation
    before = rt.tool_calls
    again = rt.science_tool(SKEPTIC, "thermal_psr_consistency")
    assert again["cached"] and rt.tool_calls == before + 1
    assert rt.get_event_detail(again["event_id"], SKEPTIC)["ok"]  # granted: it asked for it
    from agent_framework import FunctionTool  # noqa: F401  (tool objects come from MAF)

    from iceclaw.agents.tools import make_tools

    names = {t.name for t in make_tools(rt, rt.specs[THERMAL])}
    assert "request_test" not in names and "publish_challenge" not in names
    assert "submit_assessment" in names


def test_communicating_agents_keep_the_p5_event_bus():
    rt = iso_runtime(True)
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    rt.publish_claim(THERMAL, "PSR colder", [m["event_id"]])
    assert rt.state[SKEPTIC].inbox and rt.state[SKEPTIC].wake_pending
    assert rt.get_event_detail(m["event_id"], GEOLOGY)["ok"]
    from iceclaw.agents.tools import make_tools

    assert {"request_test", "publish_challenge"} <= {
        t.name for t in make_tools(rt, rt.specs[SKEPTIC])
    }


def test_assessment_is_validated_and_latest_wins():
    rt = iso_runtime(True)
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    bad = [
        rt.submit_assessment(THERMAL, "maybe", [TEMP], [m["event_id"]], "x"),
        rt.submit_assessment(THERMAL, "fault_suspected", [], [m["event_id"]], "x"),
        rt.submit_assessment(THERMAL, "fault_suspected", ["bogus"], [m["event_id"]], "x"),
        rt.submit_assessment(THERMAL, "no_fault_found", [], [], "x"),
    ]
    assert [b["error"] for b in bad] == ["BAD_ENUM", "BAD_LAYER", "BAD_LAYER", "NO_EVIDENCE"]
    assert rt.submit_assessment(THERMAL, "fault_suspected", [TEMP], [m["event_id"]], "a")["ok"]
    assert rt.submit_assessment(THERMAL, "no_fault_found", [], [m["event_id"]], "b")["ok"]
    assert rt.assessments[THERMAL]["verdict"] == "no_fault_found"
    assert len(rt.assessment_history) == 2
    assert rt.store.get_event(rt.assessments[THERMAL]["event_id"]).claim_or_request.startswith(
        "ASSESSMENT {"
    )
    plain = SwarmRuntime(
        investigation_id="p", context=rt.context, arrays=rt.arrays, store=EvidenceStore(),
        specs=list(rt.specs.values()), limits=RunLimits(budget=BUDGET),
    )  # fmt: skip
    assert plain.communication and not plain.require_assessment  # P5/P6 defaults unchanged
    assert (
        plain.submit_assessment(THERMAL, "no_fault_found", [], [], "x")["error"]
        == "NOT_A_BENCHMARK_RUN"
    )


def test_decision_rule():
    assert bm.decide({})["decision_missing"] and not bm.decide({})["flagged"]
    a = {"x": {"verdict": "no_fault_found", "suspected_layers": []},
         "y": {"verdict": "fault_suspected", "suspected_layers": [TEMP]}}  # fmt: skip
    d = bm.decide(a)
    assert d["flagged"] and d["suspected_layers"] == [TEMP] and d["n_agents_assessed"] == 2
    assert not bm.decide({"x": a["x"]})["flagged"]


# ============================================================== scripted end-to-end arms
def measure_then_assess(tool_id, modality, layer_if_fault, test):
    """Scripted fake: run one tool, then assess from its JSON result with a fixed rule."""

    def assess(ctx):
        res = json.loads(ctx.last_tool_results[-1])
        fault = test(res.get("value") or {})
        return call(
            "submit_assessment",
            verdict="fault_suspected" if fault else "no_fault_found",
            cites=[first_event_id(ctx)],
            rationale="scripted test policy",
            suspected_layers=[layer_if_fault] if fault else [],
        )

    args = {"tool_id": tool_id} | ({"modality": modality} if modality else {})
    return Script([call("run_science_tool", **args), assess, "done"])


def policies_for(arm):
    temp_bad = lambda v: v.get("min", 300) < 20  # noqa: E731
    if arm == SINGLE:
        return {GENERALIST: measure_then_assess("valid_pixel_summary", TEMP, TEMP, temp_bad)}
    return {
        THERMAL: measure_then_assess("valid_pixel_summary", TEMP, TEMP, temp_bad),
        GEOLOGY: measure_then_assess("valid_pixel_summary", TERRAIN, TERRAIN, lambda v: v.get("max", 9) < 1),
        AUDITOR: measure_then_assess("valid_pixel_summary", PRED, PRED, lambda v: v.get("max", 0) > 1.5),
        SKEPTIC: measure_then_assess("valid_pixel_summary", SHADOW, SHADOW, lambda v: v.get("max", 0) > 1.0001),
    }  # fmt: skip


class Recorder:
    def __init__(self, arm):
        self.arm, self.clients = arm, []

    def __call__(self, rt, aid):
        c = ScriptedChatClient(policies_for(self.arm)[aid], aid)
        c.bind(rt, aid)
        self.clients.append((rt.investigation_id, aid, c))
        return c


def scripted_service(arm, root):
    rec = Recorder(arm)
    svc = InvestigationService(Settings(data_root=root, runs_root=root / "runs"),
                               client_factory=rec, provider_label="scripted",
                               model_label="scripted-v0")  # fmt: skip
    return svc, rec


@pytest.fixture(scope="module")
def scripted_runs(suite, tmp_path_factory):
    m, scen, ev = suite
    out = tmp_path_factory.mktemp("bench")
    cfg = cfg_for(scen, out, [bm.BASELINE, SINGLE, ISOLATED, COMMUNICATING], ["case-01", "case-02"])
    recs = {}
    for arm in (SINGLE, ISOLATED, COMMUNICATING):
        svc, rec = scripted_service(arm, out)
        bm.run_arm(cfg, arm, svc, log=lambda *_: None)
        recs[arm] = rec
    bm.run_arm(cfg, bm.BASELINE, log=lambda *_: None)
    return cfg, recs, m, scen, ev


def test_every_case_gets_a_fresh_runtime_store_and_clients(scripted_runs):
    cfg, recs, *_ = scripted_runs
    for arm, rec in recs.items():
        n_agents = len(arm_plan(arm, 16).specs)
        inv_ids = {i for i, _, _ in rec.clients}
        assert len(inv_ids) == 2 and len(rec.clients) == 2 * n_agents  # new clients per case
        for case in ("case-01", "case-02"):
            d = cfg.run_root / arm / case
            assert (d / "events.sqlite").is_file() and (d / "decision.json").is_file()
            other = "case-02" if case == "case-01" else "case-01"
            for inv, _, c in rec.clients:
                if case in inv:
                    assert all(other not in p for p in c.prompts)  # no cross-case context


def test_all_llm_arms_share_one_total_budget(scripted_runs):
    cfg, *_ = scripted_runs
    seen = set()
    for arm in (SINGLE, ISOLATED, COMMUNICATING):
        for case in ("case-01", "case-02"):
            s = json.loads((cfg.run_root / arm / case / "run_summary.json").read_text())
            b = {k: v for k, v in s["budget"].items() if k != "max_rounds_per_agent"}
            seen.add(json.dumps(b, sort_keys=True))
            assert s["llm_calls"] <= cfg.budget.max_llm_calls
            assert s["tool_calls"] <= cfg.budget.max_tool_calls
            assert s["limits"]["max_input_tokens"] == cfg.max_input_tokens
    assert len(seen) == 1  # identical TOTAL budget for every arm


def test_four_agents_do_not_get_four_budgets(suite, tmp_path):
    m, scen, _ = suite
    tight = RunBudget(max_events=48, max_hops=5, max_tool_calls=16, max_llm_calls=3, timeout_s=60)
    cfg = cfg_for(scen, tmp_path, [ISOLATED], ["case-01"], budget=tight)
    svc, _ = scripted_service(ISOLATED, tmp_path)
    dec = bm.run_arm(cfg, ISOLATED, svc, log=lambda *_: None)[0]
    assert dec["usage"]["llm_calls"] <= 3  # shared by all four agents


def test_isolated_prompts_never_contain_other_agents_events(scripted_runs):
    _, recs, *_ = scripted_runs
    for _, aid, c in recs[ISOLATED].clients:
        for p in c.prompts:
            assert "Events published by others" not in p
            for other in (THERMAL, GEOLOGY, AUDITOR, SKEPTIC):
                if other != aid:
                    assert f"] {other} " not in p  # no digest line authored by another agent


def test_no_hidden_truth_reaches_any_model_prompt_or_tool_result(scripted_runs):
    cfg, recs, m, scen, ev = scripted_runs
    sealed = fixtures.load_sealed(ev, m)
    secrets_ = {c["fault_id"] for c in sealed.cases.values()} | {str(ev), "synthetic-region"}
    secrets_ |= {str(c["seed"]) for c in sealed.cases.values()}  # per-case seeds (large ints)
    for rec in recs.values():
        for _, _, c in rec.clients:
            text = "\n".join(c.prompts + c.tool_results)
            for tok in sorted(secrets_) + ["is_fault", "expected_detectable", "sealed", "salt"]:
                assert tok not in text, tok
            assert not re.search(r"\bPRO\b", text)


def test_freeze_then_score_and_tampering_after_freeze_is_refused(scripted_runs):
    cfg, _, m, scen, ev = scripted_runs
    for arm in cfg.arms:
        bm.freeze_arm(cfg, arm)
    with pytest.raises(RuntimeError):
        bm.freeze_arm(cfg, SINGLE)  # immutable
    with pytest.raises(RuntimeError):
        bm.run_arm(cfg, SINGLE, None)  # frozen arms cannot be re-run
    out = bm.score(cfg, ev)
    for arm in cfg.arms:
        r = out["arms"][arm]
        assert r["n_cases"] == 2 and r["declared_subset"] == ["case-01", "case-02"]
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == 2
    assert out["arms"][bm.BASELINE]["usage_total"]["llm_calls"] == 0
    assert out["arms"][COMMUNICATING]["usage_total"]["llm_calls"] > 0
    assert (cfg.run_root / "scores" / "arm_summary.csv").is_file()
    # edit one original log after freezing: scoring refuses
    p = cfg.run_root / ISOLATED / "case-01" / "events.jsonl"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed after freezing"):
        bm.score(cfg, ev)


def test_frozen_subset_hash_binds_flags_and_subset(suite):
    m, scen, ev = suite
    f = freeze_detections(m, {"case-01": True}, ["case-01"])
    res = score_detections(f, scen, ev, m.suite_id)
    assert res["n_cases"] == 1 and res["declared_subset"] == ["case-01"]
    from dataclasses import replace

    forged = replace(f, flags=(("case-01", False),))
    with pytest.raises(fixtures.SuiteIntegrityError):
        score_detections(forged, scen, ev, m.suite_id)
    with pytest.raises(ValueError):
        freeze_detections(m, {"case-01": True}, ["case-99"])
    full = freeze_detections(m, {c: False for c in m.detection_case_ids})
    assert full.declared_subset is None  # full-suite hashes unchanged from P3/P4


# =================================================================================== CLI
def test_cli_refuses_paid_runs_without_confirmation(suite, tmp_path, capsys):
    m, scen, _ = suite
    cfg = cfg_for(scen, tmp_path / "out", [bm.BASELINE, SINGLE], ["case-01"])
    p = tmp_path / "cfg.json"
    p.write_text(cfg.model_dump_json())
    assert main(["benchmark", "--config", str(p), "--action", "run"]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "model_calls" in err
    assert not (cfg.run_root / SINGLE).exists()
    assert main(["benchmark", "--config", str(p), "--action", "run", "--arms", bm.BASELINE]) == 0
    assert (cfg.run_root / bm.BASELINE / "case-01" / "decision.json").is_file()
    assert main(["benchmark", "--config", str(p), "--action", "estimate"]) == 0
    assert '"model_calls": 16' in capsys.readouterr().out


def test_live_llm_arm_requires_the_configured_model(suite, tmp_path):
    m, scen, _ = suite
    cfg = cfg_for(scen, tmp_path, [SINGLE], ["case-01"]).model_copy(update={"model": "gpt-6-luna",
                                                                            "provider": "openai"})  # fmt: skip
    svc = InvestigationService(Settings(data_root=tmp_path, runs_root=tmp_path, llm_model="other-model",
                                        openai_api_key="placeholder-not-used"))  # fmt: skip
    with pytest.raises(RuntimeError, match="differs from the benchmark config"):
        bm.run_arm(cfg, SINGLE, svc)
    assert not (cfg.run_root / SINGLE).exists()  # refused before any case directory exists


def test_firewall_benchmark_runner_is_not_imported_by_agent_or_evaluator_code():
    import ast

    src = REPO / "src" / "iceclaw"
    for sub in ("science", "data", "agents", "swarm", "reports", "evaluation", "ui"):
        for f in (src / sub).rglob("*.py"):
            tree = ast.parse(f.read_text())
            mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            assert "iceclaw.benchmark" not in mods, f
            if sub == "evaluation":
                assert not any(x.startswith(("iceclaw.agents", "iceclaw.swarm", "iceclaw.services"))
                               for x in mods), f  # fmt: skip


# ======================================================================== real paired tile
SMOKE = REPO / ".upstream" / "moonatlas-science" / "examples" / "science-smoke"
SOM = REPO / "data" / "sombench" / "prospectivity_dataset"


@pytest.mark.skipif(not (SMOKE.is_dir() and SOM.is_dir()), reason="real paired tile absent")
def test_real_paired_suite_builds_from_the_genuine_tile_without_pro(tmp_path):
    s = Settings(
        data_root=tmp_path, runs_root=tmp_path, moonatlas_roots=(SMOKE,), sombench_root=SOM
    )
    src = pr.load_real_paired_source(s)
    assert src.region_id == "ice-s-0001-0002" and set(src.case.layers) == set(pr.LAYER_ORDER)
    m = pr.build_paired_suite(src, tmp_path / "scen", tmp_path / "eval", suite_seed=1, salt=SALT)
    blob = "\n".join(p.read_text() for p in (tmp_path / "scen").rglob("*.json"))
    assert "ice-s-0001-0002" not in blob and "patch_0001_0002" not in blob
    assert not re.search(r"\bPRO\b", blob)
    assert not [p for p in tmp_path.rglob("*") if "PRO" in re.split(r"[._]", p.name)]
    s1 = pr.load_paired_scenario(tmp_path / "scen", m.suite_id, "case-01")
    check_scenario(s1)
    assert pr.verify_paired_suite(tmp_path / "scen", tmp_path / "eval") == []
    shutil.rmtree(tmp_path / "scen")


def test_cli_benchmark_works_as_a_module_entry_point(tmp_path):
    """Regression: functions defined after the __main__ guard broke `python -m iceclaw.cli`."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "ICECLAW_SKIP_DOTENV": "1"}
    cp = subprocess.run(
        [sys.executable, "-m", "iceclaw.cli", "benchmark", "--config", str(tmp_path / "none.json")],
        capture_output=True, text=True, timeout=120, env=env, cwd=tmp_path,
    )  # fmt: skip
    assert cp.returncode == 1 and "CONFIG_NOT_FOUND" in cp.stderr, cp.stderr
