"""P7 protocol v2: guaranteed per-agent minimums, explicit 'incomplete' verdicts, pre-registration.

Offline only (SYNTHETIC arrays, scripted fake models). v1 artifacts are never touched.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

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
    context_for,
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
from iceclaw.services import InvestigationService
from iceclaw.swarm.runtime import AgentReserve, BudgetExhausted, RunLimits, SwarmRuntime
from iceclaw.swarm.store import EvidenceStore

REPO = Path(__file__).resolve().parents[1]
SALT = b"\x02" * 16
BUDGET = RunBudget(max_events=48, max_hops=5, max_tool_calls=16, max_llm_calls=16, timeout_s=60)
RESERVE = AgentReserve(llm_calls=3, tool_calls=3, input_tokens=12_000, assessment_events=1)


def runtime(reserve=RESERVE, communication=False, budget=BUDGET, max_input_tokens=64_000):
    arrays = synthetic_arrays()
    plan = arm_plan(COMMUNICATING if communication else ISOLATED, budget.max_llm_calls)
    return SwarmRuntime(
        investigation_id="inv-v2", context=context_for(arrays, budget), arrays=arrays,
        store=EvidenceStore(), specs=list(plan.specs),
        limits=RunLimits(budget=budget, reserve=reserve, max_input_tokens=max_input_tokens),
        communication=communication, require_assessment=True,
    )  # fmt: skip


def burn_tools(rt, agent, modality, start=0):
    """Distinct quantile requests until the runtime refuses; returns the number accepted.

    ``start`` offsets the requested quantile so repeated bursts never hit the (free) cache."""
    n = 0
    for i in range(start, start + 40):
        r = rt.science_tool(agent, "quantiles", modality, qs=[round(0.001 + 0.0025 * i, 4)])
        if not r.get("ok"):
            assert r["error"] in ("TOOL_BUDGET_EXHAUSTED", "EVENT_BUDGET_EXHAUSTED"), r
            return n
        assert not r.get("cached"), r
        n += 1
    return n


# ============================================================== reserve accounting (unit)
def test_without_reserves_one_greedy_agent_can_starve_the_others():
    rt = runtime(reserve=None)
    assert burn_tools(rt, GEOLOGY, TERRAIN) == 16  # v1 behaviour: the whole shared pool
    assert rt.science_tool(THERMAL, "thermal_psr_consistency")["error"] == "TOOL_BUDGET_EXHAUSTED"


def test_reserves_guarantee_every_agent_its_minimum_tools():
    rt = runtime()
    # geology: own 3 + common pool 16 - 4*3 = 4  -> 7; the other three keep 3 each
    assert burn_tools(rt, GEOLOGY, TERRAIN) == 7
    assert burn_tools(rt, AUDITOR, PRED) == 3
    assert burn_tools(rt, SKEPTIC, SHADOW) == 3
    assert rt.science_tool(THERMAL, "thermal_psr_consistency")["ok"]  # starvation prevented
    assert rt.tool_calls <= BUDGET.max_tool_calls == 16


def test_unused_reserve_returns_to_the_pool_after_assessment_or_exit():
    rt = runtime()
    m = rt.science_tool(AUDITOR, "valid_pixel_summary", PRED)
    assert rt.submit_assessment(AUDITOR, "no_fault_found", [], [m["event_id"]], "ok")["ok"]
    # auditor's 2 unused reserved tool calls are now common: geology can use 3 + 4 + 2
    assert burn_tools(rt, GEOLOGY, TERRAIN) == 9
    rt.state[SKEPTIC].exited = True  # an exited agent holds nothing
    assert burn_tools(rt, GEOLOGY, TERRAIN, start=100) == 3
    assert rt.science_tool(THERMAL, "thermal_psr_consistency")["ok"]  # thermal still guaranteed


def test_model_call_and_token_reserves_are_enforced_before_any_spend():
    rt = runtime()
    now = datetime.now(UTC)
    for _ in range(7):  # own 3 + common 4
        rid = rt.llm_before_call(GEOLOGY)
        rt.llm_record(GEOLOGY, started=now, request=[], response={}, error=None, reservation=rid,
                      usage={"input_token_count": 100, "output_token_count": 5, "total_token_count": 105})  # fmt: skip
    with pytest.raises(BudgetExhausted) as e:
        rt.llm_before_call(GEOLOGY)
    assert e.value.kind == "LLM_CALLS"
    assert rt.llm_before_call(THERMAL)  # the others' minimum is intact
    rt2 = runtime(max_input_tokens=64_000)
    rid = rt2.llm_before_call(GEOLOGY)
    rt2.llm_record(GEOLOGY, started=now, request=[], response={}, error=None, reservation=rid,
                   usage={"input_token_count": 27_000, "output_token_count": 5, "total_token_count": 27_005})  # fmt: skip
    # geology used 27k > its 12k reserve; 64k - 27k - 3*12k held = 1k left for geology
    assert rt2.available_to(GEOLOGY)["input_tokens"] == 1_000
    with pytest.raises(BudgetExhausted) as e2:
        rt2.llm_before_call(GEOLOGY, measure_chars=20_000)
    assert e2.value.kind == "INPUT_TOKENS"
    assert rt2.llm_before_call(THERMAL, measure_chars=20_000)  # 10k estimate fits its reserve


def test_every_agent_keeps_an_event_slot_for_its_assessment():
    small = RunBudget(max_events=8, max_hops=5, max_tool_calls=16, max_llm_calls=16, timeout_s=60)
    rt = runtime(budget=small)
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    n = 1 + burn_tools(rt, GEOLOGY, TERRAIN)  # events, not tools, are the limit here
    assert rt.store.count_events() == n == 8 - 4  # 4 slots stay reserved for assessments
    assert rt.publish_claim(GEOLOGY, "x", [], evidence_type="hypothesis")["error"] == (
        "EVENT_BUDGET_EXHAUSTED"
    )
    assert rt.submit_assessment(THERMAL, "no_fault_found", [], [m["event_id"]], "ok")["ok"]


def test_defaults_keep_v1_and_p5_p6_behaviour():
    assert RunLimits(budget=BUDGET).reserve is None
    rt = runtime(reserve=None)
    assert rt.held_by_others(THERMAL, "tool") == 0 and rt.held_by_others(THERMAL, "llm") == 0


# ============================================================================ decisions
def test_decide_v2_distinguishes_incomplete_from_no_fault():
    ids = [THERMAL, GEOLOGY, AUDITOR, SKEPTIC]
    nf = {"verdict": "no_fault_found", "suspected_layers": []}
    fs = {"verdict": "fault_suspected", "suspected_layers": [TEMP]}
    all_nf = bm.decide_v2({a: nf for a in ids}, ids)
    assert all_nf["team_decision"] == bm.NOT_FLAGGED and all_nf["n_incomplete_verdicts"] == 0
    part = bm.decide_v2({GEOLOGY: nf, AUDITOR: nf, SKEPTIC: nf}, ids)
    assert part["team_decision"] == bm.INCOMPLETE and not part["flagged"]
    assert (
        part["individual_verdicts"][THERMAL] == "incomplete" and part["n_incomplete_verdicts"] == 1
    )
    alarm = bm.decide_v2({GEOLOGY: fs}, ids)
    assert (
        alarm["team_decision"] == bm.FLAGGED
        and alarm["flagged"]
        and alarm["n_incomplete_verdicts"] == 3
    )
    none = bm.decide_v2({}, [GENERALIST])
    assert none["team_decision"] == bm.INCOMPLETE and none["individual_verdicts"] == {
        GENERALIST: "incomplete"
    }


# ============================================================================== config
def v2_config(scen, out, **kw):
    return bm.BenchmarkConfigV2(
        scenarios_root=str(scen), output_root=str(out),
        arms=kw.pop("arms", [bm.BASELINE, SINGLE, ISOLATED, COMMUNICATING]), case_ids=kw.pop("case_ids", []),
        budget=kw.pop("budget", BUDGET), max_input_tokens=kw.pop("max_input_tokens", 64_000),
        provider="scripted", model="scripted-v0",
        reserve=kw.pop("reserve", bm.ReservePlan(llm_calls_per_agent=3, tool_calls_per_agent=3,
                                                 input_tokens_per_agent=12_000)),
        preregistration={"fault_ids": list(pr.PAIRED_V2_FAULT_IDS)}, **kw,
    )  # fmt: skip


def test_v2_config_validation_and_v1_hashes_unchanged(tmp_path):
    bad = v2_config(tmp_path, tmp_path, reserve=bm.ReservePlan(llm_calls_per_agent=5,
                    tool_calls_per_agent=3, input_tokens_per_agent=1))  # fmt: skip
    p = tmp_path / "bad.json"
    p.write_text(bad.model_dump_json())
    with pytest.raises(ValueError, match="model calls"):
        bm.load_config(p)
    good = v2_config(tmp_path, tmp_path)
    p.write_text(good.model_dump_json())
    cfg = bm.load_config(p)
    assert bm.is_v2(cfg) and cfg.sha256() == good.sha256() and cfg.suite_id == "p7-paired-v2"
    assert bm.planned_case_count(cfg) == 6
    # the frozen v1 configs still hash to the names of their existing output folders
    assert bm.load_config(REPO / "benchmarks" / "p7_pilot.json").sha256()[:12] == "663766aa59f1"
    assert (
        bm.load_config(REPO / "benchmarks" / "p7_baseline_full.json").sha256()[:12]
        == "1a6047ac4711"
    )
    assert not bm.is_v2(bm.load_config(REPO / "benchmarks" / "p7_pilot.json"))


@pytest.mark.skipif(not (REPO / "benchmarks" / "p7_v2.json").is_file(), reason="v2 config absent")
def test_preregistered_v2_config_is_internally_consistent():
    cfg = bm.load_config(REPO / "benchmarks" / "p7_v2.json")
    assert bm.is_v2(cfg) and cfg.model == "gpt-6-luna" and cfg.provider == "openai"
    assert cfg.suite_id == pr.PAIRED_V2_SUITE_ID and cfg.case_ids == []
    assert (
        bm.planned_fault_ids(cfg) == pr.PAIRED_V2_FAULT_IDS
        and 4 <= len(pr.PAIRED_V2_FAULT_IDS) <= 6
    )
    assert cfg.arms == [bm.BASELINE, SINGLE, ISOLATED, COMMUNICATING]
    assert cfg.decision_rule == bm.DECISION_RULE_V2
    pre = REPO / "benchmarks" / "p7_v2_preregistration.json"
    rec = json.loads(pre.read_text())
    assert rec["config_sha256"] == cfg.sha256()  # registration binds this exact config


# ================================================================================ suite
def test_v2_suite_has_two_clean_and_four_faults_with_a_private_seed(tmp_path):
    src = pr.paired_source_from_arrays(
        synthetic_arrays(64), "south", {"source_mode": "upstream_published_example"}
    )
    m = pr.build_paired_suite(src, tmp_path / "s", tmp_path / "e", suite_id=pr.PAIRED_V2_SUITE_ID,
                              fault_ids=pr.PAIRED_V2_FAULT_IDS, suite_version="2", salt=SALT)  # fmt: skip
    sealed = fixtures.load_sealed(tmp_path / "e", m)
    assert m.suite_version == "2" and len(m.cases) == 6
    assert sorted(sealed.labels.values()) == [False, False, True, True, True, True]
    assert sealed.suite_seed > 2**20  # a fresh random 63-bit seed, not a repo constant
    text = (tmp_path / "s" / pr.PAIRED_V2_SUITE_ID / "manifest.json").read_text()
    for tok in list(pr.PAIRED_V2_FAULT_IDS) + ["is_fault", "suite_seed", "synthetic-region"]:
        assert tok not in text, tok
    assert pr.verify_paired_suite(tmp_path / "s", tmp_path / "e", pr.PAIRED_V2_SUITE_ID) == []


# ======================================================= scripted end-to-end (v1 vs v2 rules)
def greedy_then_assess(modality, n_tools):
    """Scripted fake: try ``n_tools`` distinct measurements, then assess citing one of them."""
    state = {"ids": []}

    def policy(ctx):
        for r in ctx.last_tool_results[-n_tools:]:
            d = json.loads(r)
            if d.get("ok") and d.get("event_id") and d["event_id"] not in state["ids"]:
                state["ids"].append(d["event_id"])
        if ctx.roundtrip == 0:
            if modality is None:
                return [("run_science_tool", {"tool_id": "thermal_psr_consistency"})]
            return [("run_science_tool", {"tool_id": "quantiles", "modality": modality,
                                          "qs": [round(0.01 + 0.03 * i, 2)]}) for i in range(n_tools)]  # fmt: skip
        if ctx.roundtrip == 1 and state["ids"]:
            return [("submit_assessment", {"verdict": "no_fault_found", "cites": state["ids"][:1],
                                           "rationale": "scripted"})]  # fmt: skip
        return "done"

    return policy


POLICIES = {
    GEOLOGY: (TERRAIN, 8, 0.0),
    AUDITOR: (PRED, 8, 0.0),
    SKEPTIC: (SHADOW, 8, 0.0),
    THERMAL: (None, 1, 0.05),  # slow: the others act first
}


class Factory:
    def __call__(self, rt, aid):
        if aid == GENERALIST:
            policy, lat = greedy_then_assess(TEMP, 4), 0.0
        else:
            mod, n, lat = POLICIES[aid]
            policy = greedy_then_assess(mod, n)
        c = ScriptedChatClient(policy, aid)
        c.latency_s = lat
        c.bind(rt, aid)
        return c


def svc(root):
    return InvestigationService(Settings(data_root=root, runs_root=root / "runs"), client_factory=Factory(),
                                provider_label="scripted", model_label="scripted-v0")  # fmt: skip


@pytest.fixture(scope="module")
def v2_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2")
    src = pr.paired_source_from_arrays(
        synthetic_arrays(64), "south", {"source_mode": "upstream_published_example"}
    )
    m = pr.build_paired_suite(src, root / "scen", root / "eval", suite_id=pr.PAIRED_V2_SUITE_ID,
                              fault_ids=pr.PAIRED_V2_FAULT_IDS, suite_version="2", salt=SALT)  # fmt: skip
    return root, m


def test_greedy_agents_starve_thermal_under_v1_rules_but_not_under_v2(v2_suite):
    root, m = v2_suite
    v1 = bm.BenchmarkConfig(scenarios_root=str(root / "scen"), output_root=str(root / "o1"),
                            suite_id=pr.PAIRED_V2_SUITE_ID, arms=[ISOLATED], case_ids=["case-01"],
                            budget=BUDGET, max_input_tokens=200_000, provider="scripted", model="scripted-v0")  # fmt: skip
    d1 = bm.run_arm(v1, ISOLATED, svc(root), log=lambda *_: None)[0]
    assert THERMAL not in d1["verdicts"]  # starved: no tool left, so no citable measurement
    v2 = v2_config(root / "scen", root / "o2", arms=[ISOLATED, COMMUNICATING], case_ids=["case-01"],
                   max_input_tokens=200_000)  # fmt: skip
    for arm in (ISOLATED, COMMUNICATING):
        d2 = bm.run_arm(v2, arm, svc(root), log=lambda *_: None)[0]
        assert d2["individual_verdicts"][THERMAL] == "no_fault_found", (arm, d2)
        assert d2["team_decision"] == bm.NOT_FLAGGED and d2["n_incomplete_verdicts"] == 0
        assert d2["usage"]["tool_calls"] <= 16 and d2["usage"]["llm_calls"] <= 16  # same totals
        assert d2["reserve"]["tool_calls"] == 3


def test_idle_communicating_skeptic_gets_a_final_turn_under_v2(v2_suite):
    root, _ = v2_suite
    v2 = v2_config(root / "scen", root / "o3", arms=[COMMUNICATING], case_ids=["case-02"],
                   max_input_tokens=200_000)  # fmt: skip
    d = bm.run_arm(v2, COMMUNICATING, svc(root), log=lambda *_: None)[0]
    # no agent published a claim, so the Skeptic was never woken by the bus: the v2 final call
    # still gave it a turn, and it assessed
    assert d["individual_verdicts"][SKEPTIC] == "no_fault_found"


def test_v2_scoring_reports_incomplete_and_complete_case_metrics(v2_suite, tmp_path):
    root, m = v2_suite
    cfg = v2_config(root / "scen", tmp_path / "o", arms=[bm.BASELINE, SINGLE, ISOLATED],
                    max_input_tokens=200_000, case_ids=["case-01", "case-02"])  # fmt: skip
    for arm in cfg.arms:
        bm.run_arm(cfg, arm, svc(root) if arm != bm.BASELINE else None, log=lambda *_: None)
    for arm in cfg.arms:
        bm.freeze_arm(cfg, arm)
    out = bm.score(cfg, root / "eval")
    for arm in cfg.arms:
        r = out["arms"][arm]
        assert {
            "incomplete_cases",
            "incomplete_individual_verdicts",
            "secondary_complete_cases_only",
        } <= set(r)
        assert r["usage_total"]["elapsed_s"] >= 0
    assert out["format"] == bm.FORMAT_V2 and out["decision_rule"] == bm.DECISION_RULE_V2
    rows = (cfg.run_root / "scores" / "case_outcomes.csv").read_text().splitlines()
    assert "team_decision" in rows[0] and "individual_verdicts" in rows[0]
    assert out["arms"][bm.BASELINE]["incomplete_cases"] == 0


def test_cli_estimate_works_before_the_v2_suite_exists(tmp_path, capsys):
    cfg = v2_config(tmp_path / "no-suite-yet", tmp_path / "o")
    p = tmp_path / "v2.json"
    p.write_text(cfg.model_dump_json())
    assert main(["benchmark", "--config", str(p), "--action", "estimate"]) == 0
    est = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert est["n_cases"] == 6 and est["upper_bound"]["model_calls"] == 16 * 6 * 3


def test_single_arm_never_gets_a_reserve(v2_suite, tmp_path):
    root, _ = v2_suite
    cfg = v2_config(
        root / "scen", tmp_path, arms=[SINGLE], case_ids=["case-03"], max_input_tokens=200_000
    )
    d = bm.run_arm(cfg, SINGLE, svc(root), log=lambda *_: None)[0]
    assert d["reserve"] is None and set(d["individual_verdicts"]) == {GENERALIST}
    assert replace(RESERVE, llm_calls=0).llm_calls == 0
