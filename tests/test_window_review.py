"""Targeted cross-modal window review: runtime rules, scripted end-to-end pathway, runner, UI.

SCRIPTED providers only. These tests demonstrate that the request -> result -> cross-citation
mechanism, role permissions, completeness rules and budgets work; they are NOT research outcomes
and say nothing about how real models behave or whether communication helps (the frozen P7
result was negative). No API key, no model call.
"""

import json
import re
import time
from pathlib import Path

import pytest
from agent_helpers import AUDITOR, GEOLOGY, PRED, SKEPTIC, TEMP, TERRAIN, THERMAL, context_for
from bundle_helpers import scripted_service
from streamlit.testing.v1 import AppTest

from iceclaw import services
from iceclaw.agents.provider import ScriptedChatClient
from iceclaw.agents.review import review_plan, target_text
from iceclaw.agents.tools import make_tools
from iceclaw.config import Settings
from iceclaw.contracts import RunBudget
from iceclaw.reports import window_review as wr
from iceclaw.science import triage as tg
from iceclaw.services import InvestigationService
from iceclaw.swarm.runtime import RunLimits, SwarmRuntime
from iceclaw.swarm.store import EvidenceStore
from iceclaw.ui import runner, views

REPO = Path(__file__).resolve().parents[1]
W = [0, 16, 0, 16]  # window id 0 of the synthetic 32x32 bundle (16-px windows)
BUDGET = RunBudget(max_events=48, max_hops=5, max_tool_calls=16, max_llm_calls=24, timeout_s=60)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("rev")
    svc = scripted_service(root / "runs", root / "src", emit_reasoning=True)
    private = svc.investigate("synthetic-region", run_id="inv-rev-private")
    public = svc.publish(private.run_dir, out_id="inv-rev-public")
    return svc, private, public, root


# ------------------------------------------------------------------ scripted specialists
def ids_from(ctx, state):
    for r in ctx.last_tool_results:
        try:
            d = json.loads(r)
        except json.JSONDecodeError:
            continue
        if d.get("ok") and d.get("tool_run_id") and d["event_id"] not in state["ids"]:
            state["ids"].append(d["event_id"])


def specialist(measure, finding, *, claim=False, never_review=False, answer=None):
    """measure: list of run_science_tool argument dicts (all on the target window)."""
    st = {"ids": [], "stage": 0, "answered": set()}

    def policy(ctx):
        ids_from(ctx, st)
        if answer:
            m = re.search(r"\[(evt-\d+)\] skeptic test_request", ctx.last_user_text)
            if m and m.group(1) not in st["answered"]:
                st["answered"].add(m.group(1))
                return [("run_science_tool", {**answer, "window": W, "in_response_to": m.group(1)})]
        if st["stage"] == 0:
            st["stage"] = 1
            return [("run_science_tool", {**a, "window": W}) for a in measure]
        if st["stage"] == 1 and st["ids"] and not never_review:
            st["stage"] = 2
            calls = []
            if claim:
                calls.append(
                    ("publish_claim", {"text": "window evidence summary", "cites": st["ids"][:1]})
                )
            calls.append(("submit_review", {
                "finding": finding, "cites": st["ids"][:1], "rationale": "scripted test policy",
                "missing_evidence": ["no independent thermal time series"],
                "alternatives": ["terrain-driven illumination"], "follow_up": ["re-examine at finer resolution"],
            }))  # fmt: skip
            return calls
        return "nothing further"

    return policy


def skeptic_policy(communicating=True):
    st = {"ids": [], "stage": 0}

    def policy(ctx):
        ids_from(ctx, st)
        if not communicating:  # isolated: measure on its own, then review
            if st["stage"] == 0:
                st["stage"] = 1
                return [
                    (
                        "run_science_tool",
                        {"tool_id": "valid_pixel_summary", "modality": TEMP, "window": W},
                    )
                ]
            if st["stage"] == 1 and st["ids"]:
                st["stage"] = 2
                return [("submit_review", {"finding": "insufficient_evidence", "cites": st["ids"][:1],
                                           "rationale": "scripted"})]  # fmt: skip
            return "done"
        if st["stage"] == 0:
            st["stage"] = 1
            return [("request_test", {"to_agent": THERMAL, "tool_id": "valid_pixel_summary",
                                      "rationale": "need the raw temperature spread on the window",
                                      "modality": TEMP, "window": W})]  # fmt: skip
        m = re.search(r"\[(evt-\d+)\] thermal test_result", ctx.last_user_text)
        if st["stage"] == 1 and m:
            st["stage"] = 2
            return [("submit_review", {"finding": "insufficient_evidence", "cites": [m.group(1)],
                                       "rationale": "relied on thermal's result",
                                       "missing_evidence": ["geology and prediction cross-check"]})]  # fmt: skip
        return "waiting"

    return policy


def factory(arm, geology_files_review=True):
    comm = arm == "communicating_specialists"
    pol = {
        THERMAL: specialist([{"tool_id": "thermal_psr_consistency"}], "supports_screening", claim=comm,
                            answer={"tool_id": "valid_pixel_summary", "modality": TEMP} if comm else None),
        GEOLOGY: specialist([{"tool_id": "terrain_slope_summary"}], "weak_or_conflicting",
                            never_review=not geology_files_review),
        AUDITOR: specialist([{"tool_id": "valid_pixel_summary", "modality": PRED}], "supports_screening"),
        SKEPTIC: skeptic_policy(comm),
    }  # fmt: skip
    made = []

    def make(rt, aid):
        c = ScriptedChatClient(pol[aid], aid)
        c.bind(rt, aid)
        made.append(c)
        return c

    make.clients = made
    return make


def service(root, fac, latency=0.0):
    svc = InvestigationService(
        Settings(data_root=root, runs_root=root / "runs"),
        client_factory=fac,
        provider_label="scripted",
        model_label="scripted-v0",
    )
    return svc


def run_review(world, tmp_path, arm="communicating_specialists", **kw):
    _, private, public, _ = world
    fac = factory(arm, **kw)
    res = services.investigate_window(
        service(tmp_path, fac),
        source_dir=public.run_dir / "source",
        window_id=0,
        out_dir=tmp_path / "review",
        source_label="inv-rev-public",
        arm=arm,
        budget=BUDGET,
    )
    return res, fac


# ===================================================== the genuine cross-agent request/result path
def test_scripted_communicating_review_links_request_result_and_cross_citations(world, tmp_path):
    res, fac = run_review(world, tmp_path)
    r = res.record
    m = r["metrics"]
    assert r["status"] == "COMPLETE" and r["incomplete_reasons"] == [] and m["review_complete"]
    assert set(r["reviews"]) == {THERMAL, GEOLOGY, AUDITOR, SKEPTIC}
    assert r["agent_findings"][GEOLOGY] == "weak_or_conflicting" and r["agreement"] == "mixed"
    # distinct specialists supplied distinct layers ON the target window
    lay = r["layers_measured_on_window"]
    assert lay[TEMP] == [THERMAL] and lay[TERRAIN] == [GEOLOGY] and lay[PRED] == [AUDITOR]
    # a real request/result pair, skeptic -> thermal, linked in the append-only log
    (req,) = r["requests"]
    assert req["from"] == SKEPTIC and req["to"] == THERMAL and req["answered"]
    ev = {x["event_id"]: x for x in r["evidence"]}
    result = ev[req["result_event_ids"][0]]
    assert result["kind"] == "test_result" and result["supplier"] == THERMAL
    assert result["answers_request"] == req["event_id"] and result["window_scoped"]
    # the skeptic's review cites THERMAL's result: a cross-agent citation by construction
    assert THERMAL in r["reviews"][SKEPTIC]["cites_other_agents"]
    assert any(
        c["agent"] == SKEPTIC and c["supplier"] == THERMAL for c in r["cross_agent_citations"]
    )
    assert any(
        c["agent"] == THERMAL and c["supplier"] == SKEPTIC for c in r["cross_agent_citations"]
    )
    assert m["n_test_requests"] == 1 and m["n_test_requests_answered"] == 1
    assert m["n_challenges"] == 0 and m["n_revisions"] == 0  # none forced, none invented
    # numbers: every recorded measurement independently recomputes
    assert (
        r["numerical_verification"]["failed_event_ids"] == []
        and m["numerical_verification_success"]
    )
    assert m["n_measurements_checked"] == m["n_measurements_reproduced"] > 0
    # specialists' own words are carried verbatim and labelled as AI hypotheses
    assert "re-examine at finer resolution" in [x["text"] for x in r["suggested_follow_up"]]
    assert r["missing_evidence"] and r["competing_explanations"] and "AI hypotheses" in r["label"]
    # the real event store holds the linked events; nothing was injected by the runner
    events = [
        json.loads(x) for x in (tmp_path / "review" / "events.jsonl").read_text().splitlines()
    ]
    kinds = {e["event_kind"] for e in events}
    assert {"measurement", "test_request", "test_result", "completion"} <= kinds
    assert all(e["agent_id"] in {THERMAL, GEOLOGY, AUDITOR, SKEPTIC, "supervisor"} for e in events)
    # budgets: same TOTALS, not per agent
    assert (
        m["usage"]["llm_calls"] <= BUDGET.max_llm_calls
        and m["usage"]["tool_calls"] <= BUDGET.max_tool_calls
    )
    assert (
        json.loads((tmp_path / "review" / "review.json").read_text())["source"]["bundle"]
        == "inv-rev-public"
    )


def test_a_missing_review_makes_the_review_incomplete_and_substitutes_nothing(world, tmp_path):
    res, _ = run_review(world, tmp_path, geology_files_review=False)
    r = res.record
    assert r["status"] == "INCOMPLETE" and not r["metrics"]["review_complete"]
    assert GEOLOGY not in r["reviews"] and GEOLOGY not in r["agent_findings"]
    assert any("no review filed by geology" in x for x in r["incomplete_reasons"])
    assert r["metrics"]["n_reviews_filed"] == 3 and r["metrics"]["n_incomplete_reviews"] == 1
    assert set(r["agent_findings"].values()) <= set(["supports_screening", "insufficient_evidence"])


def test_isolated_arm_has_no_communication_tools_and_makes_no_requests(world, tmp_path):
    res, fac = run_review(world, tmp_path, arm="isolated_specialists")
    r = res.record
    assert r["status"] == "COMPLETE" and r["requests"] == [] and r["cross_agent_citations"] == []
    assert r["metrics"]["n_test_requests"] == 0
    for c in fac.clients:
        assert not any("Events published by others" in p for p in c.prompts)


def test_tool_surfaces_and_runtime_rules(world):
    _, _, public, _ = world
    from iceclaw.reports import package as pkg

    arrays = pkg.load_package(public.run_dir / "source").arrays
    target = {
        "label": "w000",
        "window": W,
        "centre_lat": -80.1,
        "centre_lon": -36.0,
        "category": "followup",
    }
    plan = review_plan("communicating_specialists", 16, target)
    rt = SwarmRuntime(
        investigation_id="r", context=context_for(arrays, BUDGET), arrays=arrays, store=EvidenceStore(),
        specs=list(plan.specs), limits=RunLimits(budget=BUDGET), review_mode=True,
        require_assessment=True, review_window=tuple(W),
    )  # fmt: skip
    names = {sp.agent_id: {t.name for t in make_tools(rt, sp)} for sp in plan.specs}
    for n in names.values():
        assert "submit_review" in n and "submit_assessment" not in n
    assert {"request_test", "publish_challenge"} <= names[SKEPTIC]
    whole = rt.science_tool(THERMAL, "thermal_psr_consistency")  # whole tile, NOT the window
    assert (
        rt.submit_review(THERMAL, "supports_screening", [whole["event_id"]], "x")["error"]
        == "NOT_WINDOW_SCOPED"
    )
    assert rt.submit_review(THERMAL, "maybe", [whole["event_id"]], "x")["error"] == "BAD_ENUM"
    assert rt.submit_review(THERMAL, "supports_screening", [], "x")["error"] == "NO_EVIDENCE"
    assert rt.submit_assessment(THERMAL, "no_fault_found", [], [whole["event_id"]], "x")[
        "error"
    ] == ("NOT_A_BENCHMARK_RUN")
    on = rt.science_tool(THERMAL, "thermal_psr_consistency", window=W)
    assert rt.submit_review(THERMAL, "supports_screening", [on["event_id"]], "ok")["ok"]
    assert rt.assessments[THERMAL]["verdict"] == "supports_screening"
    # an agent keeps its role permissions: geology cannot touch the temperature layer
    assert (
        rt.science_tool(GEOLOGY, "valid_pixel_summary", TEMP, window=W)["error"]
        == "MODALITY_NOT_ALLOWED"
    )
    with pytest.raises(ValueError):
        SwarmRuntime(investigation_id="x", context=context_for(arrays, BUDGET), arrays=arrays, store=EvidenceStore(),
                     specs=list(plan.specs), limits=RunLimits(budget=BUDGET), review_mode=True)  # fmt: skip
    plain = SwarmRuntime(investigation_id="p", context=context_for(arrays, BUDGET), arrays=arrays, store=EvidenceStore(),
                         specs=list(plan.specs), limits=RunLimits(budget=BUDGET))  # fmt: skip
    assert plain.review_mode is False and plain.submit_review(THERMAL, "x", [], "y")["error"] == (
        "NOT_A_REVIEW_RUN"
    )  # P5/P6/P7 defaults are unchanged


def test_prompts_name_the_window_but_never_hand_over_statistics(world):
    _, _, public, _ = world
    from iceclaw.reports import package as pkg

    arrays = pkg.load_package(public.run_dir / "source").arrays
    res = tg.screen_arrays(arrays, sensitivity=False)
    w = tg.window_by_id(res, 0)
    target = {"label": w["label"], "window": w["window"], "centre_lat": w["centre_lat"],
              "centre_lon": w["centre_lon"], "category": "followup"}  # fmt: skip
    plan = review_plan("communicating_specialists", 16, target)
    stats = [str(w["tmax_k"]["median"]), str(w["slope_deg"]["median"]), str(w["prediction_index"]["mean"]),
             str(w["shadow_fraction"]), str(res["index_threshold_raw"])]  # fmt: skip
    for sp in plan.specs:
        assert "window=[0,16,0,16]" in sp.instructions and "w000" in sp.instructions
        assert (
            "submit_review" in sp.instructions and "not independent confirmation" in sp.instructions
        )
        for s in stats:
            assert s not in sp.instructions
    assert "consistent' and 'insufficient evidence' are valid" in plan.specs[3].instructions or (
        "valid" in plan.specs[3].instructions
    )
    assert "selected by deterministic code" in target_text(target)
    assert plan.communication and not review_plan("isolated_specialists", 16, target).communication
    assert SKEPTIC in {sp.agent_id for sp in plan.specs if sp.starts_idle}


def test_review_records_the_declared_runtime_limits(world, tmp_path):
    res, _ = run_review(world, tmp_path)
    rt = res.record["runtime"]
    assert rt["digest_max_chars"] == services.REVIEW_DIGEST_CHARS == 3000
    assert (
        rt["max_prompt_chars"] == services.REVIEW_PROMPT_CHARS and rt["max_input_tokens"] == 64_000
    )
    assert rt["reserve"]["llm_calls"] == 3 and rt["reserve"]["input_tokens"] == 12_000
    assert (
        services.DEFAULT_REVIEW_BUDGET.max_llm_calls,
        services.DEFAULT_REVIEW_BUDGET.max_tool_calls,
    ) == (16, 16)


def test_review_record_is_a_pure_function_of_the_log(world, tmp_path):
    res, _ = run_review(world, tmp_path)
    from iceclaw.reports import package as pkg

    arrays = pkg.load_package(world[2].run_dir / "source").arrays
    store = EvidenceStore.open_readonly(tmp_path / "review" / "events.sqlite")
    again = wr.build_review_record(
        store, arrays, arm=res.record["arm"], agent_ids=[THERMAL, GEOLOGY, AUDITOR, SKEPTIC],
        target=res.record["target"], run=res.summary,
    )  # fmt: skip
    store.close()
    keep = ("status", "agreement", "evidence", "requests", "cross_agent_citations", "metrics")
    assert {k: again[k] for k in keep} == {k: res.record[k] for k in keep}
    assert (
        wr.parse_review_text("REVIEW {bad json} | x") is None
        and wr.parse_review_text("hello") is None
    )


def test_unknown_window_and_tampered_source_are_refused_before_any_model_call(world, tmp_path):
    _, _, public, _ = world
    fac = factory("communicating_specialists")
    with pytest.raises(tg.TriageError) as e:
        services.investigate_window(service(tmp_path, fac), source_dir=public.run_dir / "source",
                                    window_id=999, out_dir=tmp_path / "x", arm="isolated_specialists")  # fmt: skip
    assert e.value.code == "UNKNOWN_WINDOW" and fac.clients == []
    import shutil

    from iceclaw.reports import package as pkg

    bad = tmp_path / "bad-source"
    shutil.copytree(public.run_dir / "source", bad)
    f = next((bad / "files").glob("*SLOPE.tif"))
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(pkg.PackageError):
        services.investigate_window(service(tmp_path, fac), source_dir=bad, window_id=0,
                                    out_dir=tmp_path / "y", arm="isolated_specialists")  # fmt: skip
    assert fac.clients == []


def test_paid_review_needs_configuration_and_never_runs_silently(world, tmp_path):
    _, _, public, _ = world
    st = Settings(data_root=tmp_path, runs_root=tmp_path / "runs", llm_model="gpt-6-luna")  # no key
    with pytest.raises(Exception) as e:
        services.investigate_window(InvestigationService(st), source_dir=public.run_dir / "source",
                                    window_id=0, out_dir=tmp_path / "z")  # fmt: skip
    assert getattr(e.value, "code", "") == "MISSING_API_KEY" and not (tmp_path / "z").exists()


# ===================================================== runner (single-flight) and the page
def test_manager_runs_one_review_at_a_time_and_saves_it(world, tmp_path):
    _, _, public, _ = world
    runs = tmp_path / "runs"
    fac = factory("communicating_specialists")

    def slow(rt, aid):
        c = fac(rt, aid)
        c.latency_s = 0.03
        return c

    svc = InvestigationService(Settings(data_root=tmp_path, runs_root=runs), client_factory=slow,
                               provider_label="scripted", model_label="scripted-v0")  # fmt: skip
    mgr = runner.LiveRunManager(runs, lambda: svc)
    r1 = mgr.start_review(
        source_dir=public.run_dir / "source", window_id=0, bundle_name="inv-rev-public"
    )
    assert r1.started
    for _ in range(4):
        again = mgr.start_review(source_dir=public.run_dir / "source", window_id=0, bundle_name="x")
        assert not again.started and again.code == "ALREADY_RUNNING"
    assert mgr.start("synthetic-region").code == "ALREADY_RUNNING"  # shares the guard
    seen = 0
    t0 = time.time()
    while mgr.snapshot().state == runner.RUNNING and time.time() < t0 + 60:
        seen = max(seen, len(mgr.progress().events))
        time.sleep(0.01)
    mgr.wait(60)
    snap = mgr.snapshot()
    assert snap.state == runner.SUCCEEDED and snap.kind == "review" and mgr.starts_attempted == 1
    assert snap.ended_reason in ("COMPLETE", "INCOMPLETE") and seen > 0
    assert not (runs / runner.MARKER).exists()
    saved = views.saved_reviews(runs, "inv-rev-public")
    assert len(saved) == 1 and saved[0]["review_id"] and views.saved_reviews(runs, "other") == []
    assert views.discover_public_bundles(runs) == []  # reviews never pose as bundles


def test_manager_review_refuses_without_configuration_before_creating_anything(world, tmp_path):
    _, _, public, _ = world
    st = Settings(data_root=tmp_path, runs_root=tmp_path / "runs", llm_model="gpt-6-luna")
    mgr = runner.LiveRunManager(st.runs_root, lambda: InvestigationService(st))
    res = mgr.start_review(source_dir=public.run_dir / "source", window_id=0, bundle_name="b")
    assert not res.started and res.code == "MISSING_API_KEY"
    assert not st.runs_root.exists() and mgr.snapshot().state == runner.IDLE


def page_env(monkeypatch, runs_root):
    monkeypatch.setenv("ICECLAW_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", "/nonexistent-iceclaw")
    monkeypatch.setenv("ICECLAW_SOMBENCH_ROOT", "/nonexistent-iceclaw-sombench")
    monkeypatch.setenv("ICECLAW_DATA_ROOT", "/nonexistent-iceclaw")


def text_of(at, kind):
    return "\n".join(getattr(e, "value", "") or "" for e in getattr(at, kind))


@pytest.fixture
def clean_state():
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    runner.forget_managers()
    yield
    runner.forget_managers()


def test_page_shows_screening_labels_and_separates_it_from_attention(
    world, tmp_path, monkeypatch, clean_state
):
    import shutil

    _, _, public, _ = world
    runs = tmp_path / "runs"
    shutil.copytree(public.run_dir, runs / public.run_id)
    page_env(monkeypatch, runs)
    from iceclaw.services import InvestigationService as S

    def no(*a, **k):
        raise AssertionError("the page must not call a model")

    monkeypatch.setattr(S, "investigate", no)
    monkeypatch.setattr(services, "investigate_window", no)
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    subs = [s.value for s in at.subheader]
    assert any("Research follow-up screening" in s for s in subs) and any(
        "Verify and export" in s for s in subs
    )
    warn = text_of(at, "warning")
    assert "POST-PROCESSING" in warn and "not an action of the agents" in warn
    md = text_of(at, "markdown")
    assert "LPSR ≥ 0.5" in md and "110 K" in md and "15°" in md and "90th" in md
    assert "evidence discrepancy for review, not a confirmed model fault" in md
    cap = text_of(at, "caption")
    assert "Separate from the investigation-attention ranking" in cap
    assert "not independent confirmation of ice" in cap
    labels = {m.label: m.value for m in at.metric}
    assert labels["Follow-up windows"] == "1" and labels["Pixels meeting all conditions"] == "15"
    start = next(b for b in at.button if "Start targeted review" in b.label)
    assert start.disabled  # no confirmation, no API configuration


def test_page_review_click_needs_confirmation_and_shows_the_saved_review(
    world, tmp_path, monkeypatch, clean_state
):
    import shutil

    _, _, public, root = world
    runs = tmp_path / "runs"
    shutil.copytree(public.run_dir, runs / public.run_id)
    page_env(monkeypatch, runs)
    fac = factory("communicating_specialists")
    svc = InvestigationService(Settings(data_root=tmp_path, runs_root=runs), client_factory=fac,
                               provider_label="scripted", model_label="scripted-v0")  # fmt: skip
    mgr = runner.LiveRunManager(runs, lambda: svc)
    monkeypatch.setattr(runner, "get_manager", lambda *a, **k: mgr)
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    go = next(b for b in at.button if "Start targeted review" in b.label)
    assert go.disabled and mgr.starts_attempted == 0
    for _ in range(3):
        at.run()  # plain reruns never start a review
    assert mgr.starts_attempted == 0
    at.checkbox(key="confirm_review").check().run()
    go = next(b for b in at.button if "Start targeted review" in b.label)
    assert not go.disabled and mgr.starts_attempted == 0
    go.click().run()
    assert mgr.starts_attempted == 1
    mgr.wait(60)
    for _ in range(2):
        at.run()
    assert mgr.starts_attempted == 1 and not at.exception, at.exception
    saved = views.saved_reviews(runs, public.run_id)
    assert len(saved) == 1
    page = (
        text_of(at, "success")
        + text_of(at, "warning")
        + text_of(at, "caption")
        + text_of(at, "markdown")
    )
    assert "review of w000" in page and "Independent measurements considered" in page
    assert "Independent recomputation" in page and "AI hypotheses" in page
    assert "Saved targeted reviews of this bundle (1)" in page
