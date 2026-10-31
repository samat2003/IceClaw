"""Offline tests of the specialist collective on REAL Microsoft Agent Framework objects.

The model is a scripted fake (``ScriptedChatClient``) running inside MAF's actual function-
invocation loop, so tool routing, tool schemas, message flow and metering are the genuine ones.
A fake model does NOT prove live API operation; that is covered by the separate live test/run.
Arrays are SYNTHETIC unless a test says "real region".
"""

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from agent_helpers import (
    AUDITOR,
    EVT,
    GEOLOGY,
    PRED,
    SKEPTIC,
    TEMP,
    TERRAIN,
    THERMAL,
    call,
    first_event_id,
    make_runtime,
    parse,
    quiet,
    run_with,
)

from iceclaw.agents.provider import ScriptedChatClient, build_openai_client
from iceclaw.agents.specialists import ALLOWED_TOOLS, build_collective, build_specs
from iceclaw.config import ConfigError, Settings, load_settings
from iceclaw.contracts import EventKind


def kinds(rt, kind=None, agent=None):
    out = [
        e
        for e in rt.store
        if (kind is None or e.event_kind is kind) and (agent is None or e.agent_id == agent)
    ]
    return out


# --------------------------------------------------------------------------- policies
def measure_claim_conclude(tool, claim, **tool_args):
    """Policy: run one tool, publish a claim citing it, conclude citing both, then stop."""

    def step(n, ctx, ids):
        return None

    state = {}

    def policy(ctx):
        n = len(ctx.last_tool_results)
        if "start of the investigation" not in ctx.last_user_text:
            return quiet(ctx)
        if n == 0:
            return call("run_science_tool", tool_id=tool, **tool_args)
        if n == 1:
            state["m"] = first_event_id(ctx)
            return call("publish_claim", text=claim, cites=[state["m"]], status="supported")
        if n == 2:
            state["c"] = first_event_id(ctx)
            return call(
                "conclude", summary="done", cites=[state["m"], state["c"]], status="supported"
            )
        return "finished"

    return policy


BASIC = {
    THERMAL: measure_claim_conclude(
        "thermal_psr_consistency", "PSR pixels are colder than non-PSR."
    ),
    GEOLOGY: measure_claim_conclude("terrain_slope_summary", "Slopes are moderate."),
    AUDITOR: measure_claim_conclude(
        "valid_pixel_summary", "Prediction range is ordinary.", modality=PRED
    ),
    SKEPTIC: quiet,
}


def my_event(ctx, kind, agent):
    """Id of the latest event of ``kind`` authored by ``agent`` in the 'Your recent events' block."""
    block = ctx.last_user_text.split("Your recent events:")[-1].split("Events published")[0]
    ids = [m[0] for m in EVT.findall(block) if m[1] == agent and m[2] == kind]
    return ids[-1]


def seen(ctx, kind, agent):
    return [m[0] for m in EVT.findall(ctx.last_user_text) if m[1] == agent and m[2] == kind]


# =========================================================================== basic collective
def test_four_real_maf_agents_use_real_tools_and_publish_authentic_events():
    rt = make_runtime()
    summary, clients = asyncio.run(run_with(rt, BASIC))
    assert summary.ended_reason == "QUIESCENT" and summary.problems == []
    by_agent = {
        a: [e.event_kind for e in kinds(rt, agent=a)] for a in (THERMAL, GEOLOGY, AUDITOR, SKEPTIC)
    }
    for a in (THERMAL, GEOLOGY, AUDITOR):
        assert by_agent[a] == [EventKind.MEASUREMENT, EventKind.CLAIM, EventKind.COMPLETION], a
    assert by_agent[SKEPTIC] == []  # woken by claims, found nothing warranted: no event invented
    last = list(rt.store)[-1]
    assert last.agent_id == "supervisor" and last.claim_or_request == "run ended: QUIESCENT"
    # tool routing went through MAF's real function loop: results were fed back to the model
    for a in (THERMAL, GEOLOGY, AUDITOR):  # MAF fed each tool result back to the model
        results = [parse(r) for r in clients[a].tool_results]
        assert results and all(r["ok"] for r in results), a
        assert any(r.get("tool_run_id") for r in results), a
    assert len(rt.store.tool_runs()) == 3 and rt.tool_calls == 3
    assert {r["agent_id"] for r in rt.store.tool_runs()} == {THERMAL, GEOLOGY, AUDITOR}
    # each claim cites its own measurement and inherits provenance
    for a in (THERMAL, GEOLOGY, AUDITOR):
        m, c, _ = kinds(rt, agent=a)
        assert c.parent_event_ids == [m.event_id] and c.source_artifact_ids == m.source_artifact_ids
    # model calls were metered and stored (3 turns x 4 roundtrips... exact count is recorded, not guessed)
    rows = rt.store.llm_calls()
    assert summary.llm_calls == len(rows) == sum(c._n for c in clients.values())
    assert all(r["provider"] == "scripted" and r["ok"] == 1 and r["request_json"] for r in rows)
    assert summary.input_tokens > 0 and summary.total_tokens == sum(r["total_tokens"] for r in rows)


def test_skeptic_is_idle_until_woken_and_never_invents_a_debate():
    rt = make_runtime()
    pol = {THERMAL: quiet, GEOLOGY: quiet, AUDITOR: quiet, SKEPTIC: quiet}
    summary, clients = asyncio.run(run_with(rt, pol))
    assert summary.ended_reason == "QUIESCENT"
    assert clients[SKEPTIC]._n == 0  # nothing published => the skeptic never spent a model call
    assert [e.agent_id for e in rt.store] == ["supervisor"]
    assert summary.rounds[SKEPTIC] == 0 and rt.tool_calls == 0


def test_no_conflict_run_has_no_challenge_or_revision():
    rt = make_runtime()
    summary, _ = asyncio.run(run_with(rt, BASIC))
    assert not kinds(rt, EventKind.CHALLENGE) and not kinds(rt, EventKind.REVISION)
    assert summary.rounds[SKEPTIC] >= 1  # it did look (woken by claims) and chose to say nothing


# ====================================================================== cross-agent reaction
def skeptic_challenging():
    state = {"challenged": False, "evidence": None}

    def policy(ctx):
        n = len(ctx.last_tool_results)
        claims = seen(ctx, "claim", THERMAL)
        revs = seen(ctx, "revision", THERMAL)
        if revs and not state.get("closed"):
            state["closed"] = True
            return (
                call(
                    "conclude",
                    summary="Revision cites new evidence; no further contradiction.",
                    cites=revs,
                    status="supported",
                )
                if n == 0
                else "ok"
            )
        if claims and not state["challenged"]:
            if n == 0:
                return call("run_science_tool", tool_id="quantiles", modality=TEMP)
            if n == 1:
                state["challenged"] = True
                state["evidence"] = first_event_id(ctx)
                return call("publish_challenge", target_event_id=claims[-1],
                            text="The 110 K threshold is arbitrary; the temperature spread is wide.",
                            evidence_ids=[state["evidence"]], alternatives=["threshold choice"])  # fmt: skip
            return "challenged"
        return quiet(ctx) if n == 0 else "ok"

    return policy


def thermal_reacting():
    first = measure_claim_conclude("thermal_psr_consistency", "PSR is colder.", cold_trap_k=110.0)

    def policy(ctx):
        if "start of the investigation" in ctx.last_user_text:
            return first(ctx)
        chs = seen(ctx, "challenge", SKEPTIC)
        n = len(ctx.last_tool_results)
        if chs:
            if n == 0:
                return call(
                    "run_science_tool", tool_id="thermal_psr_consistency", cold_trap_k=130.0
                )
            if n == 1:
                return call("publish_revision", revised_event_id=my_event(ctx, "claim", THERMAL),
                            new_evidence_ids=[first_event_id(ctx)], responding_to=chs,
                            text="The PSR cold fraction depends strongly on the chosen threshold.",
                            status="unresolved", limitations=["threshold is a screening parameter"])  # fmt: skip
            return "revised"
        return quiet(ctx)

    return policy


def test_evidence_triggered_challenge_and_linked_revision_through_maf():
    rt = make_runtime()
    pol = {**BASIC, THERMAL: thermal_reacting(), SKEPTIC: skeptic_challenging()}
    summary, clients = asyncio.run(run_with(rt, pol))
    assert summary.ended_reason == "QUIESCENT" and summary.problems == []
    claim = next(e for e in kinds(rt, EventKind.CLAIM, THERMAL))
    ch = kinds(rt, EventKind.CHALLENGE)
    rev = kinds(rt, EventKind.REVISION)
    assert len(ch) == 1 and len(rev) == 1
    ch, rev = ch[0], rev[0]
    assert ch.agent_id == SKEPTIC and ch.parent_event_ids[0] == claim.event_id
    assert rev.agent_id == THERMAL
    assert rev.parent_event_ids[0] == claim.event_id and ch.event_id in rev.parent_event_ids
    new_ev = [
        p for p in rev.parent_event_ids if rt.store.get_event(p).event_kind is EventKind.MEASUREMENT
    ]
    assert new_ev and all(rt.store.seq_of(p) > rt.store.seq_of(claim.event_id) for p in new_ev)
    hops = [rt.store.hop_of(x) for x in (claim.event_id, ch.event_id, rev.event_id)]
    assert hops == [1, 2, 3]  # measurement(0) -> claim -> challenge -> revision
    # thermal's reaction turn really saw the skeptic's published challenge text
    assert any("threshold is arbitrary" in p for p in clients[THERMAL].prompts)
    assert any(
        claim.claim_or_request in p for p in clients[SKEPTIC].prompts
    )  # skeptic saw the claim
    # the earlier claim is preserved, never overwritten
    assert rt.store.get_event(claim.event_id).claim_or_request == "PSR is colder."
    assert [m.agent_id for m in kinds(rt, EventKind.MEASUREMENT)].count(SKEPTIC) == 1


def test_test_request_chain_across_agents_through_maf():
    """Auditor asks thermal for a test; thermal answers with in_response_to; results link back."""
    rt = make_runtime()

    def auditor(ctx):
        n = len(ctx.last_tool_results)
        if "start of the investigation" in ctx.last_user_text:
            if n == 0:
                return call("request_test", to_agent=THERMAL, tool_id="thermal_psr_consistency",
                            rationale="check cold fraction at a lower threshold", params={"cold_trap_k": 95.0})  # fmt: skip
            return "asked"
        return quiet(ctx)

    def thermal(ctx):
        n = len(ctx.last_tool_results)
        reqs = seen(ctx, "test_request", AUDITOR)
        if reqs:
            if n == 0:
                return call("run_science_tool", tool_id="thermal_psr_consistency",
                            in_response_to=reqs[-1], cold_trap_k=95.0)  # fmt: skip
            return "answered"
        return quiet(ctx)

    pol = {THERMAL: thermal, GEOLOGY: quiet, AUDITOR: auditor, SKEPTIC: quiet}
    summary, _ = asyncio.run(run_with(rt, pol))
    assert summary.ended_reason == "QUIESCENT" and summary.problems == []
    (req,), (res,) = kinds(rt, EventKind.TEST_REQUEST), kinds(rt, EventKind.TEST_RESULT)
    assert req.agent_id == AUDITOR and res.agent_id == THERMAL
    assert res.parent_event_ids == [req.event_id] and res.tool_run_id
    assert json.loads(req.claim_or_request)["to"] == THERMAL
    assert rt.state[THERMAL].rounds == 2  # first turn (nothing to do) + woken by the request


# ================================================================================= budgets
def test_llm_call_ceiling_is_hard_across_the_maf_loop():
    rt = make_runtime(max_llm_calls=5)

    def greedy(ctx):
        return call(
            "run_science_tool",
            tool_id="valid_pixel_summary",
            modality=PRED,
            window=[0, 4 + ctx.roundtrip, 0, 4],
        )

    pol = {THERMAL: greedy, GEOLOGY: greedy, AUDITOR: greedy, SKEPTIC: quiet}
    summary, clients = asyncio.run(run_with(rt, pol))
    assert summary.llm_calls == 5 and len(rt.store.llm_calls()) == 5
    assert summary.ended_reason == "BUDGET_EXHAUSTED"
    assert sum(c._n for c in clients.values()) == 5  # no model call was made past the ceiling
    assert any(r and r.startswith("BUDGET_") for r in summary.exit_reasons.values())
    assert summary.problems == []


def test_tool_call_ceiling_through_maf_agents_get_structured_refusals():
    rt = make_runtime(max_tool_calls=2)
    results = []

    def greedy(ctx):
        n = len(ctx.last_tool_results)
        if "start of the investigation" not in ctx.last_user_text or n >= 4:
            return "stop"
        results.extend(ctx.last_tool_results[-1:])
        return call("run_science_tool", tool_id="quantiles", modality=PRED, qs=[0.1 * (n + 1)])

    pol = {THERMAL: quiet, GEOLOGY: quiet, AUDITOR: greedy, SKEPTIC: quiet}
    summary, _ = asyncio.run(run_with(rt, pol))
    assert rt.tool_calls == 2 and summary.tool_calls == 2 and len(rt.store.tool_runs()) == 2
    assert any("TOOL_BUDGET_EXHAUSTED" in r for r in results)
    assert summary.problems == []


def test_event_budget_is_hard_and_supervisor_closing_event_is_reserved():
    rt = make_runtime(max_events=3)
    pol = {
        THERMAL: measure_claim_conclude("thermal_psr_consistency", "c"),
        GEOLOGY: measure_claim_conclude("terrain_slope_summary", "c"),
        AUDITOR: measure_claim_conclude("valid_pixel_summary", "c", modality=PRED),
        SKEPTIC: quiet,
    }
    summary, _ = asyncio.run(run_with(rt, pol))
    agent_events = [e for e in rt.store if e.agent_id != "supervisor"]
    assert len(agent_events) == 3  # agents were stopped at the ceiling
    assert list(rt.store)[-1].agent_id == "supervisor"  # the closing record is always written
    assert summary.ended_reason == "QUIESCENT"


def test_round_cap_and_deterministic_quiescence():
    rt = make_runtime(max_rounds=1)
    pol = {**BASIC, THERMAL: thermal_reacting(), SKEPTIC: skeptic_challenging()}
    summary, _ = asyncio.run(run_with(rt, pol))
    assert all(r <= 1 for r in summary.rounds.values())
    assert summary.exit_reasons[SKEPTIC] in (None, "MAX_ROUNDS", "RUN_ENDED")
    assert summary.ended_reason in ("QUIESCENT", "BUDGET_EXHAUSTED") and summary.problems == []
    assert not kinds(rt, EventKind.REVISION)  # thermal had no round left to react


def test_wall_clock_timeout_cancels_everything_and_is_recorded():
    rt = make_runtime(timeout_s=0.4)

    class Slow(ScriptedChatClient):
        def _inner_get_response(self, *, messages, stream, options, **kw):
            real = super()._inner_get_response(
                messages=messages, stream=stream, options=options, **kw
            )

            async def slow():
                try:
                    await asyncio.sleep(5)
                    return await real
                finally:
                    real.close()

            return slow()

    from iceclaw.agents.specialists import build_prompt

    def factory(runtime, aid):
        c = Slow(quiet, aid)
        c.bind(runtime, aid)
        return c

    runners = build_collective(rt, factory)

    async def go():
        s = await rt.run(runners, build_prompt)
        await asyncio.sleep(0)
        return s, [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    summary, leftover = asyncio.run(go())
    assert summary.ended_reason == "TIMEOUT" and summary.elapsed_s < 3
    assert list(rt.store)[-1].claim_or_request == "run ended: TIMEOUT"
    assert leftover == []  # no orphaned agent tasks


def test_agent_turn_errors_are_recorded_and_retried_once():
    rt = make_runtime()
    boom = {"n": 0}

    def flaky(ctx):
        if boom["n"] == 0:
            boom["n"] += 1
            raise RuntimeError("simulated provider outage")
        return quiet(ctx)

    always = lambda ctx: (_ for _ in ()).throw(RuntimeError("down"))  # noqa: E731
    pol = {THERMAL: flaky, GEOLOGY: always, AUDITOR: quiet, SKEPTIC: quiet}
    summary, _ = asyncio.run(run_with(rt, pol))
    fails = kinds(rt, EventKind.FAILURE)
    assert [f.agent_id for f in fails].count(THERMAL) == 1  # recovered on the retry
    assert [f.agent_id for f in fails].count(GEOLOGY) == 2  # gave up after two attempts
    assert all(f.error_code == "AGENT_TURN_ERROR" for f in fails)
    assert summary.exit_reasons[GEOLOGY] == "TURN_ERROR"
    assert summary.exit_reasons[THERMAL] != "TURN_ERROR"  # recovered
    bad = [r for r in rt.store.llm_calls() if r["ok"] == 0]
    assert len(bad) == 3 and all("RuntimeError" in r["error"] for r in bad)
    assert summary.problems == []


# ================================================================== schemas / permissions / firewall
def test_maf_tool_schemas_expose_no_paths_files_or_evaluator_handles():
    rt = make_runtime()
    allowed_props = {
        "tool_id", "modality", "window", "in_response_to", "k_mad", "top_n", "edge_width_px", "qs",
        "psr_threshold", "cold_trap_k", "steep_deg", "text", "cites", "evidence_type", "status",
        "alternatives", "limitations", "target_event_id", "evidence_ids", "to_agent", "rationale",
        "params", "revised_event_id", "new_evidence_ids", "responding_to", "summary", "kinds",
        "since_event_id", "event_id", "tool_run_id",
    }  # fmt: skip
    from iceclaw.agents.tools import make_tools

    for spec in rt.specs.values():
        names = []
        for t in make_tools(rt, spec):
            props = set(t.parameters().get("properties", {}))
            names.append(t.name)
            assert props <= allowed_props, (spec.agent_id, t.name, props - allowed_props)
            assert not any(
                re.search(r"path|file|dir|root|url|ref", p)
                for p in props
                if p not in {"in_response_to", "evidence_ids"}
            )
        assert ("recompute_measurement" in names) == (spec.agent_id == SKEPTIC)
        assert {"run_science_tool", "publish_claim", "conclude"} <= set(names)


def test_role_tool_and_layer_matrix():
    assert "terrain_slope_summary" not in ALLOWED_TOOLS[THERMAL]
    assert "thermal_psr_consistency" not in ALLOWED_TOOLS[GEOLOGY]
    assert (
        "boundary_check" in ALLOWED_TOOLS[AUDITOR]
        and "boundary_check" not in ALLOWED_TOOLS[THERMAL]
    )
    specs = {s.agent_id: s for s in build_specs()}
    assert specs[SKEPTIC].starts_idle and specs[SKEPTIC].can_recompute
    assert specs[SKEPTIC].allowed_modalities == {PRED, TEMP, "pixelwise_shadow_mask", TERRAIN}
    assert not any(s.starts_idle or s.can_recompute for a, s in specs.items() if a != SKEPTIC)
    assert len({s.agent_id for s in specs.values()}) == 4


FORBIDDEN = [
    "PRO",
    "reference",
    "sealed",
    "evaluation",
    "fault",
    "salt",
    "expected_",
    ".webp",
    "/home/",
    "sk-",
]


def test_prompts_and_tool_results_leak_nothing_from_the_evaluator_side():
    rt = make_runtime()
    pol = {**BASIC, THERMAL: thermal_reacting(), SKEPTIC: skeptic_challenging()}
    _, clients = asyncio.run(run_with(rt, pol))
    blob = "\n".join("\n".join(c.prompts) for c in clients.values())
    blob += "\n".join(r["request_json"] + (r["response_json"] or "") for r in rt.store.llm_calls())
    for tok in FORBIDDEN:
        pattern = rf"\b{tok}\b" if tok == "PRO" else re.escape(tok)
        assert not re.search(pattern, blob), tok
    ev_text = "\n".join(e.model_dump_json() for e in rt.store)
    for tok in FORBIDDEN:
        pattern = rf"\b{tok}\b" if tok == "PRO" else re.escape(tok)
        assert not re.search(pattern, ev_text), tok


def _imported(path):
    import ast

    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out |= {n.module, *(f"{n.module}.{a.name}" for a in n.names)}
    return out


def test_agent_side_modules_do_not_touch_files_or_evaluation():
    src = Path(__file__).resolve().parents[1] / "src" / "iceclaw"
    for f in list((src / "agents").glob("*.py")) + list((src / "swarm").glob("*.py")):
        imports = _imported(f)
        assert not any(i.startswith("iceclaw.evaluation") for i in imports), f
        assert "reference_path" not in f.read_text(), f
    for f in ("agents/tools.py", "agents/specialists.py", "agents/provider.py"):
        imports = _imported(src / f)
        assert not ({"os", "pathlib", "shutil", "glob", "subprocess", "builtins"} & imports), f
        assert "open(" not in (src / f).read_text(), f


def test_tool_actions_run_on_the_event_loop_thread_not_worker_threads():
    """MAF runs sync tools in threads; ours are coroutines so runtime state is never raced."""
    import inspect
    import threading

    from iceclaw.agents.tools import make_tools

    rt = make_runtime()
    assert all(inspect.iscoroutinefunction(t.func) for t in make_tools(rt, rt.specs[THERMAL]))
    threads = set()
    orig = rt.science_tool

    def spy(*a, **k):
        threads.add(threading.get_ident())
        return orig(*a, **k)

    rt.science_tool = spy  # type: ignore[method-assign]
    loop_thread = {}

    async def go():
        loop_thread["id"] = threading.get_ident()
        await run_with(rt, BASIC)

    asyncio.run(go())
    assert threads == {loop_thread["id"]}


# =================================================================================== provider
def test_openai_client_requires_runtime_settings_and_never_guesses_a_model():
    rt = make_runtime()
    no_key = load_settings({"ICECLAW_LLM_MODEL": "m"})
    with pytest.raises(ConfigError) as e:
        build_openai_client(no_key, rt, THERMAL, rt.limits)
    assert e.value.code == "MISSING_API_KEY"
    no_model = load_settings({"OPENAI_API_KEY": "k"})
    with pytest.raises(ConfigError) as e:
        build_openai_client(no_model, rt, THERMAL, rt.limits)
    assert e.value.code == "MISSING_LLM_MODEL"
    wrong = load_settings(
        {"OPENAI_API_KEY": "k", "ICECLAW_LLM_MODEL": "m", "ICECLAW_LLM_PROVIDER": "other"}
    )
    with pytest.raises(ConfigError) as e:
        build_openai_client(wrong, rt, THERMAL, rt.limits)
    assert e.value.code == "UNSUPPORTED_PROVIDER"


def test_openai_client_construction_is_offline_bounded_and_does_not_expose_the_key():
    rt = make_runtime(max_llm_calls=7)
    s = load_settings(
        {"OPENAI_API_KEY": "sk-test-NOT-A-REAL-KEY-123", "ICECLAW_LLM_MODEL": "model-x"}
    )
    c = build_openai_client(s, rt, GEOLOGY, rt.limits)
    from iceclaw.agents.provider import InstrumentedOpenAIChatClient

    assert isinstance(c, InstrumentedOpenAIChatClient) and c.model == "model-x"
    cfg = c.function_invocation_configuration
    assert cfg["max_iterations"] == rt.limits.max_roundtrips_per_turn
    assert cfg["max_function_calls"] == rt.limits.max_function_calls_per_turn
    assert "NOT-A-REAL-KEY" not in repr(c) and "NOT-A-REAL-KEY" not in repr(s)
    assert c._agent_id == GEOLOGY and c._runtime is rt


def test_secrets_never_reach_stored_records(tmp_path):
    os.environ["OPENAI_API_KEY"] = "sk-test-SECRET-MARKER-999"
    try:
        rt = make_runtime(store=None)
        asyncio.run(run_with(rt, BASIC))
        rt.write_outputs(tmp_path, rt.summary("QUIESCENT", 0.0))
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
    for p in tmp_path.iterdir():
        assert "SECRET-MARKER" not in p.read_text(), p.name
    assert {p.name for p in tmp_path.iterdir()} == {
        "events.jsonl", "tool_runs.json", "llm_calls.jsonl", "run_summary.json"
    }  # fmt: skip


def test_outputs_are_machine_verifiable(tmp_path):
    rt = make_runtime()
    pol = {**BASIC, THERMAL: thermal_reacting(), SKEPTIC: skeptic_challenging()}
    summary, _ = asyncio.run(run_with(rt, pol))
    rt.write_outputs(tmp_path, summary)
    from iceclaw.contracts import EvidenceEvent

    events = [
        EvidenceEvent.model_validate_json(x)
        for x in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    ids = [e.event_id for e in events]
    assert len(set(ids)) == len(ids) == summary.events
    seen_ids: set[str] = set()
    for e in events:  # every parent precedes its child; challenge/revision always have parents
        assert set(e.parent_event_ids) <= seen_ids
        if e.event_kind in (EventKind.CHALLENGE, EventKind.REVISION):
            assert e.parent_event_ids
        seen_ids.add(e.event_id)
    runs = json.loads((tmp_path / "tool_runs.json").read_text())
    tool_event_ids = {e.event_id for e in events if e.tool_run_id}
    assert {r["event_id"] for r in runs} == tool_event_ids
    assert json.loads((tmp_path / "run_summary.json").read_text())["problems"] == []
    for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines():
        assert json.loads(line)["provider"] == "scripted"


# ==================================================================================== real region
REPO = Path(__file__).resolve().parents[1]
SMOKE = Path(
    os.environ.get(
        "ICECLAW_MOONATLAS_SMOKE_ROOT", REPO / ".upstream/moonatlas-science/examples/science-smoke"
    )
)
SOM = Path(
    os.environ.get("ICECLAW_SOMBENCH_TEST_ROOT", REPO / "data/sombench/prospectivity_dataset")
)
needs_real = pytest.mark.skipif(
    not ((SMOKE / "manifest.json").is_file() and (SOM / "download_manifest.json").is_file()),
    reason="real MOONATLAS / SomBench data not present",
)


def real_settings(som):
    return Settings(
        data_root=Path("/nonexistent-iceclaw-data"), runs_root=Path("/nonexistent-iceclaw-runs"),
        moonatlas_roots=(SMOKE,), sombench_root=som,
    )  # fmt: skip


@needs_real
def test_only_the_verified_paired_region_is_investigable():
    from iceclaw.data.catalog import PredictionUnavailable, RegionNotSupported
    from iceclaw.swarm.runtime import RegionNotInvestigable, load_paired_region

    b = make_runtime().limits.budget
    ctx, arrays = load_paired_region(real_settings(SOM), "ice-s-0001-0002", b)
    assert ctx.region_id == "ice-s-0001-0002" and set(arrays.layers) >= {PRED, TEMP, TERRAIN}
    assert {"thermal_psr_consistency", "terrain_slope_summary"} <= set(ctx.available_tools)
    with pytest.raises(PredictionUnavailable):  # one of the 11 observation-only tiles
        load_paired_region(real_settings(SOM), "ice-s-0003-0002", b)
    with pytest.raises(RegionNotSupported):
        load_paired_region(real_settings(SOM), "ice-s-9999-9999", b)
    with pytest.raises(RegionNotInvestigable):  # prediction-only (no observations configured)
        load_paired_region(real_settings(None), "ice-s-0001-0002", b)


@needs_real
def test_real_region_with_scripted_models_cites_real_geotiff_sources_and_numbers():
    import numpy as np
    import rasterio

    from iceclaw.agents.specialists import build_specs
    from iceclaw.data import moonatlas_adapter as ma
    from iceclaw.swarm.runtime import RunLimits, SwarmRuntime, load_paired_region
    from iceclaw.swarm.store import EvidenceStore

    b = make_runtime().limits.budget
    ctx, arrays = load_paired_region(real_settings(SOM), "ice-s-0001-0002", b)
    rt = SwarmRuntime(investigation_id="inv-real", context=ctx, arrays=arrays, store=EvidenceStore(),
                      specs=build_specs(), limits=RunLimits(budget=b), provider="scripted", model="v0")  # fmt: skip
    pol = {
        THERMAL: measure_claim_conclude("thermal_psr_consistency", "PSR vs non-PSR measured."),
        GEOLOGY: measure_claim_conclude("terrain_slope_summary", "Slope measured."),
        AUDITOR: measure_claim_conclude(
            "valid_pixel_summary", "Prediction summarized.", modality=PRED
        ),
        SKEPTIC: quiet,
    }
    summary, clients = asyncio.run(run_with(rt, pol))
    assert summary.ended_reason == "QUIESCENT" and summary.problems == []
    ms = {e.agent_id: e for e in kinds(rt, EventKind.MEASUREMENT)}
    runs = {r["agent_id"]: r["payload"] for r in rt.store.tool_runs()}
    t = runs[THERMAL]
    assert t["value"]["psr_mean_k"] == pytest.approx(139.21508, abs=1e-4)
    assert t["value"]["psr_minus_non_psr_k"] == pytest.approx(-107.77419, abs=1e-4)
    assert runs[GEOLOGY]["value"]["mean_deg"] == pytest.approx(8.734015, abs=1e-5)
    assert runs[AUDITOR]["value"]["max"] == pytest.approx(1.015736, abs=1e-5)
    # sources are the ORIGINAL GeoTIFF/prediction files, with their real checksums
    assert set(ms[THERMAL].source_artifact_ids) == {
        "sombench:patch_0001_0002_S_80S_LPSR.tif", "sombench:patch_0001_0002_S_80S_TMAX.tif"
    }  # fmt: skip
    for k, sha in t["source_sha256"].items():
        assert ma.sha256_file(SOM / k.removeprefix("sombench:")) == sha
    with rasterio.open(SOM / "patch_0001_0002_S_80S_SLOPE.tif") as ds:
        sl = ds.read(1).astype(np.float64)
    assert runs[GEOLOGY]["value"]["max_deg"] == pytest.approx(float(sl.max()), abs=1e-9)
    # nothing absolute or evaluator-related reached any model prompt
    blob = "\n".join("\n".join(c.prompts) for c in clients.values())
    assert str(SOM) not in blob and str(SMOKE) not in blob and "/home/" not in blob
    assert not re.search(r"\bPRO\b", blob) and "reference" not in blob.lower()


@needs_real
def test_skeptic_recompute_cross_check_on_real_measurement():
    from iceclaw.swarm.runtime import RunLimits, SwarmRuntime, load_paired_region
    from iceclaw.swarm.store import EvidenceStore

    b = make_runtime().limits.budget
    ctx, arrays = load_paired_region(real_settings(SOM), "ice-s-0001-0002", b)
    rt = SwarmRuntime(investigation_id="inv-real2", context=ctx, arrays=arrays, store=EvidenceStore(),
                      specs=build_specs(), limits=RunLimits(budget=b))  # fmt: skip
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    r = rt.recompute_measurement(SKEPTIC, m["tool_run_id"])
    assert r["ok"] and r["reproduced"]
    assert rt.store.verify() == []
    _ = parse
