"""Store + runtime rules, exercised directly (no model): validation, budgets, hops, delivery.

Arrays are SYNTHETIC (see agent_helpers); real-region integration lives in test_agents.py.
"""

import json
import sqlite3
from datetime import UTC, datetime

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
    make_runtime,
)

from iceclaw.contracts import EventKind, EvidenceEvent
from iceclaw.swarm.store import EvidenceStore, StoreError


def ev(i: str, kind="claim", parents=(), **kw):
    base = dict(
        event_id=i,
        investigation_id="inv-test",
        region_id="r",
        agent_id="a",
        timestamp_utc=datetime.now(UTC),
        event_kind=kind,
        parent_event_ids=list(parents),
        claim_or_request="x",
    )
    base.update(kw)
    return EvidenceEvent(**base)


# ----------------------------------------------------------------------------- store
def test_store_is_append_only_even_for_raw_sql():
    rt = make_runtime()  # populate EVERY table: triggers act per row, so empty tables prove nothing
    s = rt.store
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    rt.publish_claim(THERMAL, "c", [m["event_id"]])  # creates an event_parents row
    rt.llm_record(THERMAL, started=datetime.now(UTC), request=[], response={}, error=None, usage={})
    n_events = s.count_events()
    for table in ("events", "event_parents", "tool_runs", "llm_calls"):
        assert s._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] >= 1, table
        for stmt in (f"DELETE FROM {table}", f"UPDATE {table} SET rowid = rowid + 1000"):
            with pytest.raises(sqlite3.DatabaseError):
                s._db.execute(stmt)
    with pytest.raises(sqlite3.DatabaseError):
        s._db.execute("UPDATE events SET agent_id='forged'")
    assert s.count_events() == n_events and s.get_event(m["event_id"]).agent_id == THERMAL


def test_store_parent_validation_duplicates_and_hops():
    s = EvidenceStore()
    s.append_event(ev("e1"), 0)
    with pytest.raises(StoreError) as e:
        s.append_event(ev("e2", "challenge", ["nope"]), 1)
    assert e.value.code == "UNKNOWN_PARENT"
    with pytest.raises(StoreError) as e:
        s.append_event(ev("e1"), 0)
    assert e.value.code == "DUPLICATE_EVENT_ID"
    s.append_event(ev("e2", "challenge", ["e1"]), 1)
    assert s.hop_of("e2") == 1 and s.seq_of("e2") > s.seq_of("e1")
    assert s.verify() == []
    assert [e.event_id for e in s] == ["e1", "e2"]


def test_store_persists_to_file_and_exports_replayable_jsonl(tmp_path):
    p = tmp_path / "events.sqlite"
    s = EvidenceStore(p)
    s.append_event(ev("e1"), 0)
    s.append_event(ev("e2", "challenge", ["e1"]), 1)
    out = tmp_path / "events.jsonl"
    assert s.export_jsonl(out) == 2
    lines = [EvidenceEvent.model_validate_json(x) for x in out.read_text().splitlines()]
    assert [e.event_id for e in lines] == ["e1", "e2"]
    s.close()
    again = EvidenceStore(p)  # reopen: history is durable
    assert again.count_events() == 2 and again.verify() == []


# -------------------------------------------------------------------- science tool actions
def test_science_tool_publishes_authentic_measurement_event_and_tool_run():
    rt = make_runtime()
    r = rt.science_tool(AUDITOR, "valid_pixel_summary", PRED)
    assert r["ok"] and not r["cached"] and r["valid_pixel_count"] == 32 * 32
    e = rt.store.get_event(r["event_id"])
    assert e.event_kind is EventKind.MEASUREMENT and e.agent_id == AUDITOR
    assert e.tool_id == "valid_pixel_summary" and e.tool_run_id == r["tool_run_id"]
    assert e.source_artifact_ids == [f"synthetic:{PRED}"]
    m, ev_id, agent = rt.store.get_tool_run(r["tool_run_id"])
    assert (
        ev_id == e.event_id
        and agent == AUDITOR
        and m.source_sha256 == {f"synthetic:{PRED}": "a" * 64}
    )
    assert rt.tool_calls == 1 and rt.store.verify() == []


def test_identical_tool_call_is_deduplicated_and_free():
    rt = make_runtime()
    a = rt.science_tool(THERMAL, "thermal_psr_consistency")
    b = rt.science_tool(THERMAL, "thermal_psr_consistency")
    assert b["cached"] and b["event_id"] == a["event_id"] and rt.tool_calls == 1
    c = rt.science_tool(THERMAL, "thermal_psr_consistency", cold_trap_k=130.0)
    assert not c["cached"] and rt.tool_calls == 2 and c["event_id"] != a["event_id"]


@pytest.mark.parametrize(
    "agent,tool,modality,code",
    [
        (THERMAL, "terrain_slope_summary", None, "TOOL_NOT_PERMITTED"),
        (GEOLOGY, "thermal_psr_consistency", None, "TOOL_NOT_PERMITTED"),
        (GEOLOGY, "valid_pixel_summary", PRED, "MODALITY_NOT_ALLOWED"),
        (THERMAL, "valid_pixel_summary", TERRAIN, "MODALITY_NOT_ALLOWED"),
        (AUDITOR, "valid_pixel_summary", None, "MODALITY_REQUIRED"),
        (THERMAL, "thermal_psr_consistency", TEMP, "MODALITY_NOT_ALLOWED"),
        (THERMAL, "not_a_tool", None, "UNKNOWN_TOOL"),
        (AUDITOR, "valid_pixel_summary", "/etc/passwd", "MODALITY_NOT_ALLOWED"),
    ],
)
def test_role_permissions_are_enforced(agent, tool, modality, code):
    rt = make_runtime()
    r = rt.science_tool(agent, tool, modality)
    assert not r["ok"] and r["error"] == code
    assert rt.tool_calls == 0 and rt.store.count_events() == 0  # rejected calls cost nothing


def test_tool_parameter_validation():
    rt = make_runtime()
    bad = [
        dict(tool_id="thermal_psr_consistency", steep_deg=10.0),  # wrong tool's parameter
        dict(tool_id="thermal_psr_consistency", cold_trap_k=-5.0),
        dict(tool_id="thermal_psr_consistency", cold_trap_k=1e9),
        dict(tool_id="valid_pixel_summary", modality=TEMP, k_mad=3.0),
        dict(tool_id="quantiles", modality=TEMP, qs=[1.5]),
        dict(tool_id="valid_pixel_summary", modality=TEMP, window=[0, 1]),
    ]
    for kw in bad:
        tid = kw.pop("tool_id")
        r = rt.science_tool(THERMAL, tid, **kw)
        assert not r["ok"] and r["error"] in ("BAD_PARAM", "BAD_WINDOW"), (tid, kw, r)
    assert rt.tool_calls == 0
    ok = rt.science_tool(THERMAL, "quantiles", TEMP, qs=[0.1, 0.9])
    assert ok["ok"] and set(ok["value"]) == {"q0.1", "q0.9"}


def test_window_out_of_coverage_is_a_structured_measurement_not_a_guess():
    rt = make_runtime()
    r = rt.science_tool(AUDITOR, "valid_pixel_summary", PRED, window=[0, 99, 0, 5])
    assert r["ok"] and r["value"] is None and r["unavailable_reason"] == "OUT_OF_COVERAGE"
    e = rt.store.get_event(r["event_id"])
    assert e.error_code == "OUT_OF_COVERAGE" and e.limitations == ["unavailable: OUT_OF_COVERAGE"]


def test_tool_budget_ceiling_is_hard_and_graceful():
    rt = make_runtime(max_tool_calls=3)
    ids = []
    for k in (100.0, 110.0, 120.0, 130.0, 140.0):
        r = rt.science_tool(THERMAL, "thermal_psr_consistency", cold_trap_k=k)
        ids.append(r)
    assert [r["ok"] for r in ids] == [True, True, True, False, False]
    assert ids[3]["error"] == "TOOL_BUDGET_EXHAUSTED" and rt.tool_calls == 3
    assert len(rt.store.tool_runs()) == 3


def test_science_tool_failure_is_recorded_as_failure_event():
    rt = make_runtime()
    n = rt.arrays.layers[PRED].values.shape[0]
    # a 1-pixel window has no complete gradient stencil -> ScienceToolError NO_VALID_PIXELS
    r = rt.science_tool(AUDITOR, "gradient_stats", PRED, window=[0, 1, 0, 1])
    assert not r["ok"] and r["error"] == "NO_VALID_PIXELS" and n == 32
    e = rt.store.get_event(r["event_id"])
    assert e.event_kind is EventKind.FAILURE and e.error_code == "NO_VALID_PIXELS"
    assert rt.tool_calls == 1  # the attempt was real work and is counted


# ------------------------------------------------------------------ claims / debate rules
def seed(rt):
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    c = rt.publish_claim(THERMAL, "PSR is colder", [m["event_id"]], status="provisional")
    return m["event_id"], c["event_id"]


def test_claim_requires_evidence_unless_labelled_hypothesis():
    rt = make_runtime()
    r = rt.publish_claim(THERMAL, "PSR is colder", [])
    assert not r["ok"] and r["error"] == "EVIDENCE_REQUIRED"
    assert rt.publish_claim(THERMAL, "maybe cold traps", [], evidence_type="hypothesis")["ok"]
    assert rt.publish_claim(THERMAL, "x", ["evt-9999"])["error"] == "UNKNOWN_EVENT"
    assert rt.publish_claim(THERMAL, "x", [], evidence_type="bogus")["error"] == "BAD_ENUM"
    m, _ = seed(rt)
    assert rt.publish_claim(THERMAL, "y", [m], status="challenged")["error"] == "BAD_ENUM"


def test_claim_inherits_provenance_from_cited_measurement():
    rt = make_runtime()
    m, c = seed(rt)
    e = rt.store.get_event(c)
    me = rt.store.get_event(m)
    assert e.parent_event_ids == [m] and e.source_artifact_ids == me.source_artifact_ids
    assert e.spatial_footprint == me.spatial_footprint and e.analysis_resolution_m == 240.0


def test_challenge_rules_and_hops():
    rt = make_runtime()
    m, c = seed(rt)
    assert rt.publish_challenge(THERMAL, c, "self")["error"] == "SELF_CHALLENGE"
    assert rt.publish_challenge(SKEPTIC, "evt-9999", "x")["error"] == "UNKNOWN_EVENT"
    assert (
        rt.publish_challenge(SKEPTIC, m, "x")["error"] == "BAD_TARGET"
    )  # cannot challenge a measurement
    k = rt.science_tool(SKEPTIC, "quantiles", TEMP)
    ch = rt.publish_challenge(SKEPTIC, c, "tails look broad", [k["event_id"]], ["noise"])
    assert ch["ok"]
    e = rt.store.get_event(ch["event_id"])
    assert e.event_kind is EventKind.CHALLENGE and e.parent_event_ids == [c, k["event_id"]]
    assert e.hypothesis_status.value == "challenged"
    assert (
        rt.store.hop_of(m) == 0 and rt.store.hop_of(c) == 1 and rt.store.hop_of(ch["event_id"]) == 2
    )


def test_revision_requires_own_claim_and_NEW_evidence():
    rt = make_runtime()
    m, c = seed(rt)
    assert rt.publish_revision(THERMAL, c, [m], "same evidence")["error"] == "NEW_EVIDENCE_REQUIRED"
    assert rt.publish_revision(SKEPTIC, c, [m], "not mine")["error"] == "BAD_TARGET"
    new = rt.science_tool(THERMAL, "thermal_psr_consistency", cold_trap_k=130.0)
    ch = rt.publish_challenge(SKEPTIC, c, "threshold is arbitrary")
    r = rt.publish_revision(
        THERMAL, c, [new["event_id"]], "threshold-dependent", "unresolved", [ch["event_id"]]
    )
    assert r["ok"]
    e = rt.store.get_event(r["event_id"])
    assert e.event_kind is EventKind.REVISION
    assert e.parent_event_ids == [c, new["event_id"], ch["event_id"]]
    assert rt.store.seq_of(new["event_id"]) > rt.store.seq_of(c)
    assert (
        rt.publish_revision(THERMAL, "evt-9999", [new["event_id"]], "x")["error"] == "UNKNOWN_EVENT"
    )
    # a measurement older than the claim cannot qualify
    assert rt.publish_revision(THERMAL, r["event_id"], [m], "x")["error"] == "NEW_EVIDENCE_REQUIRED"


def test_test_request_and_result_chain():
    rt = make_runtime()
    m, c = seed(rt)
    bad = rt.request_test(SKEPTIC, GEOLOGY, "thermal_psr_consistency", "x")
    assert not bad["ok"] and "geology" in bad["message"]  # addressee cannot run it
    assert rt.request_test(SKEPTIC, SKEPTIC, "quantiles", "x", TEMP)["error"] == "BAD_ADDRESSEE"
    assert rt.request_test(SKEPTIC, "ghost", "quantiles", "x", TEMP)["error"] == "BAD_ADDRESSEE"
    req = rt.request_test(SKEPTIC, THERMAL, "thermal_psr_consistency", "check threshold", cites=[c],
                          cold_trap_k=125.0)  # fmt: skip
    assert req["ok"] and req["addressed_to"] == THERMAL
    re = rt.store.get_event(req["event_id"])
    assert re.event_kind is EventKind.TEST_REQUEST and re.tool_id == "thermal_psr_consistency"
    assert json.loads(re.claim_or_request)["params"] == {"cold_trap_k": 125.0}
    wrong = rt.science_tool(AUDITOR, "valid_pixel_summary", PRED, in_response_to=req["event_id"])
    assert wrong["error"] == "BAD_REQUEST_REF"  # addressed to thermal, not the auditor
    assert (
        rt.science_tool(THERMAL, "quantiles", TEMP, in_response_to=c)["error"] == "BAD_REQUEST_REF"
    )
    res = rt.science_tool(
        THERMAL, "thermal_psr_consistency", in_response_to=req["event_id"], cold_trap_k=125.0
    )
    assert res["ok"]
    e = rt.store.get_event(res["event_id"])
    assert e.event_kind is EventKind.TEST_RESULT and e.parent_event_ids == [req["event_id"]]
    assert rt.store.hop_of(res["event_id"]) == rt.store.hop_of(req["event_id"]) + 1
    assert rt.store.verify() == []


def test_event_and_hop_budgets_are_hard_ceilings():
    rt = make_runtime(max_events=3)
    assert rt.publish_claim(THERMAL, "h1", [], evidence_type="hypothesis")["ok"]
    assert rt.publish_claim(THERMAL, "h2", [], evidence_type="hypothesis")["ok"]
    assert rt.publish_claim(THERMAL, "h3", [], evidence_type="hypothesis")["ok"]
    r = rt.publish_claim(THERMAL, "h4", [], evidence_type="hypothesis")
    assert r["error"] == "EVENT_BUDGET_EXHAUSTED" and rt.store.count_events() == 3

    rt = make_runtime(max_hops=1)
    a = rt.publish_claim(THERMAL, "root", [], evidence_type="hypothesis")["event_id"]  # hop 0
    b = rt.publish_challenge(SKEPTIC, a, "c")["event_id"]  # hop 1
    new = rt.science_tool(THERMAL, "thermal_psr_consistency")["event_id"]  # hop 0, after the claim
    n_before = rt.store.count_events()
    # responding to the hop-1 challenge would make the revision hop 2 > max_hops
    r = rt.publish_revision(THERMAL, a, [new], "revised", responding_to=[b])
    assert r["error"] == "HOP_LIMIT" and rt.store.count_events() == n_before
    # without citing the challenge the revision stays at hop 1 and is allowed
    assert rt.publish_revision(THERMAL, a, [new], "revised")["ok"]


def test_duplicate_events_are_rejected():
    rt = make_runtime()
    a = rt.publish_claim(THERMAL, "same", [], evidence_type="hypothesis")
    b = rt.publish_claim(THERMAL, "Same ", [], evidence_type="hypothesis")
    assert b["error"] == "DUPLICATE" and b["existing_event_id"] == a["event_id"]


def test_recompute_measurement_is_skeptic_only_and_reproduces():
    rt = make_runtime()
    m = rt.science_tool(AUDITOR, "valid_pixel_summary", PRED)
    assert rt.recompute_measurement(THERMAL, m["tool_run_id"])["error"] == "TOOL_NOT_PERMITTED"
    assert rt.recompute_measurement(SKEPTIC, "nope")["error"] == "UNKNOWN_TOOL_RUN"
    r = rt.recompute_measurement(SKEPTIC, m["tool_run_id"])
    assert r["ok"] and r["reproduced"] and r["original_event_id"] == m["event_id"]
    e = rt.store.get_event(r["event_id"])
    assert e.agent_id == SKEPTIC and e.parent_event_ids == [m["event_id"]]
    assert rt.store.verify() == []
    assert rt.recompute_measurement(SKEPTIC, m["tool_run_id"])["error"] == "DUPLICATE"


# ------------------------------------------------------------------------------ delivery
def test_delivery_wakes_only_concerned_agents():
    rt = make_runtime()
    for a in rt.state.values():
        a.pending_first = False
    m, c = seed(rt)
    # a claim wakes only agents with wake_on_claims (the skeptic); others just receive it
    assert rt.state[SKEPTIC].wake_pending and not rt.state[GEOLOGY].wake_pending
    assert [e.event_id for e in rt.state[GEOLOGY].inbox] == [m, c]
    assert not rt.state[THERMAL].inbox  # never delivered to its own author
    for s in rt.state.values():
        s.wake_pending = False
    req = rt.request_test(SKEPTIC, GEOLOGY, "terrain_slope_summary", "check slope")
    assert rt.state[GEOLOGY].wake_pending and not rt.state[THERMAL].wake_pending
    for s in rt.state.values():
        s.wake_pending = False
    ch = rt.publish_challenge(SKEPTIC, c, "challenge thermal")
    assert rt.state[THERMAL].wake_pending and not rt.state[AUDITOR].wake_pending
    assert req["ok"] and ch["ok"]


def test_listing_and_reading_events():
    rt = make_runtime()
    m, c = seed(rt)
    lines = rt.list_events(SKEPTIC)
    assert any(m in x for x in lines) and any("claim" in x and c in x for x in lines)
    assert rt.list_events(SKEPTIC, kinds=["claim"]) == [x for x in lines if "claim" in x]
    detail = rt.get_event_detail(m)
    assert detail["ok"] and detail["measurement"]["tool_id"] == "thermal_psr_consistency"
    assert rt.get_event_detail("evt-0")["error"] == "UNKNOWN_EVENT"
    assert "synthetic:" not in json.dumps(detail["measurement"]["value"])


def test_llm_accounting_and_hard_ceiling():
    from iceclaw.swarm.runtime import BudgetExhausted

    rt = make_runtime(max_llm_calls=2)
    rt.llm_before_call(THERMAL)
    rt.llm_before_call(AUDITOR)
    with pytest.raises(BudgetExhausted) as e:
        rt.llm_before_call(GEOLOGY)
    assert e.value.kind == "LLM_CALLS" and rt.llm_calls == 2
    rt.llm_record(THERMAL, started=datetime.now(UTC), request=[{"role": "user"}], response={"x": 1},
                  error=None, usage={"input_token_count": 10, "output_token_count": 5, "total_token_count": 15})  # fmt: skip
    assert rt.tokens == {"input": 10, "output": 5, "total": 15}
    row = rt.store.llm_calls()[0]
    assert row["provider"] == "scripted" and row["total_tokens"] == 15 and row["ok"] == 1


def test_agent_ids_must_be_unique_and_not_supervisor():
    from agent_helpers import build_specs

    s = build_specs()
    with pytest.raises(ValueError):
        make_runtime(specs=[s[0], s[0]])


def test_unused_layer_constants_exist():
    assert SHADOW and TERRAIN and TEMP and PRED
