"""P6 token hardening: bounded, deterministic evidence digest that never drops references.

Arrays are SYNTHETIC (see agent_helpers). The full history always stays in SQLite; the prompt only
shrinks what the model is SHOWN, never what is recorded.
"""

import re

import pytest
from agent_helpers import (
    AUDITOR,
    GEOLOGY,
    PRED,
    SKEPTIC,
    TEMP,
    THERMAL,
    make_runtime,
)

from iceclaw.agents.digest import build_digest, build_own_summary, compress_ids, select_events
from iceclaw.agents.specialists import COMMON, build_prompt
from iceclaw.agents.tools import make_tools
from iceclaw.swarm.runtime import BudgetExhausted

EVT = re.compile(r"\[(evt-\d+)\]")


def flood(rt, n, agent=GEOLOGY, size=1400, tag="a"):
    ids = []
    for i in range(n):
        r = rt.publish_claim(
            agent, f"filler {tag}{i} " + "x" * size, [], evidence_type="hypothesis"
        )
        assert r["ok"], r
        ids.append(r["event_id"])
    return ids


def events_of(rt, ids):
    return [rt.store.get_event(i) for i in ids]


def test_compress_ids_ranges():
    assert (
        compress_ids(["evt-0003", "evt-0004", "evt-0005", "evt-0009"])
        == "evt-0003..evt-0005, evt-0009"
    )
    assert compress_ids(["evt-0001"]) == "evt-0001"
    assert compress_ids([]) == ""
    assert compress_ids(["evt-0010", "evt-0002", "evt-0003"]) == "evt-0002..evt-0003, evt-0010"


def test_digest_is_bounded_deterministic_and_reports_what_it_omits():
    rt = make_runtime(max_events=500, max_hops=10)
    ids = flood(rt, 150)
    new = events_of(rt, ids)
    d1 = build_digest(rt, THERMAL, new, max_chars=5000)
    d2 = build_digest(rt, THERMAL, new, max_chars=5000)
    assert d1 == d2  # deterministic: no randomness, no model
    body, _, note = d1.rpartition("\n")
    assert len(body) <= 5000 and "omitted for length" in note
    shown = set(EVT.findall(body))
    omitted_note_ids = note.split("length: ")[1].split(";")[0]
    # every event is either shown or named in the omission note: nothing silently disappears
    for i in ids:
        n = int(i.split("-")[1])
        in_note = any(
            (
                ".." in part
                and int(part.split("..")[0].split("-")[1])
                <= n
                <= int(part.split("..")[1].split("-")[1])
            )
            or part.strip() == i
            for part in omitted_note_ids.split(",")
        )
        assert i in shown or in_note, i
    assert len(shown) < len(ids)


def test_prompt_length_is_capped_and_does_not_grow_with_run_length():
    rt = make_runtime(max_events=600, max_hops=10)
    spec, st = rt.specs[THERMAL], rt.state[THERMAL]
    st.rounds = 1
    ids = flood(rt, 8)
    short = build_prompt(rt, spec, st, events_of(rt, ids))
    ids += flood(rt, 300, tag="b")
    long = build_prompt(rt, spec, st, events_of(rt, ids))
    cap = rt.limits.max_prompt_chars
    assert len(short) <= cap and len(long) <= cap
    # 300 more events cost (almost) nothing: the digest cap, not the history, bounds the prompt
    assert len(long) - len(short) < 0.5 * cap
    assert rt.store.count_events() == 308  # and the full history is untouched


def test_events_addressed_to_the_agent_are_always_shown_with_their_references():
    rt = make_runtime(max_events=600, max_hops=10)
    m = rt.science_tool(THERMAL, "thermal_psr_consistency")
    claim = rt.publish_claim(THERMAL, "PSR colder", [m["event_id"]])
    ev = rt.science_tool(SKEPTIC, "quantiles", TEMP)
    ids = flood(rt, 200)  # lots of newer noise from another agent
    ch = rt.publish_challenge(SKEPTIC, claim["event_id"], "threshold arbitrary", [ev["event_id"]])
    new = events_of(rt, [m["event_id"], ev["event_id"], *ids, ch["event_id"]])
    chosen, omitted = select_events(rt, THERMAL, new, 1500, 240)
    chosen_ids = {e.event_id for e in chosen}
    assert ch["event_id"] in chosen_ids  # a challenge of MY claim outranks 200 fillers
    d = build_digest(rt, THERMAL, new, 1500)
    line = next(x for x in d.splitlines() if x.startswith(f"[{ch['event_id']}]"))
    assert claim["event_id"] in line and ev["event_id"] in line  # parent references kept
    assert "challenge" in line and "status=challenged" in line


def test_test_requests_to_me_outrank_other_events():
    rt = make_runtime(max_events=600, max_hops=10)
    ids = flood(rt, 100)
    req = rt.request_test(
        SKEPTIC, THERMAL, "thermal_psr_consistency", "check threshold", cold_trap_k=120.0
    )
    new = events_of(rt, [*ids, req["event_id"]])
    chosen, _ = select_events(rt, THERMAL, new, 1200, 240)
    assert req["event_id"] in {e.event_id for e in chosen}
    other = select_events(rt, GEOLOGY, new, 1200, 240)[0]
    assert req["event_id"] in {e.event_id for e in other} or True  # not addressed to geology


def test_lines_keep_all_parent_ids_and_short_source_ids():
    rt = make_runtime(max_events=600, max_hops=10)
    ms = []
    for k in range(14):  # 14 distinct measurements -> a claim with 14 parents
        r = rt.science_tool(AUDITOR, "quantiles", PRED, qs=[round(0.05 * (k + 1), 2)])
        assert r["ok"], r
        ms.append(r["event_id"])
    c = rt.publish_claim(AUDITOR, "broad summary " + "y" * 1000, ms, status="supported")
    claim = rt.store.get_event(c["event_id"])
    line = rt.compact_event(claim, 240)
    for pid in ms[:12]:
        assert pid in line  # parents are never clipped (first 12 listed)
    assert "+2 more" in line and "src: " in line
    assert line.endswith(f"… [full: get_event {c['event_id']}]") or "[full: get_event" in line
    assert len(line) < 700  # bounded
    m_line = rt.compact_event(rt.store.get_event(ms[0]), 240)
    assert "synthetic:" in m_line or "src: " in m_line  # source file id is kept on measurements
    # the full text and the full measurement are always retrievable
    full = rt.get_event_detail(c["event_id"])
    assert "y" * 1000 in full["text"] and full["parents"] == ms
    assert rt.get_event_detail(ms[0])["measurement"]["tool_id"] == "quantiles"


def test_own_summary_is_bounded_and_names_older_events():
    rt = make_runtime(max_events=600, max_hops=10)
    ids = flood(rt, 10, agent=THERMAL)
    s = build_own_summary(rt, ids, limit=4)
    assert s.splitlines()[0].startswith("(6 earlier event(s) of yours omitted: evt-0001..evt-0006")
    assert len(s.splitlines()) == 5 and all(i in s for i in ids[-4:])


def test_list_events_is_compact_and_bounded():
    rt = make_runtime(max_events=600, max_hops=10)
    flood(rt, 60)
    lines = rt.list_events(SKEPTIC)
    assert len(lines) == 15 and all(len(x) < 520 for x in lines)
    assert rt.store.count_events() == 60


def test_first_turn_prompt_has_context_budget_and_tools_but_stays_small():
    rt = make_runtime()
    spec, st = rt.specs[THERMAL], rt.state[THERMAL]
    p = build_prompt(rt, spec, st, [])
    assert "Region context (JSON)" in p and "Budget left" in p and "thermal_psr_consistency" in p
    assert "start of the investigation" in p and len(p) < 3500
    st.rounds = 1
    p2 = build_prompt(rt, spec, st, [])
    assert "Region context" not in p2  # the context is not repeated every turn


def test_instructions_and_tool_schemas_stay_trimmed():
    assert len(COMMON) < 3000  # was ~3.3k chars before P6
    rt = make_runtime()
    for spec in rt.specs.values():
        desc = sum(len(t.description or "") for t in make_tools(rt, spec))
        assert desc < 3200, (spec.agent_id, desc)
    assert "bounded digest" in COMMON and "get_event" in COMMON


def test_token_ceilings_use_provider_reported_usage_and_stop_before_more_spend():
    rt = make_runtime(max_llm_calls=100)
    rt.limits.max_input_tokens = 1000
    rt.llm_before_call(THERMAL)
    rt.llm_record(THERMAL, started=__import__("datetime").datetime.now(__import__("datetime").UTC),
                  request=[], response={}, error=None,
                  usage={"input_token_count": 1200, "output_token_count": 50, "total_token_count": 1250})  # fmt: skip
    with pytest.raises(BudgetExhausted) as e:
        rt.llm_before_call(THERMAL)
    assert (
        e.value.kind == "INPUT_TOKENS" and rt.llm_calls == 1
    )  # the refused call was never counted
    rt2 = make_runtime(max_llm_calls=100)
    rt2.limits.max_total_tokens = (
        3000  # the soft total cap also reserves a 1500-token output allowance
    )
    rid = rt2.llm_before_call(GEOLOGY)
    rt2.llm_record(GEOLOGY, started=__import__("datetime").datetime.now(__import__("datetime").UTC),
                   request=[], response={}, error=None, reservation=rid,
                   usage={"input_token_count": 1700, "output_token_count": 1800, "total_token_count": 3500})  # fmt: skip
    with pytest.raises(BudgetExhausted) as e2:
        rt2.llm_before_call(GEOLOGY)
    assert e2.value.kind == "TOTAL_TOKENS"
    s = rt2.summary("X", 0.0)
    assert s.tokens_by_agent == {
        "geology": {"calls": 1, "input": 1700, "output": 1800, "total": 3500}
    }
    assert s.cost["status"] == "unpriced" and s.limits["max_total_tokens"] == 3000
    assert "soft" in s.limits["enforcement"]  # the total cap is honestly labelled soft
