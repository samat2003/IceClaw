"""Token ceilings are enforced with PRE-CALL RESERVATIONS (P6 corrections).

Background (measured on the recorded P6 live runs, see docs/INVESTIGATION_BUNDLE.md §7):
* runs A/B overshot their 70k cap although NOTHING was in flight: the last call started below the
  cap and simply added its own input (check used completed usage only, ignoring the call's size);
* run C overshot 8 316 tokens because two calls started while another call was in flight.
Both are reproduced here as arithmetic and shown fixed. Arrays are SYNTHETIC; no model is called.
"""

import asyncio
from datetime import UTC, datetime

import pytest
from agent_helpers import (
    AUDITOR,
    GEOLOGY,
    PRED,
    SKEPTIC,
    THERMAL,
    call,
    make_runtime,
    quiet,
    run_with,
)

from iceclaw.agents.provider import ScriptedChatClient, measure_request_chars
from iceclaw.agents.specialists import build_collective, build_prompt
from iceclaw.swarm.runtime import (
    INITIAL_TOKENS_PER_CHAR,
    OUTPUT_ALLOWANCE_TOKENS,
    BudgetExhausted,
)


def usage(i: int, o: int = 100) -> dict:
    return {"input_token_count": i, "output_token_count": o, "total_token_count": i + o}


def finish(rt, agent, rid, measure, input_tokens, output_tokens=100):
    rt.llm_record(agent, started=datetime.now(UTC), request=[], response={}, error=None,
                  usage=usage(input_tokens, output_tokens), reservation=rid, measure_chars=measure)  # fmt: skip


# ----------------------------------------------------------- the two observed failure modes
def test_legacy_check_overshoots_both_by_own_size_and_by_in_flight_calls():
    """No size information (est 0) reproduces the old behaviour: cap 10 000 is exceeded."""
    rt = make_runtime(max_llm_calls=100)
    rt.limits.max_input_tokens = 10_000
    # (a) sequential, run A/B style: 6 000 done, next call is 6 000 -> old rule admits it (6 000 < cap)
    r1 = rt.llm_before_call(THERMAL)
    finish(rt, THERMAL, r1, 0, 6_000)
    r2 = rt.llm_before_call(THERMAL)  # admitted although its own 6 000 will exceed the cap
    finish(rt, THERMAL, r2, 0, 6_000)
    assert rt.tokens["input"] == 12_000 > 10_000
    # (b) concurrent, run C style: 3 calls admitted before any completes
    rt2 = make_runtime(max_llm_calls=100)
    rt2.limits.max_input_tokens = 10_000
    rids = [rt2.llm_before_call(a) for a in (THERMAL, GEOLOGY, AUDITOR)]
    for a, rid in zip((THERMAL, GEOLOGY, AUDITOR), rids, strict=True):
        finish(rt2, a, rid, 0, 4_000)
    assert rt2.tokens["input"] == 12_000 > 10_000


def test_reservation_refuses_a_call_whose_own_size_would_cross_the_cap():
    rt = make_runtime(max_llm_calls=100)
    rt.limits.max_input_tokens = 10_000
    measure = 12_000  # est = 12 000 * 0.5 = 6 000 tokens before calibration
    r1 = rt.llm_before_call(THERMAL, measure)
    assert rt.reserved_input_tokens == 6_000
    finish(rt, THERMAL, r1, measure, 5_000)  # actual 5 000 (calibrates ratio 0.4167 -> x1.25)
    assert rt.reserved_input_tokens == 0 and rt.tokens["input"] == 5_000
    # est now ~ 12 000 * 0.5208 = 6 250: 5 000 + 6 250 > 10 000 -> refused (old rule admitted it)
    with pytest.raises(BudgetExhausted) as e:
        rt.llm_before_call(THERMAL, measure)
    assert e.value.kind == "INPUT_TOKENS" and rt.llm_calls == 1  # the refused call is not counted


def test_in_flight_reservations_block_concurrent_overshoot_and_release_frees_room():
    rt = make_runtime(max_llm_calls=100)
    rt.limits.max_input_tokens = 10_000
    m = 8_000  # est 4 000 each before calibration
    a = rt.llm_before_call(THERMAL, m)
    b = rt.llm_before_call(GEOLOGY, m)
    assert rt.reserved_input_tokens == 8_000 and rt.peak_inflight == 2
    with pytest.raises(BudgetExhausted):  # 8 000 reserved + 4 000 > 10 000
        rt.llm_before_call(AUDITOR, m)
    finish(
        rt, THERMAL, a, m, 3_000
    )  # calibrates: 3000/8000 = 0.375 -> x1.25 = 0.46875 -> est 3 750
    with pytest.raises(BudgetExhausted):  # 3 000 used + 4 000 reserved (b) + 3 750 > 10 000
        rt.llm_before_call(AUDITOR, m)
    finish(rt, GEOLOGY, b, m, 3_000)
    c = rt.llm_before_call(AUDITOR, m)  # 6 000 used + 0 reserved + 3 750 <= 10 000
    assert rt.reserved_input_tokens == 3_750
    finish(rt, AUDITOR, c, m, 3_000)
    assert rt.tokens["input"] == 9_000 <= 10_000
    assert rt.reserved_input_tokens == 0


def test_async_concurrent_callers_never_exceed_the_cap_when_estimates_hold():
    async def go():
        rt = make_runtime(max_llm_calls=1000)
        cap = 20_000
        rt.limits.max_input_tokens = cap
        m = 6_000  # unit actual usage 2 000 (ratio 0.333): est stays above actual

        async def caller(agent, n):
            for _ in range(n):
                try:
                    rid = rt.llm_before_call(agent, m)
                except BudgetExhausted:
                    return
                await asyncio.sleep(0.001)  # the call is genuinely in flight here
                finish(rt, agent, rid, m, 2_000)

        await asyncio.gather(*(caller(a, 50) for a in (THERMAL, GEOLOGY, AUDITOR, SKEPTIC)))
        return rt, cap

    rt, cap = asyncio.run(go())
    assert rt.tokens["input"] <= cap  # reservations made the cap hold under concurrency
    assert rt.peak_inflight >= 2  # ...and calls really were concurrent
    assert rt.reserved_input_tokens == 0 and rt.llm_calls >= 5


def test_estimator_is_conservative_first_then_calibrated_from_provider_usage():
    rt = make_runtime()
    assert rt.estimate_input_tokens(10_000) == int(10_000 * INITIAL_TOKENS_PER_CHAR)
    rid = rt.llm_before_call(THERMAL, 10_000)
    finish(rt, THERMAL, rid, 10_000, 2_500)  # observed 0.25 tokens/char
    assert rt.estimate_input_tokens(10_000) == 3_125  # 0.25 * 1.25 safety margin
    rid = rt.llm_before_call(THERMAL, 10_000)
    finish(rt, THERMAL, rid, 10_000, 4_000)  # a denser request: the MAX ratio is used from now on
    assert rt.estimate_input_tokens(10_000) == 5_000


def test_total_cap_is_labelled_soft_and_reserves_an_output_allowance():
    rt = make_runtime(max_llm_calls=100)
    rt.limits.max_total_tokens = 2 * OUTPUT_ALLOWANCE_TOKENS
    rt.llm_before_call(THERMAL, 1_000)  # est 500 + allowance 1500 fits
    with pytest.raises(BudgetExhausted) as e:
        rt.llm_before_call(GEOLOGY, 1_000)  # one more in flight: 500+500+2*1500 > 3000
    assert e.value.kind == "TOTAL_TOKENS"
    assert "soft" in rt.summary("X", 0.0).limits["enforcement"]


def test_reported_overshoot_is_measured_when_an_estimate_was_too_low():
    rt = make_runtime(max_llm_calls=100)
    rt.limits.max_input_tokens = 1_000
    rid = rt.llm_before_call(THERMAL, 100)  # est 50 -> admitted
    finish(rt, THERMAL, rid, 100, 1_700)  # the provider reports far more than estimated
    s = rt.summary("X", 0.0)
    assert s.limits["input_cap_overshoot_tokens"] == 700  # honest: measured, not hidden
    assert s.limits["peak_inflight_calls"] == 1
    rt.limits.max_input_tokens = None
    assert rt.summary("X", 0.0).limits["input_cap_overshoot_tokens"] is None


# -------------------------------------------------------------- measurement and the real client path
def test_request_measure_includes_instructions_and_tool_schemas():
    from iceclaw.agents.specialists import build_specs
    from iceclaw.agents.tools import make_tools

    rt = make_runtime()
    spec = build_specs()[0]
    tools = make_tools(rt, spec)
    msgs = [{"role": "user", "contents": [{"type": "text", "text": "hello"}]}]
    base = measure_request_chars(msgs, {})
    full = measure_request_chars(
        msgs, {"instructions": spec.instructions, "tools": tools, "store": False}
    )
    assert full > base + len(spec.instructions)  # instructions AND tool schemas are counted
    assert full - base - len(spec.instructions) > 3_000  # eight tool schemas are not negligible
    assert measure_request_chars(msgs, None) == base


def _huge_policy(n_calls):
    def policy(ctx):
        n = len(ctx.last_tool_results)
        if n < n_calls:
            return call("run_science_tool", tool_id="valid_pixel_summary", modality=PRED,
                        window=[0, 4 + n, 0, 4 + n])  # fmt: skip
        return "done"

    return policy


def test_capped_run_end_to_end_stays_under_the_cap_with_concurrent_agents():
    pol = {a: _huge_policy(6) for a in (THERMAL, GEOLOGY, AUDITOR)}
    pol[SKEPTIC] = quiet
    pol[THERMAL] = _huge_policy(6)
    # uncapped baseline
    base_rt = make_runtime(max_llm_calls=100, max_tool_calls=100, max_events=200)
    base, _ = asyncio.run(run_with(base_rt, {**pol}, latency_s=0.005))
    assert base.input_tokens > 0 and base.limits["input_cap_overshoot_tokens"] is None
    # same run with a cap well below the uncapped usage
    cap = int(base.input_tokens * 0.6)
    rt = make_runtime(max_llm_calls=100, max_tool_calls=100, max_events=200)
    rt.limits.max_input_tokens = cap
    summary, clients = asyncio.run(run_with(rt, {**pol}, latency_s=0.005))
    assert summary.input_tokens <= cap  # the cap held
    assert summary.limits["input_cap_overshoot_tokens"] == 0
    assert summary.llm_calls < base.llm_calls
    assert summary.ended_reason == "BUDGET_EXHAUSTED"
    assert any(str(r).startswith("BUDGET_INPUT_TOKENS") for r in summary.exit_reasons.values())
    assert summary.limits["peak_inflight_calls"] >= 2  # the agents really ran concurrently
    assert rt.reserved_input_tokens == 0  # no reservation leaked
    assert summary.problems == []


def test_reservation_is_released_on_provider_errors_and_on_cancellation():
    # provider error inside the call
    rt = make_runtime(max_llm_calls=100)

    def boom(ctx):
        raise RuntimeError("provider outage")

    asyncio.run(run_with(rt, {a: boom for a in (THERMAL, GEOLOGY, AUDITOR, SKEPTIC)}))
    assert rt.reserved_input_tokens == 0 and rt._inflight == {}
    assert all(r["ok"] == 0 for r in rt.store.llm_calls()) and rt.store.llm_calls()
    # wall-clock timeout cancels calls that are in flight
    rt2 = make_runtime(timeout_s=0.3)

    class Slow(ScriptedChatClient):
        def _inner_get_response(self, *, messages, stream, options, **kw):
            async def slow():
                await asyncio.sleep(5)

            return slow()

    def factory(runtime, aid):
        c = Slow(quiet, aid)
        c.bind(runtime, aid)
        return c

    async def go():
        return await rt2.run(build_collective(rt2, factory), build_prompt)

    s = asyncio.run(go())
    assert s.ended_reason == "TIMEOUT"
    assert rt2._inflight == {}  # cancelled in-flight calls did not leak reservations


def test_bundle_manifest_records_the_enforcement_label_and_peak_concurrency(tmp_path):
    from bundle_helpers import scripted_service

    res = scripted_service(tmp_path / "runs", tmp_path / "src").investigate(
        "synthetic-region", run_id="inv-policy"
    )
    import json

    m = json.loads((res.run_dir / "input_manifest.json").read_text())
    pp = m["prompt_policy"]
    assert "pre-call reservation" in pp["enforcement"] and "soft" in pp["enforcement"]
    assert pp["peak_inflight_calls"] >= 1 and pp["input_cap_overshoot_tokens"] == 0
    assert pp["max_input_tokens"] == 150_000
