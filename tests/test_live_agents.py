"""LIVE integration test (real OpenAI API). Skipped unless explicitly enabled:

    ICECLAW_RUN_LIVE_TESTS=1 python -m pytest tests/test_live_agents.py -q

It spends a tiny, hard-bounded amount of API budget (<= 4 model calls, <= 2 tool calls) on ONE
agent (the Prediction Auditor) over the real paired region to confirm the complete runtime:
settings -> real MAF Agent + OpenAI client -> real function-tool call -> evidence event ->
stored model call with token usage. A mock never proves this; the fast suite never runs it.
"""

import asyncio
import os
from pathlib import Path

import pytest

from iceclaw.agents.provider import build_openai_client
from iceclaw.agents.specialists import AUDITOR, build_collective, build_prompt, build_specs
from iceclaw.config import load_dotenv_file, load_settings
from iceclaw.contracts import EventKind, RunBudget
from iceclaw.swarm.runtime import RunLimits, SwarmRuntime, load_paired_region
from iceclaw.swarm.store import EvidenceStore

pytestmark = pytest.mark.live
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("ICECLAW_RUN_LIVE_TESTS") != "1", reason="live tests are opt-in")
def test_live_single_agent_roundtrip_on_real_paired_region(monkeypatch):
    # the autouse fixture isolates tests from .env; a live run deliberately opts back in
    monkeypatch.delenv("ICECLAW_SKIP_DOTENV", raising=False)
    monkeypatch.delenv("ICECLAW_SOMBENCH_ROOT", raising=False)
    monkeypatch.chdir(REPO)
    load_dotenv_file()
    settings = load_settings()
    settings.require_live_llm()  # clear MISSING_API_KEY / MISSING_LLM_MODEL if not configured

    budget = RunBudget(max_events=12, max_hops=3, max_tool_calls=2, max_llm_calls=4, timeout_s=120)
    ctx, arrays = load_paired_region(settings, "ice-s-0001-0002", budget)
    spec = next(s for s in build_specs() if s.agent_id == AUDITOR)
    limits = RunLimits(budget=budget, max_rounds_per_agent=1)
    rt = SwarmRuntime(
        investigation_id="inv-live-test", context=ctx, arrays=arrays, store=EvidenceStore(),
        specs=[spec], limits=limits, provider=settings.llm_provider, model=settings.llm_model,
    )  # fmt: skip
    runners = build_collective(rt, lambda r, aid: build_openai_client(settings, r, aid, limits))
    summary = asyncio.run(rt.run(runners, build_prompt))

    rows = rt.store.llm_calls()
    assert 1 <= len(rows) <= 4 and summary.llm_calls == len(rows)
    assert all(
        r["provider"] == settings.llm_provider and r["model"] == settings.llm_model for r in rows
    )
    assert any(r["ok"] == 1 for r in rows), [r["error"] for r in rows]
    assert summary.total_tokens > 0 and summary.problems == []
    events = list(rt.store)
    assert not [e for e in events if e.error_code == "AGENT_TURN_ERROR"], [
        e.claim_or_request for e in events
    ]
    # the real model chose to call a real tool -> authentic measurement tied to the source file
    meas = [e for e in events if e.event_kind is EventKind.MEASUREMENT and e.agent_id == AUDITOR]
    assert meas and meas[0].tool_run_id and meas[0].source_artifact_ids
    assert rt.tool_calls <= 2 and rt.store.get_tool_run(meas[0].tool_run_id) is not None
