"""One limited-budget LIVE four-specialist investigation of the verified paired region.

Spends real OpenAI API budget, so it refuses to call the model unless ``--live`` is given.
Without ``--live`` it only validates configuration and the region (no model call, no spend).

Settings come from the IceClaw runtime (``ICECLAW_LLM_PROVIDER``, ``ICECLAW_LLM_MODEL``,
``OPENAI_API_KEY``; local ``.env`` is loaded by IceClaw and never printed). The model ID is never
guessed. Only the verified PAIRED region ``ice-s-0001-0002`` is accepted.

    python scripts/p5_live_investigation.py            # dry run: config + region check only
    python scripts/p5_live_investigation.py --live     # real model calls within the budget
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from iceclaw.agents.provider import build_openai_client
from iceclaw.agents.specialists import build_collective, build_prompt, build_specs
from iceclaw.config import ConfigError, load_dotenv_file, load_settings
from iceclaw.contracts import RunBudget
from iceclaw.swarm.runtime import RunLimits, SwarmRuntime, load_paired_region
from iceclaw.swarm.store import EvidenceStore

REGION = "ice-s-0001-0002"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="actually call the model (spends budget)")
    ap.add_argument("--region", default=REGION)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--max-hops", type=int, default=5)
    ap.add_argument("--max-tool-calls", type=int, default=16)
    ap.add_argument("--max-llm-calls", type=int, default=28)
    ap.add_argument("--timeout-s", type=float, default=420.0)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (yours)")
    ap.add_argument(
        "--price-out", type=float, default=None, help="USD per 1M output tokens (yours)"
    )
    args = ap.parse_args(argv)

    load_dotenv_file()  # IceClaw's own loader; values are never printed
    settings = load_settings()
    print("settings:", json.dumps(settings.redacted()))
    budget = RunBudget(
        max_events=args.max_events, max_hops=args.max_hops, max_tool_calls=args.max_tool_calls,
        max_llm_calls=args.max_llm_calls, timeout_s=args.timeout_s,
    )  # fmt: skip
    try:
        ctx, arrays = load_paired_region(settings, args.region, budget)
    except Exception as e:
        print(f"REGION NOT INVESTIGABLE: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"region {ctx.region_id}: layers {sorted(arrays.layers)}; tools {ctx.available_tools}")
    try:
        settings.require_live_llm()
    except ConfigError as e:
        print(f"LIVE MODE UNAVAILABLE: {e}", file=sys.stderr)
        return 2
    if not args.live:
        print("dry run OK (configuration valid). Re-run with --live to spend API budget.")
        return 0

    inv = f"inv-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-p5live"
    out = args.out_dir or (settings.runs_root / inv)
    out.mkdir(parents=True, exist_ok=True)
    store = EvidenceStore(out / "events.sqlite")
    limits = RunLimits(budget=budget, max_rounds_per_agent=args.max_rounds)
    rt = SwarmRuntime(
        investigation_id=inv, context=ctx, arrays=arrays, store=store, specs=build_specs(),
        limits=limits, provider=settings.llm_provider, model=settings.llm_model or "",
    )  # fmt: skip
    runners = build_collective(
        rt, lambda runtime, aid: build_openai_client(settings, runtime, aid, limits)
    )
    summary = asyncio.run(rt.run(runners, build_prompt))
    rt.write_outputs(out, summary)

    cost = None
    if args.price_in is not None and args.price_out is not None:
        cost = round(
            summary.input_tokens / 1e6 * args.price_in
            + summary.output_tokens / 1e6 * args.price_out,
            4,
        )
    note = (
        "from user-supplied prices" if cost is not None else "not computed: no price list supplied"
    )
    report = {
        **summary.__dict__,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "estimated_cost_usd": cost,
        "cost_note": note,
    }
    (out / "run_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"outputs: {out}")
    return 0 if not summary.problems else 3


if __name__ == "__main__":
    sys.exit(main())
