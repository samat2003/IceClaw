"""Shared harness for offline agent tests: SYNTHETIC arrays + scripted (fake) model policies.

Nothing here is lunar science. The fake model sees only the prompt/digest text and tool results
(exactly what a real model would see) and answers with tool calls produced by small policies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from iceclaw.agents.provider import Policy, ScriptContext, ScriptedChatClient
from iceclaw.agents.specialists import (
    AUDITOR,
    GEOLOGY,
    PRED,
    SHADOW,
    SKEPTIC,
    TEMP,
    TERRAIN,
    THERMAL,
    build_collective,
    build_prompt,
    build_specs,
)
from iceclaw.contracts import AgentRegionContext, RunBudget
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.science.common import RasterLayer, RegionArrays
from iceclaw.swarm.runtime import RunLimits, RunSummary, SwarmRuntime
from iceclaw.swarm.store import EvidenceStore

PROJ = {
    "kind": "polar-stereographic",
    "latitudeOfOriginDeg": -90,
    "centralMeridianDeg": 0,
    "scaleFactor": 0.994,
    "falseEastingM": 500000.0,
    "falseNorthingM": 500000.0,
    "radiusM": 1737400.0,
}
GRID = {"xMinM": 320000.0, "yMaxM": 740000.0, "pixelSizeXM": 240.0, "pixelSizeYM": 240.0}
SHA = {PRED: "a" * 64, TEMP: "b" * 64, SHADOW: "c" * 64, TERRAIN: "d" * 64}


def synthetic_arrays(n: int = 32) -> RegionArrays:
    """SYNTHETIC 4-layer region (smooth fields + one shadowed corner)."""
    rng = np.random.default_rng(3)
    rr, cc = np.mgrid[0:n, 0:n]
    pred = 0.5 + 0.3 * np.sin(rr / 5.0) * np.cos(cc / 7.0) + rng.normal(0, 0.01, (n, n))
    shadow = np.zeros((n, n))
    shadow[: n // 4, : n // 4] = 1.0
    temp = np.where(shadow >= 0.5, 90.0, 250.0) + rng.normal(0, 2.0, (n, n))
    slope = 8.0 + 6.0 * np.abs(np.sin(rr / 6.0)) + rng.normal(0, 0.5, (n, n))
    cov = np.ones((n, n), bool)

    def layer(name, v, unit):
        return RasterLayer(name, v, cov, unit, f"synthetic:{name}", SHA[name])

    corners = ((0, 0), (n, 0), (n, n), (0, n))
    fp = tuple(ma.pixel_to_lat_lon(PROJ, GRID, c, r) for c, r in corners)
    return RegionArrays(
        region_id="synthetic-region",
        footprint=fp,
        resolution_m=240.0,
        projection=PROJ,
        grid={**GRID, "widthPx": n, "heightPx": n},
        layers={
            PRED: layer(PRED, pred, "prospectivity_index"),
            TEMP: layer(TEMP, temp, "K"),
            SHADOW: layer(SHADOW, shadow, "fraction"),
            TERRAIN: layer(TERRAIN, slope, "deg"),
        },
    )


def context_for(arrays: RegionArrays, budget: RunBudget) -> AgentRegionContext:
    from iceclaw.data.catalog import available_tool_names

    n = arrays.layers[PRED].values.shape[0]
    return AgentRegionContext(
        region_id=arrays.region_id,
        pole="south",
        footprint=list(arrays.footprint),
        width=n,
        height=n,
        spatial_resolution_m=arrays.resolution_m,
        lunar_crs="+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=500000 +y_0=500000 +R=1737400",
        source_ids=["synthetic-source"],
        source_mode="upstream_published_example",  # enum value only; data are SYNTHETIC
        available_modalities=sorted(arrays.layers),
        available_tools=available_tool_names(arrays),
        budget=budget,
    )


def budget(**kw: Any) -> RunBudget:
    base = dict(max_events=60, max_hops=6, max_tool_calls=20, max_llm_calls=40, timeout_s=30)
    base.update(kw)
    return RunBudget(**base)


def make_runtime(
    arrays: RegionArrays | None = None,
    store: EvidenceStore | None = None,
    specs=None,
    max_rounds: int = 4,
    **bud: Any,
) -> SwarmRuntime:
    arrays = arrays or synthetic_arrays()
    b = budget(**bud)
    return SwarmRuntime(
        investigation_id="inv-test",
        context=context_for(arrays, b),
        arrays=arrays,
        store=store or EvidenceStore(),
        specs=specs or build_specs(),
        limits=RunLimits(budget=b, max_rounds_per_agent=max_rounds),
        provider="scripted",
        model="scripted-v0",
    )


async def run_with(
    rt: SwarmRuntime, policies: dict[str, Policy], latency_s: float = 0.0, **fic: Any
):
    """Run the collective with scripted models. Returns (summary, clients)."""
    clients: dict[str, ScriptedChatClient] = {}

    def factory(runtime: SwarmRuntime, aid: str):
        c = ScriptedChatClient(policies[aid], aid, **fic)
        c.latency_s = latency_s
        c.bind(runtime, aid)
        clients[aid] = c
        return c

    runners = build_collective(rt, factory)
    summary: RunSummary = await rt.run(runners, build_prompt)
    return summary, clients


# ------------------------------------------------------------------ policy building blocks
EVT = re.compile(r"\[(evt-\d+)\] (\w+) (\w+)")


def parse(result: str) -> dict:
    return json.loads(result)


def digest_events(ctx: ScriptContext) -> list[tuple[str, str, str]]:
    """(event_id, author, kind) for events described in the latest prompt."""
    return EVT.findall(ctx.last_user_text)


def call(name: str, **args: Any) -> list[tuple[str, dict[str, Any]]]:
    return [(name, args)]


@dataclass
class Script:
    """Tiny helper: a policy that walks a list of steps, one model round-trip per step."""

    steps: list

    def __call__(self, ctx: ScriptContext):
        i = ctx.roundtrip
        if i >= len(self.steps):
            return "done"
        s = self.steps[i]
        return s(ctx) if callable(s) else s


def first_event_id(ctx: ScriptContext) -> str:
    return parse(ctx.last_tool_results[-1])["event_id"]


def quiet(ctx: ScriptContext) -> str:
    return "nothing warranted"


AGENT_IDS = (THERMAL, GEOLOGY, AUDITOR, SKEPTIC)
