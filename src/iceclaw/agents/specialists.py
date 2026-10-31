"""The four specialists: roles, tool/layer permissions, prompts, and Agent construction.

Role text states the scientific rules and each agent's remit but NEVER prescribes findings,
numbers, a debate, or an order of work. Whether anything is challenged or revised is decided by
the model from published evidence; "no warranted contradiction" is an explicitly allowed outcome.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agent_framework import Agent

from iceclaw.agents.tools import make_tools
from iceclaw.contracts import EvidenceEvent
from iceclaw.swarm.runtime import AgentSpec, AgentState, SwarmRuntime, TurnRunner

THERMAL, GEOLOGY, AUDITOR, SKEPTIC = "thermal", "geology", "prediction_auditor", "skeptic"
PRED, TEMP, SHADOW, TERRAIN = (
    "nasa_ibm_prediction_raw",
    "pixelwise_temperature",
    "pixelwise_shadow_mask",
    "pixelwise_terrain",
)
GENERIC = (
    "valid_pixel_summary",
    "quantiles",
    "missingness",
    "gradient_stats",
    "discontinuity_candidates",
)

COMMON = """\
You are one of four independent AI specialists auditing a lunar ice-PROSPECTIVITY prediction for ONE
verified region. You act only through your tools. No supervisor tells you what to find or in what
order: the other specialists run concurrently and may publish evidence that you can build on,
check, request tests of, or challenge.

Scientific ground rules
- The NASA-IBM output is a prospectivity INDEX from a regression model (raw values can fall outside
  [0,1]). It is not a measured ice abundance, not a calibrated probability, and not a detection.
  Never state or imply that ice was found or confirmed.
- Numbers come ONLY from tool results. Every quantitative statement must cite the event ids of the
  measurements it rests on. Never invent, estimate, or recompute numbers yourself.
- Layers and units: nasa_ibm_prediction_raw [prospectivity_index], pixelwise_temperature [K],
  pixelwise_shadow_mask [fraction; permanently shadowed where >= 0.5], pixelwise_terrain [deg].
  No other layers exist. Tile-level scalars and single-point anchors are not arrays.
- Thresholds you pass to tools (cold-trap K, steep-slope degrees, ...) are YOUR parameters, not
  validated physical constants: state them and their limits when you rely on them.
- Keep apart: measurement (tool result), claim (your interpretation), hypothesis (untested idea).
  State alternative explanations and limitations honestly.
- Disagreement is NOT required. If the evidence is consistent, say so. Never manufacture a
  challenge, a conflict, or a revision; a revision needs NEW evidence published after the claim.
- Budgets are small and shared (shown each turn). Plan a few high-value calls; identical repeated
  calls return the cached event and cost nothing.

How to work
- Decide for yourself what to measure within your remit. Publish a claim that cites the measurement
  event ids, then conclude. When you are woken by others' events, read them and decide
  independently whether anything is warranted: another test, request_test to the right
  specialist, a challenge, a revision of your own claim backed by new evidence, or nothing.
- End each turn with one or two plain sentences saying what you did (no new numbers).
"""

ROLES: dict[str, tuple[str, str]] = {
    THERMAL: (
        "Thermal Scientist",
        "Your remit: temperature and permanently-shadowed-region (PSR) behaviour using "
        "pixelwise_temperature [K] and pixelwise_shadow_mask. Typical questions: how do PSR pixels "
        "compare with non-PSR pixels, is the temperature field internally plausible, and how large "
        "is the subset below a threshold YOU choose and justify as a screening parameter.",
    ),
    GEOLOGY: (
        "Geology Scientist",
        "Your remit: terrain using pixelwise_terrain [deg]. Typical questions: slope distribution, "
        "the fraction above a steepness threshold YOU choose, spatial slope discontinuities, and "
        "geological counter-explanations that slope can or cannot speak to. You cannot measure "
        "composition; say so rather than speculating beyond slope.",
    ),
    AUDITOR: (
        "Prediction Auditor",
        "Your remit: the raw NASA-IBM prediction raster (nasa_ibm_prediction_raw). Typical "
        "questions: value range and excursions outside [0,1], missing data, gradients and "
        "discontinuities, tile-edge behaviour, peak location. Decide which prediction artifacts "
        "look questionable and which look ordinary; you may ask thermal or geology for a test.",
    ),
    SKEPTIC: (
        "Skeptic / Investigation Planner",
        "You own no data. You read what others publish, may cross-check any layer, "
        "re-run any stored "
        "measurement with recompute_measurement, request tests from the specialists, challenge a "
        "claim ONLY when evidence supports doing so (cite it), and propose what to "
        "investigate next. "
        "'No warranted contradiction' is a valid and honest result. You act when others publish.",
    ),
}

ALLOWED_TOOLS = {
    THERMAL: frozenset({"thermal_psr_consistency", *GENERIC}),
    GEOLOGY: frozenset({"terrain_slope_summary", *GENERIC}),
    AUDITOR: frozenset({*GENERIC, "unit_interval_check", "boundary_check", "peak_location"}),
    SKEPTIC: frozenset(
        {
            *GENERIC,
            "unit_interval_check",
            "boundary_check",
            "peak_location",
            "thermal_psr_consistency",
            "terrain_slope_summary",
        }
    ),
}
ALLOWED_MODALITIES = {
    THERMAL: frozenset({TEMP, SHADOW}),
    GEOLOGY: frozenset({TERRAIN}),
    AUDITOR: frozenset({PRED}),
    SKEPTIC: frozenset({PRED, TEMP, SHADOW, TERRAIN}),
}


def build_specs() -> list[AgentSpec]:
    specs = []
    for aid, (title, remit) in ROLES.items():
        specs.append(
            AgentSpec(
                agent_id=aid,
                title=title,
                instructions=f"{COMMON}\nYou are the {title}.\n{remit}\n",
                allowed_tools=ALLOWED_TOOLS[aid],
                allowed_modalities=ALLOWED_MODALITIES[aid],
                can_recompute=aid == SKEPTIC,
                starts_idle=aid == SKEPTIC,  # nothing to review until others publish
                wake_on_claims=aid == SKEPTIC,
            )
        )
    return specs


# --------------------------------------------------------------------------------- prompts
def _budget_line(rt: SwarmRuntime) -> str:
    b = rt.limits.budget
    return (
        f"Budget remaining (shared): tool calls "
        f"{max(0, b.max_tool_calls - rt.tool_calls)}/{b.max_tool_calls}, "
        f"model calls {max(0, b.max_llm_calls - rt.llm_calls)}/{b.max_llm_calls}, "
        f"events {max(0, b.max_events - rt.store.count_events())}/{b.max_events}, "
        f"max reaction depth {b.max_hops}. "
        f"Your rounds: {{rounds}}/{rt.limits.max_rounds_per_agent}."
    )


def build_prompt(
    rt: SwarmRuntime, spec: AgentSpec, st: AgentState, new: list[EvidenceEvent]
) -> str:
    """Per-turn user message: context, budget, own history, newly delivered events."""
    parts = [_budget_line(rt).format(rounds=st.rounds + 1)]
    if st.rounds == 0:
        parts.append("Region context (JSON): " + rt.context.model_dump_json())
        parts.append(
            "Your tools: "
            + ", ".join(sorted(spec.allowed_tools))
            + "; layers: "
            + ", ".join(sorted(spec.allowed_modalities))
        )
    if st.own_event_ids:
        own = [rt.describe_event(e) for i in st.own_event_ids[-8:] if (e := rt.store.get_event(i))]
        parts.append("Your recent events:\n" + "\n".join(own))
    if new:
        lines = [rt.describe_event(e) for e in new[-25:]]
        parts.append("Events published by others since your last turn:\n" + "\n".join(lines))
    if st.rounds == 0 and not new:
        parts.append(
            "This is the start of the investigation. Decide what within your remit is worth "
            "measuring "
            "with your limited budget, measure it, publish a claim citing your evidence, "
            "then conclude."
        )
    else:
        parts.append(
            "Decide whether anything is warranted now (a test, a request, a challenge, a "
            "revision backed by "
            "NEW evidence, or a conclusion). It is fine to do nothing."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------- MAF agents
class MafTurnRunner:
    """One stateless Agent turn per call (history lives in the evidence log, not in a session)."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    async def take_turn(self, prompt: str) -> str:
        response = await self.agent.run(prompt)
        return response.text or ""


ClientFactory = Callable[[SwarmRuntime, str], Any]


def build_collective(runtime: SwarmRuntime, client_factory: ClientFactory) -> dict[str, TurnRunner]:
    """One real MAF ``Agent`` per specialist, each with its own client and approved tools."""
    runners: dict[str, TurnRunner] = {}
    for spec in runtime.specs.values():
        agent = Agent(
            client_factory(runtime, spec.agent_id),
            instructions=spec.instructions,
            name=spec.agent_id,
            description=spec.title,
            tools=make_tools(runtime, spec),
            default_options={"store": False},  # no server-side conversation storage
        )
        runners[spec.agent_id] = MafTurnRunner(agent)
    return runners


def ensure_json(x: Any) -> str:  # tiny helper for scripts/tests
    return json.dumps(x, indent=2, default=str)
