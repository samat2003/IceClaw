"""P7 LLM arms: one generalist, four isolated specialists, four communicating specialists.

All three arms use the same task text, the same tool implementations, the same machine-readable
assessment tool and (set by the benchmark runner) the same TOTAL model-call, token, tool-call,
event and wall-time budget. They differ only in team structure:

* ``single_generalist``: one Microsoft Agent Framework agent with every layer and tool.
* ``isolated_specialists``: the four P5 roles with their P5 permissions, working concurrently but
  unable to read, cite, request from or challenge each other (``communication=False``).
* ``communicating_specialists``: the same four roles with the P5 event bus (claims wake the
  Skeptic, test requests / challenges / revisions allowed).

Role text never prescribes findings or the presence of a fault; "no fault found" is a valid verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_framework import Agent

from iceclaw.agents.digest import build_digest, build_own_summary
from iceclaw.agents.specialists import (
    ALLOWED_MODALITIES,
    ALLOWED_TOOLS,
    AUDITOR,
    GEOLOGY,
    PRED,
    ROLES,
    SHADOW,
    SKEPTIC,
    TEMP,
    TERRAIN,
    THERMAL,
    MafTurnRunner,
    _budget_line,
)
from iceclaw.agents.tools import make_tools
from iceclaw.contracts import EvidenceEvent
from iceclaw.swarm.runtime import TOOL_PARAMS, AgentSpec, AgentState, SwarmRuntime, TurnRunner

SINGLE, ISOLATED, COMMUNICATING = (
    "single_generalist",
    "isolated_specialists",
    "communicating_specialists",
)
LLM_ARMS = (SINGLE, ISOLATED, COMMUNICATING)
GENERALIST = "generalist"
SPECIALISTS = (THERMAL, GEOLOGY, AUDITOR, SKEPTIC)

TASK = """\
TASK: screen ONE lunar data case for data or processing faults, acting only through tools.
The case has four co-registered layers of one region [unit]: nasa_ibm_prediction_raw
[prospectivity_index: a regression model's ice-prospectivity index; raw values may leave [0,1]],
pixelwise_temperature [K], pixelwise_shadow_mask [fraction; shadowed where >= 0.5],
pixelwise_terrain [deg]. A case is either an internally consistent copy of the data or it contains
a synthetic fault (for example a wrong unit or scale, a corrupted or missing block, an inverted
mask, a layer spatially misaligned with the others, mislabeled layers). You are not told which.

Rules
- Numbers come ONLY from tool results; cite the measurement event ids. Never invent numbers.
- Thresholds you pass to tools are your screening parameters, not physical constants.
- The prospectivity index is not ice; never imply that ice was found.
- Budgets are small and shared by everyone working on this case; make a few high-value calls and
  issue independent tool calls together in one step. Identical repeat calls may be cached.
- Natural nodata (uncovered pixels) is not a fault by itself.
- Before you finish, call submit_assessment with your verdict (fault_suspected or no_fault_found)
  citing measurement ids; you may replace it later if new evidence changes your view.
- End each turn with one or two plain sentences, no new numbers.
"""

FRAMING = {
    SINGLE: (
        "You are the only scientist on this case. You have every layer and every tool. Your "
        "submitted verdict is the final decision for the case."
    ),
    ISOLATED: (
        "You are one of four specialists screening this case INDEPENDENTLY: you cannot see, cite "
        "or contact the others and they cannot see you. The team counts the case as faulty if any "
        "member's latest verdict is fault_suspected."
    ),
    COMMUNICATING: (
        "You are one of four specialists screening this case TOGETHER: the others run concurrently "
        "and publish evidence you can read (bounded digest; get_event for full text), build on, "
        "ask them to test (request_test) or challenge. Disagreement is not required; never "
        "manufacture a challenge. The team counts the case as faulty if any member's latest "
        "verdict is fault_suspected."
    ),
}

GENERALIST_REMIT = (
    "Your remit: all four layers. Check each layer's values, units and missing data, and whether "
    "the layers are consistent with each other (for example shadowed pixels should be cold)."
)
ISOLATED_SKEPTIC_REMIT = (
    "Your remit: you own no single layer; independently cross-check any layer and the "
    "relationships between layers with your tools."
)
ALL_MODALITIES = frozenset({PRED, TEMP, SHADOW, TERRAIN})


@dataclass(frozen=True)
class ArmPlan:
    arm: str
    specs: tuple[AgentSpec, ...]
    communication: bool
    max_rounds_per_agent: int
    max_roundtrips_per_turn: int
    max_function_calls_per_turn: int


def _instructions(arm: str, title: str, remit: str) -> str:
    return f"{TASK}\n{FRAMING[arm]}\nYou are the {title}.\n{remit}\n"


def arm_plan(arm: str, total_llm_calls: int) -> ArmPlan:
    """Team structure for ``arm``. Per-turn caps let every arm reach the SAME total budget."""
    if arm == SINGLE:
        spec = AgentSpec(
            agent_id=GENERALIST,
            title="Generalist Lunar Data Scientist",
            instructions=_instructions(arm, "Generalist Lunar Data Scientist", GENERALIST_REMIT),
            allowed_tools=frozenset(TOOL_PARAMS),
            allowed_modalities=ALL_MODALITIES,
            can_recompute=True,
        )
        # one agent: it may spend the whole shared budget, within or across turns
        return ArmPlan(arm, (spec,), False, 6, total_llm_calls, total_llm_calls)
    if arm not in (ISOLATED, COMMUNICATING):
        raise ValueError(f"unknown LLM arm {arm!r}")
    comm = arm == COMMUNICATING
    specs = []
    for aid in SPECIALISTS:
        title, remit = ROLES[aid]
        if aid == SKEPTIC and not comm:
            remit = ISOLATED_SKEPTIC_REMIT
        specs.append(
            AgentSpec(
                agent_id=aid,
                title=title,
                instructions=_instructions(arm, title, remit),
                allowed_tools=ALLOWED_TOOLS[aid],
                allowed_modalities=ALLOWED_MODALITIES[aid],
                can_recompute=aid == SKEPTIC,
                starts_idle=comm and aid == SKEPTIC,  # with no bus it must start on its own
                wake_on_claims=comm and aid == SKEPTIC,
            )
        )
    return ArmPlan(arm, tuple(specs), comm, 3, 6, 10)


def build_benchmark_prompt(
    rt: SwarmRuntime, spec: AgentSpec, st: AgentState, new: list[EvidenceEvent]
) -> str:
    """Per-turn message: budget, case context (first turn), own events, others' events (only when
    communication is on), and the step instruction. Nothing about faults or truth is included."""
    parts = [_budget_line(rt).format(rounds=st.rounds + 1)]
    if rt.limits.reserve is not None:
        av = rt.available_to(spec.agent_id)
        parts.append(
            "Available to you now (your guaranteed share plus the common pool): "
            + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in av.items())
            + ". Your unused guaranteed share returns to the team once you file your assessment."
        )
    if st.rounds == 0:
        parts.append("Case context (JSON): " + rt.context.model_dump_json())
        parts.append(
            "Your tools: "
            + ", ".join(sorted(spec.allowed_tools))
            + "; layers: "
            + ", ".join(sorted(spec.allowed_modalities))
        )
    own = build_own_summary(rt, st.own_event_ids)
    if own:
        parts.append("Your recent events:\n" + own)
    if rt.communication and new:
        digest = build_digest(rt, spec.agent_id, new, rt.limits.digest_max_chars)
        if digest:
            parts.append("Events published by others since your last turn:\n" + digest)
    if st.rounds == 0 and not (rt.communication and new):
        parts.append(
            "Start: decide what is worth measuring within your remit and budget, measure it, then "
            "call submit_assessment."
        )
    elif spec.agent_id not in rt.assessments:
        parts.append(
            "You have not filed submit_assessment yet. Measure anything still essential, then file "
            "your verdict (fault_suspected or no_fault_found) citing your measurement ids."
        )
    else:
        parts.append(
            "Decide whether anything is warranted now (a test, a request, a challenge, or a "
            "replaced assessment backed by new evidence). It is fine to do nothing."
        )
    prompt = "\n\n".join(parts)
    cap = rt.limits.max_prompt_chars
    if len(prompt) > cap:
        prompt = prompt[: cap - 60] + "\n[prompt clipped to the configured cap; use list_events]"
    return prompt


def build_arm_runners(runtime: SwarmRuntime, client_factory: Any) -> dict[str, TurnRunner]:
    """One fresh MAF ``Agent`` (fresh client, no server-side storage) per agent of this run."""
    runners: dict[str, TurnRunner] = {}
    for spec in runtime.specs.values():
        agent = Agent(
            client_factory(runtime, spec.agent_id),
            instructions=spec.instructions,
            name=spec.agent_id,
            description=spec.title,
            tools=make_tools(runtime, spec),
            default_options={"store": False},
        )
        runners[spec.agent_id] = MafTurnRunner(agent)
    return runners
