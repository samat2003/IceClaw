"""Targeted cross-modal window review: one shared question, distributed specialist evidence.

A deterministic screen (``science.triage``) picks a target window; this module only frames the
shared research question. It does NOT script findings, messages, disagreement or revisions, and it
does not hand any specialist a number it did not measure: the prompt names the window, its
location and why it was selected, never its statistics. Each specialist keeps the layers and tools
of its P5 role; the Skeptic owns no data, reads what the others publish and may request tests
from the owner of a layer. "The evidence is consistent" and "the evidence is insufficient" are
valid findings. If a specialist files no review the review is INCOMPLETE; nothing is substituted.
"""

from __future__ import annotations

from typing import Any

from iceclaw.agents.benchmark_arms import (
    ALL_MODALITIES,
    COMMUNICATING,
    GENERALIST,
    GENERALIST_REMIT,
    ISOLATED,
    ISOLATED_SKEPTIC_REMIT,
    SINGLE,
    SPECIALISTS,
    ArmPlan,
)
from iceclaw.agents.digest import build_digest, build_own_summary
from iceclaw.agents.specialists import (
    ALLOWED_MODALITIES,
    ALLOWED_TOOLS,
    ROLES,
    SKEPTIC,
    _budget_line,
)
from iceclaw.contracts import EvidenceEvent
from iceclaw.swarm.runtime import TOOL_PARAMS, AgentSpec, AgentState, SwarmRuntime

CATEGORY_TEXT = {
    "followup": (
        "it contains at least one pixel meeting a joint screening rule (permanently shadowed, "
        "cold, gentle slope AND a high model index)"
    ),
    "review": (
        "it is a high-model-index window whose observation layers appear to support that index "
        "less well (a model-observation discrepancy to review)"
    ),
    "other": "it was chosen for review by the operator",
}

TASK = """\
TASK: help answer ONE shared research question about ONE window of a lunar south-polar tile,
acting only through tools. Layers [unit]: nasa_ibm_prediction_raw [prospectivity_index: a model's
ice-prospectivity index, raw values may leave [0,1]], pixelwise_temperature [K],
pixelwise_shadow_mask [fraction; shadowed where >= 0.5], pixelwise_terrain [deg].

SHARED QUESTION: for the target window below, do the independent measurements from the different
layers support or weaken the case for following this window up? ("Support" means the observations
are consistent with a cold, permanently shadowed, gently sloping place that the model also rates
highly. It is NOT evidence that ice exists.)
{target}

Rules
- Measure YOUR layers ON THE TARGET WINDOW: pass window=[{r0},{r1},{c0},{c1}] to every tool you
  want judged on it. Whole-tile numbers do not answer the window question.
- Numbers come ONLY from tool results; cite the measurement event ids. Never invent numbers.
- Thresholds you pass to tools are your screening parameters, not physical constants.
- The prospectivity index is not ice; agreement between the model and its related input layers is
  not independent confirmation of ice. Never imply ice was found.
- Budgets are small and shared by everyone; make a few high-value calls and issue independent tool
  calls together in one step. Identical repeat calls may be cached.
- Before you finish call submit_review (finding, cites >= 1 measurement made on the target window,
  missing_evidence, alternatives, follow_up). You may replace it if new evidence changes your view.
- End each turn with one or two plain sentences, no new numbers.
"""

FRAMING = {
    SINGLE: "You are the only scientist on this review; your review is the team's review.",
    ISOLATED: (
        "You are one of four specialists reviewing the window INDEPENDENTLY: you cannot see, cite "
        "or contact the others and they cannot see you. The team review is complete only if all "
        "four members file a review."
    ),
    COMMUNICATING: (
        "You are one of four specialists reviewing the window TOGETHER: the others run "
        "concurrently and publish evidence you can read (bounded digest; get_event for full "
        "text), build on, ask them to test (request_test: the owner of a layer can run tools on "
        "it) or challenge. Disagreement is not required and must never be manufactured; "
        "'consistent' and 'insufficient evidence' are valid. The team review is complete only if "
        "all four members file a review."
    ),
}

SKEPTIC_REMIT_COMMUNICATING = (
    "You own no data. Read what the others publish, re-run stored measurements with "
    "recompute_measurement, request a missing measurement from the specialist who owns that layer "
    "when it would change the assessment, challenge a claim ONLY when evidence supports doing so "
    "(cite it), and record unresolved uncertainties and follow-up work. 'No warranted "
    "contradiction' is a valid, honest result. You act when others publish."
)


def target_text(target: dict[str, Any]) -> str:
    r0, r1, c0, c1 = target["window"]
    why = CATEGORY_TEXT.get(target["category"], CATEGORY_TEXT["other"])
    return (
        f"TARGET WINDOW {target['label']}: rows {r0}-{r1 - 1}, columns {c0}-{c1 - 1} (native "
        f"pixels; tools take window=[{r0},{r1},{c0},{c1}]), centre latitude "
        f"{target['centre_lat']:.4f}, longitude {target['centre_lon']:.4f}. It was selected by "
        f"deterministic code because {why}."
    )


def _instructions(arm: str, title: str, remit: str, target: dict[str, Any]) -> str:
    r0, r1, c0, c1 = target["window"]
    task = TASK.format(target=target_text(target), r0=r0, r1=r1, c0=c0, c1=c1)
    return f"{task}\n{FRAMING[arm]}\nYou are the {title}.\n{remit}\n"


def review_plan(arm: str, total_llm_calls: int, target: dict[str, Any]) -> ArmPlan:
    """Team structure for a targeted review; same totals and per-turn caps as the P7 arms."""
    if arm == SINGLE:
        spec = AgentSpec(
            agent_id=GENERALIST,
            title="Generalist Lunar Data Scientist",
            instructions=_instructions(
                arm, "Generalist Lunar Data Scientist", GENERALIST_REMIT, target
            ),
            allowed_tools=frozenset(TOOL_PARAMS),
            allowed_modalities=ALL_MODALITIES,
            can_recompute=True,
        )
        return ArmPlan(arm, (spec,), False, 6, total_llm_calls, total_llm_calls)
    if arm not in (ISOLATED, COMMUNICATING):
        raise ValueError(f"unknown review arm {arm!r}")
    comm = arm == COMMUNICATING
    specs = []
    for aid in SPECIALISTS:
        title, remit = ROLES[aid]
        if aid == SKEPTIC:
            remit = SKEPTIC_REMIT_COMMUNICATING if comm else ISOLATED_SKEPTIC_REMIT
        specs.append(
            AgentSpec(
                agent_id=aid,
                title=title,
                instructions=_instructions(arm, title, remit, target),
                allowed_tools=ALLOWED_TOOLS[aid],
                allowed_modalities=ALLOWED_MODALITIES[aid],
                can_recompute=aid == SKEPTIC,
                starts_idle=comm and aid == SKEPTIC,
                wake_on_claims=comm and aid == SKEPTIC,
            )
        )
    return ArmPlan(arm, tuple(specs), comm, 3, 6, 10)


def build_review_prompt(
    rt: SwarmRuntime, spec: AgentSpec, st: AgentState, new: list[EvidenceEvent]
) -> str:
    """Bounded per-turn message: budget, availability, own events, others' events (only when
    communication is on) and the step instruction. Contains no statistics and no truth."""
    parts = [_budget_line(rt).format(rounds=st.rounds + 1)]
    if rt.limits.reserve is not None:
        av = rt.available_to(spec.agent_id)
        parts.append(
            "Available to you now (your guaranteed share plus the common pool): "
            + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in av.items())
            + ". Your unused guaranteed share returns to the team once you file your review."
        )
    if st.rounds == 0:
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
            "Start: decide what is worth measuring on the target window within your remit and "
            "budget, measure it, then call submit_review."
        )
    elif spec.agent_id not in rt.assessments:
        parts.append(
            "You have not filed submit_review yet. Measure anything still essential on the "
            "target window, then file your review citing your measurement ids."
        )
    else:
        parts.append(
            "Decide whether anything is warranted now (a test, a request, a challenge, or a "
            "replaced review backed by new evidence). It is fine to do nothing."
        )
    prompt = "\n\n".join(parts)
    cap = rt.limits.max_prompt_chars
    if len(prompt) > cap:
        prompt = prompt[: cap - 60] + "\n[prompt clipped to the configured cap; use list_events]"
    return prompt
