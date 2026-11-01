"""Approved tool surface for the specialists, as Microsoft Agent Framework function tools.

Each tool is a thin closure over ``SwarmRuntime`` for ONE agent id. None takes a path, a file name
or any evaluator reference; all validation and every scientific number lives in the runtime and
``iceclaw.science``. Return values are JSON strings (including structured errors) so the model sees
exactly what was recorded.
"""

from __future__ import annotations

import functools
import json
from typing import Annotated, Any

from agent_framework import tool

from iceclaw.swarm.runtime import AgentSpec, SwarmRuntime

Ids = Annotated[list[str] | None, "event ids (e.g. ['evt-0003'])"]


def _json(fn):
    """Run a runtime action ON THE EVENT-LOOP THREAD and JSON-encode its dict result.

    The wrapper is ``async`` on purpose: Microsoft Agent Framework executes synchronous tool
    functions in worker threads, which would race on the runtime's counters and SQLite
    connection. A coroutine runs atomically between awaits, so every action is serialized.
    Unexpected exceptions are reported to the model instead of derailing the loop.
    """

    @functools.wraps(fn)
    async def inner(*a, **k) -> str:
        try:
            return json.dumps(fn(*a, **k), default=str)
        except Exception as e:
            return json.dumps(
                {
                    "ok": False,
                    "error": "INTERNAL_ERROR",
                    "message": f"{type(e).__name__}: {e}"[:300],
                }
            )

    return inner


def make_tools(runtime: SwarmRuntime, spec: AgentSpec) -> list[Any]:
    aid = spec.agent_id
    tools_txt = ", ".join(sorted(spec.allowed_tools))
    mods_txt = ", ".join(sorted(spec.allowed_modalities))

    @tool(
        name="run_science_tool",
        description=(
            "Run ONE approved deterministic science tool and publish the result as evidence. "
            f"Tools: {tools_txt}. Layers: {mods_txt}. modality is required except for "
            "thermal_psr_consistency / terrain_slope_summary (fixed layers). Optional "
            "window=[row0,row1,col0,col1]. Identical calls are cached free. "
            "in_response_to=<test_request id> answers a request addressed to you."
        ),
    )
    @_json
    def run_science_tool(
        tool_id: Annotated[str, "science tool id"],
        modality: Annotated[str | None, "layer name; omit for fixed-layer tools"] = None,
        window: Annotated[list[int] | None, "[row0,row1,col0,col1]"] = None,
        in_response_to: Annotated[str | None, "test_request event id to answer"] = None,
        k_mad: Annotated[float | None, "discontinuity_candidates only (0.5-50)"] = None,
        top_n: Annotated[int | None, "discontinuity_candidates only (0-10)"] = None,
        edge_width_px: Annotated[int | None, "boundary_check only (1-64)"] = None,
        qs: Annotated[list[float] | None, "quantiles only, each in [0,1]"] = None,
        psr_threshold: Annotated[float | None, "thermal_psr_consistency only"] = None,
        cold_trap_k: Annotated[
            float | None, "thermal_psr_consistency only: your chosen K threshold"
        ] = None,
        steep_deg: Annotated[
            float | None, "terrain_slope_summary only: your chosen degrees threshold"
        ] = None,
    ) -> str:
        return runtime.science_tool(
            aid,
            tool_id,
            modality,
            window,
            in_response_to,
            k_mad=k_mad,
            top_n=top_n,
            edge_width_px=edge_width_px,
            qs=qs,
            psr_threshold=psr_threshold,
            cold_trap_k=cold_trap_k,
            steep_deg=steep_deg,
        )  # type: ignore[return-value]

    @tool(
        name="publish_claim",
        description=(
            "Publish your interpretation. evidence_type 'computed_metric' (default) requires "
            "cites = "
            "measurement/test_result event ids that support every number you state; use "
            "'hypothesis' for an untested idea. status: provisional|supported|unresolved."
        ),
    )
    @_json
    def publish_claim(
        text: Annotated[str, "the claim; cite numbers only from the events you cite"],
        cites: Ids = None,
        evidence_type: Annotated[
            str, "computed_metric|observation|model_prediction|hypothesis"
        ] = "computed_metric",
        status: Annotated[str, "provisional|supported|unresolved"] = "provisional",
        alternatives: Annotated[list[str] | None, "alternative explanations"] = None,
        limitations: Annotated[list[str] | None, "limitations"] = None,
    ) -> str:
        return runtime.publish_claim(
            aid, text, cites, evidence_type, status, alternatives, limitations
        )  # type: ignore[return-value]

    @tool(
        name="publish_challenge",
        description=(
            "Challenge ANOTHER agent's claim/revision/test_result with a specific reason. Cite any "
            "evidence event ids that support the challenge. Only challenge when evidence warrants."
        ),
    )
    @_json
    def publish_challenge(
        target_event_id: Annotated[str, "the event being challenged"],
        text: Annotated[str, "what is questionable and why"],
        evidence_ids: Ids = None,
        alternatives: Annotated[list[str] | None, "alternative explanations"] = None,
        limitations: Annotated[list[str] | None, "limitations of your challenge"] = None,
    ) -> str:
        return runtime.publish_challenge(
            aid, target_event_id, text, evidence_ids, alternatives, limitations
        )  # type: ignore[return-value]

    @tool(
        name="request_test",
        description=(
            "Ask ANOTHER specialist to run one of THEIR approved tools. They are woken and "
            "may run it "
            "with in_response_to=<your request id>. to_agent must be able to run tool_id/modality."
        ),
    )
    @_json
    def request_test(
        to_agent: Annotated[str, "agent id of the addressee"],
        tool_id: Annotated[str, "tool the addressee should run"],
        rationale: Annotated[str, "why this test is informative"],
        modality: Annotated[str | None, "layer for the test"] = None,
        window: Annotated[list[int] | None, "[row0,row1,col0,col1]"] = None,
        params: Annotated[
            dict[str, float] | None, "tool parameters, e.g. {'cold_trap_k': 120}"
        ] = None,
        cites: Ids = None,
    ) -> str:
        return runtime.request_test(
            aid, to_agent, tool_id, rationale, modality, window, cites, **(params or {})
        )  # type: ignore[return-value]

    @tool(
        name="publish_revision",
        description=(
            "Revise YOUR OWN earlier claim. new_evidence_ids must include at least one measurement/"
            "test_result event published AFTER that claim. Optionally cite challenges you "
            "respond to."
        ),
    )
    @_json
    def publish_revision(
        revised_event_id: Annotated[str, "your earlier claim or revision"],
        new_evidence_ids: Annotated[
            list[str], "measurement/test_result events that changed your view"
        ],
        text: Annotated[str, "the revised assessment"],
        status: Annotated[str, "provisional|supported|unresolved"] = "provisional",
        responding_to: Ids = None,
        alternatives: Annotated[list[str] | None, "alternative explanations"] = None,
        limitations: Annotated[list[str] | None, "limitations"] = None,
    ) -> str:
        return runtime.publish_revision(
            aid,
            revised_event_id,
            new_evidence_ids,
            text,
            status,
            responding_to,
            alternatives,
            limitations,
        )  # type: ignore[return-value]

    @tool(
        name="conclude",
        description=(
            "Record your closing summary (you may still be woken by challenges or requests). Use "
            "status 'supported' only if your cited evidence supports it, else 'unresolved'."
        ),
    )
    @_json
    def conclude(
        summary: Annotated[str, "closing assessment, no new numbers"],
        cites: Ids = None,
        status: Annotated[str, "supported|unresolved|provisional"] = "unresolved",
        limitations: Annotated[list[str] | None, "unresolved questions / limits"] = None,
    ) -> str:
        return runtime.conclude(  # type: ignore[return-value]
            aid, summary, cites, status, limitations
        )

    @tool(
        name="list_events",
        description="List recent published events (compact). Optional kind filter.",
    )
    @_json
    def list_events(
        kinds: Annotated[
            list[str] | None,
            "claim|measurement|challenge|test_request|test_result|revision|completion|failure",
        ] = None,
        since_event_id: Annotated[str | None, "only events after this id"] = None,
    ) -> str:
        events = runtime.list_events(aid, kinds, since_event_id)
        return {"events": events}  # type: ignore[return-value]

    @tool(name="get_event", description="Read one event with its measurement summary, if any.")
    @_json
    def get_event(event_id: Annotated[str, "event id"]) -> str:
        return runtime.get_event_detail(event_id, aid)  # type: ignore[return-value]

    out = [run_science_tool, publish_claim]
    if runtime.communication:  # isolated agents (P7) get no tools that address other agents
        out += [publish_challenge, request_test]
    out += [publish_revision, conclude, list_events, get_event]
    if runtime.review_mode:

        @tool(
            name="submit_review",
            description=(
                "File YOUR review of the target window (a later call replaces the earlier one). "
                "finding: 'supports_screening' if YOUR evidence is consistent with the screening "
                "case for this window, 'weak_or_conflicting' if it weakens or contradicts it, "
                "'insufficient_evidence' if you cannot judge. cites: your measurement event ids "
                "(at least one measured on the target window). Also list missing_evidence, "
                "alternatives (competing explanations) and follow_up (suggested analyses)."
            ),
        )
        @_json
        def submit_review(
            finding: Annotated[str, "supports_screening|weak_or_conflicting|insufficient_evidence"],
            cites: Annotated[list[str], "measurement event ids, >= 1 on the target window"],
            rationale: Annotated[str, "one or two sentences; numbers only from cited events"],
            missing_evidence: Annotated[list[str] | None, "evidence you lacked"] = None,
            alternatives: Annotated[list[str] | None, "competing explanations"] = None,
            follow_up: Annotated[list[str] | None, "suggested follow-up analyses"] = None,
        ) -> str:
            return runtime.submit_review(  # type: ignore[return-value]
                aid, finding, cites, rationale, missing_evidence, alternatives, follow_up
            )

        out.append(submit_review)
    elif runtime.require_assessment:
        mods = ", ".join(runtime.context.available_modalities)

        @tool(
            name="submit_assessment",
            description=(
                "File your machine-readable assessment of THIS case (a later call replaces the "
                "earlier one). verdict: 'fault_suspected' if your evidence indicates a data or "
                "processing fault (wrong unit or scale, corrupted/missing block, inverted mask, "
                "layers misaligned or mislabeled, ...), else 'no_fault_found'. "
                f"suspected_layers: subset of [{mods}] (required for fault_suspected). cites: "
                "measurement event ids supporting the verdict (at least one)."
            ),
        )
        @_json
        def submit_assessment(
            verdict: Annotated[str, "fault_suspected|no_fault_found"],
            cites: Annotated[list[str], "measurement event ids supporting the verdict"],
            rationale: Annotated[str, "one or two sentences; numbers only from cited events"],
            suspected_layers: Annotated[list[str] | None, "layers you suspect"] = None,
        ) -> str:
            return runtime.submit_assessment(  # type: ignore[return-value]
                aid, verdict, suspected_layers, cites, rationale
            )

        out.append(submit_assessment)
    if spec.can_recompute:

        @tool(
            name="recompute_measurement",
            description=(
                "Independently re-run a stored deterministic measurement and publish "
                "whether it reproduced."
            ),
        )
        @_json
        def recompute_measurement(
            tool_run_id: Annotated[str, "tool_run_id from a measurement"],
        ) -> str:
            return runtime.recompute_measurement(aid, tool_run_id)  # type: ignore[return-value]

        out.append(recompute_measurement)
    return out
