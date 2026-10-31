"""Asynchronous multi-agent runtime: validated evidence publishing, delivery, budgets, termination.

Design rules (tested):

* No central LLM and no fixed scientific order. Agents start concurrently and act only through
  the validated actions below; the runtime enforces *form* (references, budgets), never content.
* Every science number comes from a deterministic IceClaw tool (``iceclaw.science``) executed
  here; it is stored as a ``ToolMeasurement`` and published as an authentic ``measurement`` /
  ``test_result`` event. Agents cannot write numbers into the log themselves.
* Agents get an ``AgentRegionContext`` and these actions only: no paths, no file access, nothing
  from the evaluator side. This module must not import ``iceclaw.evaluation`` (tested).
* Bounded by ``RunBudget`` (events, hops, tool calls, model calls, wall time) plus a per-agent
  round cap. Termination is deterministic: quiescence (nobody running, nobody woken), budget, or
  timeout, each recorded by a neutral ``supervisor`` event.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from iceclaw.config import Settings
from iceclaw.contracts import (
    AgentRegionContext,
    EventKind,
    EvidenceEvent,
    EvidenceType,
    HypothesisStatus,
    RunBudget,
    ToolMeasurement,
)
from iceclaw.data.catalog import available_tool_names, discover
from iceclaw.science.common import RegionArrays, ScienceToolError
from iceclaw.science.model_audit import TOOLS, recompute, run_tool
from iceclaw.swarm.store import EvidenceStore, StoreError

SUPERVISOR = "supervisor"
EVIDENCE_KINDS = (EventKind.MEASUREMENT, EventKind.TEST_RESULT)
ACTIONABLE = (
    EventKind.CLAIM,
    EventKind.CHALLENGE,
    EventKind.TEST_REQUEST,
    EventKind.TEST_RESULT,
    EventKind.REVISION,
)
MAX_TEXT = 1500
MAX_LIST = 6

# Which keyword parameters each tool accepts (everything else is rejected) and their bounds.
TOOL_PARAMS: dict[str, dict[str, tuple[float, float]]] = {
    "valid_pixel_summary": {},
    "quantiles": {},  # ``qs`` handled separately
    "missingness": {},
    "unit_interval_check": {},
    "gradient_stats": {},
    "discontinuity_candidates": {"k_mad": (0.5, 50.0), "top_n": (0, 10)},
    "boundary_check": {"edge_width_px": (1, 64)},
    "peak_location": {},
    "thermal_psr_consistency": {"psr_threshold": (0.0, 1.0), "cold_trap_k": (1.0, 400.0)},
    "terrain_slope_summary": {"steep_deg": (0.0, 90.0)},
}
FIXED_LAYER_TOOLS = {"thermal_psr_consistency", "terrain_slope_summary"}  # no modality argument
assert set(TOOL_PARAMS) == set(TOOLS)


class BudgetExhausted(RuntimeError):
    def __init__(self, kind: str) -> None:
        super().__init__(f"BUDGET_EXHAUSTED: {kind}")
        self.kind = kind


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    title: str
    instructions: str
    allowed_tools: frozenset[str]
    allowed_modalities: frozenset[str]
    can_recompute: bool = False
    starts_idle: bool = False  # take no first turn; act only when woken by others' events
    wake_on_claims: bool = False  # woken by other agents' claims/revisions (else only if concerned)


@dataclass
class RunLimits:
    budget: RunBudget
    max_rounds_per_agent: int = 4
    max_function_calls_per_turn: int = 10  # forwarded to MAF's function-invocation loop
    max_roundtrips_per_turn: int = 6


class TurnRunner(Protocol):
    async def take_turn(self, prompt: str) -> str: ...


@dataclass
class AgentState:
    rounds: int = 0
    busy: bool = False
    pending_first: bool = True
    wake_pending: bool = False
    exited: bool = False
    exit_reason: str | None = None
    inbox: list[EvidenceEvent] = field(default_factory=list)
    own_event_ids: list[str] = field(default_factory=list)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    failures: int = 0


@dataclass
class RunSummary:
    investigation_id: str
    region_id: str
    ended_reason: str
    elapsed_s: float
    events: int
    tool_calls: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    rounds: dict[str, int]
    exit_reasons: dict[str, str | None]
    budget: dict[str, Any]
    problems: list[str]


def _err(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": message, **extra}


def _clip(text: str, n: int = MAX_TEXT) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def _clean_list(items: list[str] | None) -> list[str]:
    return [_clip(str(i), 300) for i in (items or [])][:MAX_LIST]


def _round(v: Any) -> Any:
    if isinstance(v, float):
        return float(f"{v:.6g}")
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


class SwarmRuntime:
    def __init__(
        self,
        *,
        investigation_id: str,
        context: AgentRegionContext,
        arrays: RegionArrays,
        store: EvidenceStore,
        specs: list[AgentSpec],
        limits: RunLimits,
        provider: str = "unspecified",
        model: str = "unspecified",
    ) -> None:
        if len({s.agent_id for s in specs}) != len(specs) or SUPERVISOR in {
            s.agent_id for s in specs
        }:
            raise ValueError("agent ids must be unique and not 'supervisor'")
        self.investigation_id = investigation_id
        self.context = context
        self.arrays = arrays
        self.store = store
        self.specs = {s.agent_id: s for s in specs}
        self.limits = limits
        self.provider, self.model = provider, model
        self.state = {s.agent_id: AgentState(pending_first=not s.starts_idle) for s in specs}
        self.tool_calls = 0
        self.llm_calls = 0
        self.tokens = {"input": 0, "output": 0, "total": 0}
        self._n = 0
        self._finished = False
        self._ended_reason = "RUNNING"
        self._dedup: dict[tuple, str] = {}

    # ---------------------------------------------------------------- publishing
    def _next_id(self) -> str:
        self._n += 1
        return f"evt-{self._n:04d}"

    def _publish(
        self, agent_id: str, kind: EventKind, **fields: Any
    ) -> tuple[EvidenceEvent | None, dict | None]:
        """Validate, hop-check, budget-check, dedup, append and deliver. Returns (event, error)."""
        parents: list[str] = list(dict.fromkeys(fields.get("parent_event_ids") or []))
        fields["parent_event_ids"] = parents
        supervisor = agent_id == SUPERVISOR
        if not supervisor and self.store.count_events() >= self.limits.budget.max_events:
            return None, _err("EVENT_BUDGET_EXHAUSTED", "no events left; stop and conclude")
        parent_hops = []
        for pid in parents:
            h = self.store.hop_of(pid)
            if h is None:
                return None, _err("UNKNOWN_EVENT", f"event {pid!r} does not exist")
            parent_hops.append(h)
        hop = 0 if not parent_hops else 1 + max(parent_hops)
        if not supervisor and hop > self.limits.budget.max_hops:
            return None, _err(
                "HOP_LIMIT", f"reaction chain would reach hop {hop} > {self.limits.budget.max_hops}"
            )
        key = (
            agent_id,
            kind.value,
            (fields.get("claim_or_request") or "").strip().lower(),
            tuple(sorted(parents)),
            fields.get("tool_run_id"),
        )
        if key in self._dedup:
            return None, _err(
                "DUPLICATE", "identical event already published", existing_event_id=self._dedup[key]
            )
        try:
            event = EvidenceEvent(
                event_id=self._next_id(),
                investigation_id=self.investigation_id,
                region_id=self.context.region_id,
                agent_id=agent_id,
                timestamp_utc=datetime.now(UTC),
                event_kind=kind,
                **fields,
            )
            self.store.append_event(event, hop)
        except (ValidationError, StoreError) as e:
            self._n -= 1
            return None, _err("INVALID_EVENT", _clip(str(e), 400))
        self._dedup[key] = event.event_id
        if not supervisor:
            self.state[agent_id].own_event_ids.append(event.event_id)
        self._deliver(event)
        return event, None

    def _deliver(self, event: EvidenceEvent) -> None:
        """Hand the event to every other agent's inbox; wake only those it concerns."""
        wake: set[str] = set()
        if event.event_kind in (EventKind.CLAIM, EventKind.REVISION):
            wake = {a for a, sp in self.specs.items() if sp.wake_on_claims} - {event.agent_id}
        elif event.event_kind is EventKind.CHALLENGE:
            target = self.store.get_event(event.parent_event_ids[0])
            wake = {target.agent_id} if target else set()
        elif event.event_kind is EventKind.TEST_REQUEST:
            wake = {self._request_addressee(event)}
        elif event.event_kind is EventKind.TEST_RESULT:
            req = self.store.get_event(event.parent_event_ids[0])
            wake = {req.agent_id} if req else set()
        for aid, st in self.state.items():
            if aid == event.agent_id:
                continue
            st.inbox.append(event)
            if aid in wake and event.event_kind in ACTIONABLE and not st.exited:
                st.wake_pending = True
                st.wake.set()

    @staticmethod
    def _request_addressee(event: EvidenceEvent) -> str:
        try:
            return str(json.loads(event.claim_or_request or "{}").get("to", ""))
        except json.JSONDecodeError:
            return ""

    # ------------------------------------------------------------- action: science tool
    def _validate_tool_call(
        self,
        spec: AgentSpec,
        tool_id: str,
        modality: str | None,
        window: list[int] | None,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return (kwargs for run_tool, None) or (None, error); ``spec`` must be able to run it."""
        if tool_id not in TOOL_PARAMS:
            return None, _err(
                "UNKNOWN_TOOL", f"unknown tool {tool_id!r}", available=sorted(spec.allowed_tools)
            )
        if tool_id not in spec.allowed_tools:
            return None, _err(
                "TOOL_NOT_PERMITTED",
                f"{spec.agent_id} may not run {tool_id!r}",
                available=sorted(spec.allowed_tools),
            )
        kwargs: dict[str, Any] = {}
        if tool_id in FIXED_LAYER_TOOLS:
            if modality:
                return None, _err(
                    "MODALITY_NOT_ALLOWED", f"{tool_id} uses fixed layers; omit modality"
                )
        else:
            if not modality:
                return None, _err(
                    "MODALITY_REQUIRED",
                    "modality is required",
                    allowed=sorted(spec.allowed_modalities),
                )
            if modality not in spec.allowed_modalities or modality not in self.arrays.layers:
                return None, _err(
                    "MODALITY_NOT_ALLOWED",
                    f"{modality!r} is not available to {spec.agent_id}",
                    allowed=sorted(spec.allowed_modalities),
                )
            kwargs["modality"] = modality
        if window is not None:
            if len(window) != 4 or any(not isinstance(i, int) for i in window):
                return None, _err("BAD_WINDOW", "window must be [row0,row1,col0,col1] integers")
            kwargs["window"] = tuple(window)
        allowed = TOOL_PARAMS[tool_id]
        for name, value in params.items():
            if value is None:
                continue
            if name == "qs" and tool_id == "quantiles":
                if (
                    not isinstance(value, list)
                    or not value
                    or len(value) > 10
                    or any(not isinstance(q, (int, float)) or not 0 <= q <= 1 for q in value)
                ):
                    return None, _err("BAD_PARAM", "qs must be 1-10 numbers within [0,1]")
                kwargs["qs"] = tuple(float(q) for q in value)
                continue
            if name not in allowed:
                return None, _err("BAD_PARAM", f"parameter {name!r} does not apply to {tool_id}")
            lo, hi = allowed[name]
            if not isinstance(value, (int, float)) or not lo <= value <= hi:
                return None, _err("BAD_PARAM", f"{name} must be within [{lo}, {hi}]")
            kwargs[name] = int(value) if name in ("top_n", "edge_width_px") else float(value)
        return kwargs, None

    def science_tool(
        self,
        agent_id: str,
        tool_id: str,
        modality: str | None = None,
        window: list[int] | None = None,
        in_response_to: str | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        spec = self.specs[agent_id]
        kwargs, err = self._validate_tool_call(spec, tool_id, modality, window, params)
        if err:
            return err
        parents: list[str] = []
        kind = EventKind.MEASUREMENT
        if in_response_to:
            req = self.store.get_event(in_response_to)
            if req is None or req.event_kind is not EventKind.TEST_REQUEST:
                return _err(
                    "BAD_REQUEST_REF", f"{in_response_to!r} is not an existing test_request"
                )
            if self._request_addressee(req) != agent_id:
                return _err("BAD_REQUEST_REF", "that test request is addressed to another agent")
            parents, kind = [in_response_to], EventKind.TEST_RESULT
        if self._finished:
            return _err("RUN_ENDED", "the investigation has ended")
        if self.tool_calls >= self.limits.budget.max_tool_calls:
            return _err(
                "TOOL_BUDGET_EXHAUSTED", "no tool calls left; publish what you have and conclude"
            )
        self.tool_calls += 1
        try:
            m = run_tool(self.arrays, tool_id, **kwargs)
        except ScienceToolError as e:
            ev, perr = self._publish(
                agent_id,
                EventKind.FAILURE,
                parent_event_ids=parents,
                claim_or_request=_clip(f"{tool_id} could not produce a value: {e}"),
                tool_id=tool_id,
                error_code=e.code,
                evidence_type=EvidenceType.COMPUTED_METRIC,
                limitations=[f"tool failure {e.code}"],
            )
            return perr or {"ok": False, "error": e.code, "event_id": ev.event_id if ev else None}
        cached = self.store.get_tool_run(m.tool_run_id)
        if cached:  # deterministic dedup: identical tool+params+source already measured
            self.tool_calls -= 1
            return {
                "ok": True,
                "cached": True,
                "event_id": cached[1],
                "tool_run_id": m.tool_run_id,
                "note": "identical measurement already exists; cite this event id",
                **self._summary(m),
            }
        ev, perr = self._publish(
            agent_id,
            kind,
            parent_event_ids=parents,
            claim_or_request=_clip(
                f"{tool_id}({self._param_text(m)}) -> {json.dumps(_round(m.value))}"
                if m.value is not None
                else f"{tool_id}: unavailable"
            ),
            evidence_type=EvidenceType.COMPUTED_METRIC,
            source_artifact_ids=list(m.source_file_ids),
            tool_id=m.tool_id,
            tool_run_id=m.tool_run_id,
            spatial_footprint=m.footprint,
            analysis_resolution_m=m.resolution_m,
            error_code=m.unavailable_reason.value if m.unavailable_reason else None,
            limitations=[f"unavailable: {m.unavailable_reason.value}"]
            if m.unavailable_reason
            else [],
        )
        if perr:
            self.tool_calls -= 1 if perr["error"] != "DUPLICATE" else 0
            return perr
        assert ev is not None
        self.store.append_tool_run(m, agent_id, ev.event_id, self.investigation_id)
        return {
            "ok": True,
            "cached": False,
            "event_id": ev.event_id,
            "tool_run_id": m.tool_run_id,
            **self._summary(m),
        }

    @staticmethod
    def _param_text(m: ToolMeasurement) -> str:
        p = {k: v for k, v in m.parameters.items() if k != "unavailable_detail" and v is not None}
        return json.dumps(p, sort_keys=True)

    @staticmethod
    def _summary(m: ToolMeasurement) -> dict[str, Any]:
        return {
            "tool_id": m.tool_id,
            "unit": m.unit,
            "valid_pixel_count": m.valid_pixel_count,
            "value": _round(m.value),
            "unavailable_reason": m.unavailable_reason.value if m.unavailable_reason else None,
            "parameters": {k: v for k, v in m.parameters.items() if v is not None},
        }

    def recompute_measurement(self, agent_id: str, tool_run_id: str) -> dict[str, Any]:
        """Independently re-run a stored deterministic measurement (skeptic cross-check)."""
        spec = self.specs[agent_id]
        if not spec.can_recompute:
            return _err("TOOL_NOT_PERMITTED", f"{agent_id} may not recompute measurements")
        got = self.store.get_tool_run(tool_run_id)
        if got is None:
            return _err("UNKNOWN_TOOL_RUN", tool_run_id)
        if self.tool_calls >= self.limits.budget.max_tool_calls:
            return _err("TOOL_BUDGET_EXHAUSTED", "no tool calls left")
        self.tool_calls += 1
        original, orig_event, _ = got
        try:
            again = recompute(original, self.arrays)
        except ScienceToolError as e:
            return _err(e.code, str(e))
        from iceclaw.science.model_audit import matches

        same = matches(original, again)
        ev, perr = self._publish(
            agent_id,
            EventKind.MEASUREMENT,
            parent_event_ids=[orig_event],
            claim_or_request=f"independent recomputation of {tool_run_id}: "
            + ("reproduced within tolerance" if same else "DID NOT reproduce"),
            evidence_type=EvidenceType.COMPUTED_METRIC,
            source_artifact_ids=list(original.source_file_ids),
            tool_id=original.tool_id,
            tool_run_id=f"{original.tool_run_id}#recompute",
            limitations=[] if same else ["recomputation mismatch"],
        )
        if perr:
            self.tool_calls -= 1
            return perr
        assert ev is not None
        self.store.append_tool_run(
            again.model_copy(update={"tool_run_id": ev.tool_run_id}),
            agent_id,
            ev.event_id,
            self.investigation_id,
        )
        return {
            "ok": True,
            "event_id": ev.event_id,
            "reproduced": same,
            "original_event_id": orig_event,
        }

    # ---------------------------------------------------- actions: claims / debate
    def _existing(self, ids: list[str] | None) -> tuple[list[EvidenceEvent], dict | None]:
        out = []
        for i in dict.fromkeys(ids or []):
            e = self.store.get_event(i)
            if e is None:
                return [], _err("UNKNOWN_EVENT", f"event {i!r} does not exist")
            out.append(e)
        return out, None

    def _inherit(self, cited: list[EvidenceEvent]) -> dict[str, Any]:
        ev = [e for e in cited if e.event_kind in EVIDENCE_KINDS]
        if not ev:
            return {}
        arts = list(dict.fromkeys(a for e in ev for a in e.source_artifact_ids))
        return {
            "source_artifact_ids": arts,
            "spatial_footprint": ev[0].spatial_footprint,
            "analysis_resolution_m": ev[0].analysis_resolution_m,
        }

    def publish_claim(
        self,
        agent_id: str,
        text: str,
        cites: list[str] | None = None,
        evidence_type: str = "computed_metric",
        status: str = "provisional",
        alternatives: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            et, hs = EvidenceType(evidence_type), HypothesisStatus(status)
        except ValueError as e:
            return _err("BAD_ENUM", str(e))
        if hs is HypothesisStatus.CHALLENGED:
            return _err("BAD_ENUM", "status 'challenged' is set only by challenges")
        cited, err = self._existing(cites)
        if err:
            return err
        if et is not EvidenceType.HYPOTHESIS and not any(
            e.event_kind in EVIDENCE_KINDS for e in cited
        ):
            return _err(
                "EVIDENCE_REQUIRED",
                f"a {et.value} claim must cite at least one measurement/test_result event id; "
                "use evidence_type='hypothesis' for an untested idea",
            )
        ev, perr = self._publish(
            agent_id,
            EventKind.CLAIM,
            parent_event_ids=[e.event_id for e in cited],
            claim_or_request=_clip(text),
            evidence_type=et,
            hypothesis_status=hs,
            alternative_explanations=_clean_list(alternatives),
            limitations=_clean_list(limitations),
            **self._inherit(cited),
        )
        return perr or {"ok": True, "event_id": ev.event_id}  # type: ignore[union-attr]

    def publish_challenge(
        self,
        agent_id: str,
        target_event_id: str,
        text: str,
        evidence_ids: list[str] | None = None,
        alternatives: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        target = self.store.get_event(target_event_id)
        if target is None:
            return _err("UNKNOWN_EVENT", f"event {target_event_id!r} does not exist")
        if target.agent_id == agent_id:
            return _err(
                "SELF_CHALLENGE", "challenge another agent's event; revise your own instead"
            )
        if target.event_kind not in (
            EventKind.CLAIM,
            EventKind.REVISION,
            EventKind.TEST_RESULT,
            EventKind.COMPLETION,
        ):
            return _err("BAD_TARGET", f"cannot challenge a {target.event_kind.value} event")
        cited, err = self._existing(evidence_ids)
        if err:
            return err
        ev, perr = self._publish(
            agent_id,
            EventKind.CHALLENGE,
            parent_event_ids=[target_event_id, *[e.event_id for e in cited]],
            claim_or_request=_clip(text),
            evidence_type=EvidenceType.COMPUTED_METRIC
            if any(e.event_kind in EVIDENCE_KINDS for e in cited)
            else EvidenceType.HYPOTHESIS,
            hypothesis_status=HypothesisStatus.CHALLENGED,
            alternative_explanations=_clean_list(alternatives),
            limitations=_clean_list(limitations),
            **self._inherit(cited),
        )
        return perr or {"ok": True, "event_id": ev.event_id}  # type: ignore[union-attr]

    def request_test(
        self,
        agent_id: str,
        to_agent: str,
        tool_id: str,
        rationale: str,
        modality: str | None = None,
        window: list[int] | None = None,
        cites: list[str] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        if to_agent == agent_id or to_agent not in self.specs:
            return _err(
                "BAD_ADDRESSEE",
                f"to_agent must be another agent in {sorted(set(self.specs) - {agent_id})}",
            )
        kwargs, err = self._validate_tool_call(
            self.specs[to_agent], tool_id, modality, window, params
        )
        if err:
            return {**err, "message": f"{err['message']} (for addressee {to_agent})"}
        cited, cerr = self._existing(cites)
        if cerr:
            return cerr
        body = json.dumps(
            {
                "to": to_agent,
                "tool": tool_id,
                "modality": modality,
                "window": window,
                "params": {k: v for k, v in params.items() if v is not None},
                "rationale": _clip(rationale, 600),
            },
            sort_keys=True,
        )
        ev, perr = self._publish(
            agent_id,
            EventKind.TEST_REQUEST,
            parent_event_ids=[e.event_id for e in cited],
            claim_or_request=body,
            tool_id=tool_id,
            evidence_type=EvidenceType.HYPOTHESIS,
        )
        _ = kwargs
        if perr:
            return perr
        assert ev is not None
        return {"ok": True, "event_id": ev.event_id, "addressed_to": to_agent}

    def publish_revision(
        self,
        agent_id: str,
        revised_event_id: str,
        new_evidence_ids: list[str],
        text: str,
        status: str = "provisional",
        responding_to: list[str] | None = None,
        alternatives: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        old = self.store.get_event(revised_event_id)
        if old is None:
            return _err("UNKNOWN_EVENT", f"event {revised_event_id!r} does not exist")
        if old.agent_id != agent_id or old.event_kind not in (EventKind.CLAIM, EventKind.REVISION):
            return _err("BAD_TARGET", "you can only revise your own claim/revision")
        try:
            hs = HypothesisStatus(status)
        except ValueError as e:
            return _err("BAD_ENUM", str(e))
        cited, err = self._existing(new_evidence_ids)
        if err:
            return err
        old_seq = self.store.seq_of(revised_event_id) or 0
        qualifying = [
            e
            for e in cited
            if e.event_kind in EVIDENCE_KINDS and (self.store.seq_of(e.event_id) or 0) > old_seq
        ]
        if not qualifying:
            return _err(
                "NEW_EVIDENCE_REQUIRED",
                "a revision must cite at least one measurement/test_result event "
                "published AFTER the claim it revises",
            )
        resp, rerr = self._existing(responding_to)
        if rerr:
            return rerr
        parents = [revised_event_id, *[e.event_id for e in cited], *[e.event_id for e in resp]]
        ev, perr = self._publish(
            agent_id,
            EventKind.REVISION,
            parent_event_ids=parents,
            claim_or_request=_clip(text),
            evidence_type=EvidenceType.COMPUTED_METRIC,
            hypothesis_status=hs,
            alternative_explanations=_clean_list(alternatives),
            limitations=_clean_list(limitations),
            **self._inherit(cited),
        )
        return perr or {"ok": True, "event_id": ev.event_id}  # type: ignore[union-attr]

    def conclude(
        self,
        agent_id: str,
        summary: str,
        cites: list[str] | None = None,
        status: str = "unresolved",
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            hs = HypothesisStatus(status)
        except ValueError as e:
            return _err("BAD_ENUM", str(e))
        cited, err = self._existing(cites)
        if err:
            return err
        ev, perr = self._publish(
            agent_id,
            EventKind.COMPLETION,
            parent_event_ids=[e.event_id for e in cited],
            claim_or_request=_clip(summary),
            hypothesis_status=hs,
            limitations=_clean_list(limitations),
            evidence_type=EvidenceType.HYPOTHESIS,
        )
        return perr or {"ok": True, "event_id": ev.event_id}  # type: ignore[union-attr]

    # ------------------------------------------------------------------ reading
    def describe_event(self, e: EvidenceEvent) -> str:
        head = f"[{e.event_id}] {e.agent_id} {e.event_kind.value}"
        if e.parent_event_ids:
            head += f" (parents: {', '.join(e.parent_event_ids)})"
        if e.hypothesis_status:
            head += f" status={e.hypothesis_status.value}"
        body = e.claim_or_request or ""
        extra = []
        if e.alternative_explanations:
            extra.append("alternatives: " + "; ".join(e.alternative_explanations))
        if e.limitations:
            extra.append("limitations: " + "; ".join(e.limitations))
        if e.tool_run_id:
            got = self.store.get_tool_run(e.tool_run_id)
            if got and got[0].unit:
                extra.append(f"unit: {got[0].unit}")
        return head + ": " + body + ((" | " + " | ".join(extra)) if extra else "")

    def list_events(
        self, agent_id: str, kinds: list[str] | None = None, since_event_id: str | None = None
    ) -> list[str]:
        since = self.store.seq_of(since_event_id) if since_event_id else 0
        out = []
        for _, e in self.store.events(since or 0):
            if kinds and e.event_kind.value not in kinds:
                continue
            out.append(self.describe_event(e))
        return out[-40:]

    def get_event_detail(self, event_id: str) -> dict[str, Any]:
        e = self.store.get_event(event_id)
        if e is None:
            return _err("UNKNOWN_EVENT", event_id)
        out: dict[str, Any] = {
            "ok": True,
            "text": self.describe_event(e),
            "parents": e.parent_event_ids,
        }
        if e.tool_run_id:
            got = self.store.get_tool_run(e.tool_run_id)
            if got:
                out["measurement"] = self._summary(got[0])
        return out

    # ------------------------------------------------------------ LLM accounting
    def llm_before_call(self, agent_id: str) -> None:
        if self._finished:
            raise BudgetExhausted("RUN_ENDED")
        if self.llm_calls >= self.limits.budget.max_llm_calls:
            raise BudgetExhausted("LLM_CALLS")
        self.llm_calls += 1

    def llm_record(
        self,
        agent_id: str,
        *,
        started: datetime,
        request: Any,
        response: Any,
        error: str | None,
        usage: dict[str, int | None],
    ) -> None:
        for k, name in (
            ("input", "input_token_count"),
            ("output", "output_token_count"),
            ("total", "total_token_count"),
        ):
            self.tokens[k] += int(usage.get(name) or 0)
        self.store.append_llm_call(
            investigation_id=self.investigation_id,
            agent_id=agent_id,
            provider=self.provider,
            model=self.model,
            started_utc=started.isoformat(),
            finished_utc=datetime.now(UTC).isoformat(),
            ok=0 if error else 1,
            error=error,
            input_tokens=usage.get("input_token_count"),
            output_tokens=usage.get("output_token_count"),
            total_tokens=usage.get("total_token_count"),
            request_json=json.dumps(request, default=str),
            response_json=None if response is None else json.dumps(response, default=str),
        )

    # ----------------------------------------------------------------- the run
    def _maybe_finish(self, reason: str = "QUIESCENT") -> None:
        if self._finished:
            return
        active = [s for s in self.state.values() if not s.exited]
        if all(not s.busy and not s.pending_first and not s.wake_pending for s in active):
            self._end(reason)

    def _end(self, reason: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._ended_reason = reason
        for s in self.state.values():
            s.wake.set()

    async def _agent_loop(
        self, spec: AgentSpec, runner: TurnRunner, build_prompt: Callable[..., str]
    ) -> None:
        st = self.state[spec.agent_id]
        first = not spec.starts_idle
        while True:
            if not first:
                await st.wake.wait()
            first = False
            st.wake.clear()
            if self._finished:
                st.exit_reason = st.exit_reason or "RUN_ENDED"
                break
            if st.rounds >= self.limits.max_rounds_per_agent:
                st.exit_reason = "MAX_ROUNDS"
                break
            if self.llm_calls >= self.limits.budget.max_llm_calls:
                st.exit_reason = "LLM_BUDGET"
                break
            new, st.inbox = st.inbox, []
            st.pending_first, st.wake_pending, st.busy = False, False, True
            prompt = build_prompt(self, spec, st, new)
            try:
                await self._turn_with_retry(spec, st, runner, prompt)
            finally:
                st.busy = False
                st.rounds += 1
            if st.exit_reason:
                break
            self._maybe_finish()
        st.exited, st.pending_first, st.wake_pending, st.busy = True, False, False, False
        self._maybe_finish()

    async def _turn_with_retry(
        self, spec: AgentSpec, st: AgentState, runner: TurnRunner, prompt: str
    ) -> None:
        for attempt in (1, 2):
            try:
                await runner.take_turn(prompt)
                return
            except BudgetExhausted as e:
                st.exit_reason = f"BUDGET_{e.kind}"
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:  # provider/tool failure: record authentically, retry once
                st.failures += 1
                self._publish(
                    spec.agent_id,
                    EventKind.FAILURE,
                    claim_or_request=_clip(
                        f"turn failed (attempt {attempt}): {type(e).__name__}: {e}", 600
                    ),
                    error_code="AGENT_TURN_ERROR",
                    limitations=[f"attempt {attempt} of 2"],
                )
                if attempt == 2:
                    st.exit_reason = "TURN_ERROR"
                    return

    async def run(
        self, runners: dict[str, TurnRunner], build_prompt: Callable[..., str]
    ) -> RunSummary:
        if set(runners) != set(self.specs):
            raise ValueError("one runner per agent spec is required")
        started = time.monotonic()
        tasks = [
            asyncio.create_task(
                self._agent_loop(s, runners[s.agent_id], build_prompt), name=s.agent_id
            )
            for s in self.specs.values()
        ]
        self._maybe_finish()  # nobody scheduled to start => nothing can ever happen
        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=self.limits.budget.timeout_s)
            if self._ended_reason == "RUNNING":
                self._end("QUIESCENT")
        except TimeoutError:
            self._end("TIMEOUT")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
        reason = self._ended_reason
        if reason == "QUIESCENT" and any(
            s.exit_reason and s.exit_reason.startswith("BUDGET_") for s in self.state.values()
        ):
            reason = "BUDGET_EXHAUSTED"
        self._publish(
            SUPERVISOR,
            EventKind.COMPLETION,
            claim_or_request=f"run ended: {reason}",
            limitations=[
                f"{a}: rounds={s.rounds} exit={s.exit_reason}" for a, s in self.state.items()
            ][:MAX_LIST],
            hypothesis_status=HypothesisStatus.UNRESOLVED,
        )
        return self.summary(reason, time.monotonic() - started)

    def summary(self, reason: str, elapsed: float) -> RunSummary:
        b = self.limits.budget
        return RunSummary(
            investigation_id=self.investigation_id,
            region_id=self.context.region_id,
            ended_reason=reason,
            elapsed_s=round(elapsed, 3),
            events=self.store.count_events(),
            tool_calls=self.tool_calls,
            llm_calls=self.llm_calls,
            input_tokens=self.tokens["input"],
            output_tokens=self.tokens["output"],
            total_tokens=self.tokens["total"],
            rounds={a: s.rounds for a, s in self.state.items()},
            exit_reasons={a: s.exit_reason for a, s in self.state.items()},
            budget={**b.model_dump(), "max_rounds_per_agent": self.limits.max_rounds_per_agent},
            problems=self.store.verify(),
        )

    def write_outputs(self, directory: Path, summary: RunSummary) -> None:
        """Write events.jsonl, tool_runs.json, llm_calls.jsonl and run_summary.json."""
        directory.mkdir(parents=True, exist_ok=True)
        self.store.export_jsonl(directory / "events.jsonl")
        (directory / "tool_runs.json").write_text(json.dumps(self.store.tool_runs(), indent=1))
        with (directory / "llm_calls.jsonl").open("w", encoding="utf-8") as f:
            for row in self.store.llm_calls():
                f.write(json.dumps(row) + "\n")
        (directory / "run_summary.json").write_text(
            json.dumps(summary.__dict__, indent=2, default=str)
        )


# ---------------------------------------------------------------------------- loading
class RegionNotInvestigable(RuntimeError):
    code = "REGION_NOT_INVESTIGABLE"


def load_paired_region(
    settings: Settings, region_id: str, budget: RunBudget
) -> tuple[AgentRegionContext, RegionArrays]:
    """Context + arrays for a verified PAIRED region only.

    Observation-only tiles raise ``PredictionUnavailable``; prediction-only or failed pairings raise
    ``RegionNotInvestigable``. Nothing from the evaluator side is reachable from here.
    """
    catalog = discover(settings)
    rec = catalog.get(region_id)  # raises for observation-only / unknown regions
    if rec.status != "PAIRED" or rec.entry.readiness.value != "verified":
        raise RegionNotInvestigable(
            f"{region_id} is {rec.status} (readiness {rec.entry.readiness.value}); "
            "P5 investigates only verified PAIRED regions"
        )
    arrays = catalog.load_arrays(region_id)
    return rec.entry.to_agent_context(available_tool_names(arrays), budget), arrays


AsyncTurn = Callable[[str], Awaitable[str]]
