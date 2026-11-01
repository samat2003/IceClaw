"""Evidence record of a targeted cross-modal window review (``iceclaw-window-review/1``).

Everything here is DERIVED from the append-only event store and the verified source arrays:
which specialist supplied each measurement, which layers were measured on the target window, which
requests were answered, which agent cited whose evidence, and whether every window-scoped
measurement reproduces on an independent recomputation. Specialists' own words (finding,
rationale, missing evidence, alternatives, follow-up) are carried verbatim and labelled as AI
hypotheses. A review with any missing individual review is INCOMPLETE; no verdict is substituted.
"""

from __future__ import annotations

import json
from typing import Any

from iceclaw.contracts import EventKind, EvidenceEvent
from iceclaw.science.common import RegionArrays, ScienceToolError
from iceclaw.science.model_audit import matches, recompute
from iceclaw.swarm.store import EvidenceStore

REVIEW_FORMAT = "iceclaw-window-review/1"
COMPLETE, INCOMPLETE = "COMPLETE", "INCOMPLETE"
REVIEW_PREFIX = "REVIEW "
LABEL = (
    "AI hypotheses: the findings below are language-model judgements about deterministic "
    "measurements; they are not ice detections, abundances or calibrated likelihoods."
)


def parse_review_text(text: str | None) -> tuple[dict[str, Any], str] | None:
    """``REVIEW {json} | rationale`` -> (record, rationale); None if not a review event."""
    if not text or not text.startswith(REVIEW_PREFIX):
        return None
    head, _, rationale = text[len(REVIEW_PREFIX) :].partition(" | ")
    try:
        rec = json.loads(head)
    except json.JSONDecodeError:
        return None
    return (rec, rationale.strip()) if isinstance(rec, dict) else None


def _layers(params: dict[str, Any]) -> list[str]:
    return [x for x in (params.get("modality"), params.get("shadow_modality")) if x]


def _is_window(params: dict[str, Any], window: list[int]) -> bool:
    w = params.get("window")
    return w is not None and list(w) == list(window)


def build_review_record(
    store: EvidenceStore,
    arrays: RegionArrays,
    *,
    arm: str,
    agent_ids: list[str],
    target: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the review record; ``target`` has label, window, centre_lat/lon and category."""
    window = list(target["window"])
    events: list[EvidenceEvent] = [e for _, e in store.events()]
    by_id = {e.event_id: e for e in events}
    supplier = {e.event_id: e.agent_id for e in events}

    evidence: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for e in events:
        if e.event_kind not in (EventKind.MEASUREMENT, EventKind.TEST_RESULT) or not e.tool_run_id:
            continue
        got = store.get_tool_run(e.tool_run_id)
        if got is None:
            continue
        m = got[0]
        runs[e.event_id] = m
        evidence.append(
            {
                "event_id": e.event_id,
                "supplier": e.agent_id,
                "kind": e.event_kind.value,
                "tool": m.tool_id,
                "layers": _layers(m.parameters),
                "window_scoped": _is_window(m.parameters, window),
                "is_recomputation": e.tool_run_id.endswith("#recompute"),
                "answers_request": e.parent_event_ids[0]
                if e.event_kind is EventKind.TEST_RESULT and e.parent_event_ids
                else None,
                "value": m.value,
                "unit": m.unit,
                "valid_pixels": m.valid_pixel_count,
                "parameters": {
                    k: v for k, v in m.parameters.items() if k != "window" and v is not None
                },
            }
        )

    # latest REVIEW event per agent
    reviews: dict[str, dict[str, Any]] = {}
    for e in events:
        parsed = parse_review_text(e.claim_or_request)
        if parsed is None or e.agent_id not in agent_ids:
            continue
        rec, rationale = parsed
        cited = [p for p in e.parent_event_ids if p in runs]
        reviews[e.agent_id] = {
            "event_id": e.event_id,
            "finding": rec.get("finding"),
            "rationale": rationale,
            "missing_evidence": rec.get("missing_evidence", []),
            "follow_up": rec.get("follow_up", []),
            "alternatives": list(e.alternative_explanations),
            "cited_evidence": cited,
            "cited_suppliers": sorted({supplier[c] for c in cited}),
            "cites_window_scoped": any(
                x["window_scoped"] for x in evidence if x["event_id"] in cited
            ),
            "cites_other_agents": sorted({supplier[c] for c in cited} - {e.agent_id}),
        }

    requests = []
    for e in events:
        if e.event_kind is not EventKind.TEST_REQUEST:
            continue
        try:
            body = json.loads(e.claim_or_request or "{}")
        except json.JSONDecodeError:
            body = {}
        results = [
            r.event_id
            for r in events
            if r.event_kind is EventKind.TEST_RESULT and e.event_id in r.parent_event_ids
        ]
        requests.append(
            {
                "event_id": e.event_id,
                "from": e.agent_id,
                "to": body.get("to"),
                "tool": body.get("tool_id"),
                "result_event_ids": results,
                "answered": bool(results),
            }
        )
    challenges = [
        {"event_id": e.event_id, "from": e.agent_id, "targets": e.parent_event_ids[:1]}
        for e in events
        if e.event_kind is EventKind.CHALLENGE
    ]
    revisions = [e.event_id for e in events if e.event_kind is EventKind.REVISION]
    cross = [
        {"event_id": e.event_id, "agent": e.agent_id, "cites": p, "supplier": supplier[p]}
        for e in events
        for p in e.parent_event_ids
        if p in supplier and supplier[p] != e.agent_id and e.agent_id in agent_ids
    ]

    # numerical re-verification of every recorded measurement (independent recomputation)
    checked, reproduced, failures = 0, 0, []
    for e in events:
        m = runs.get(e.event_id)
        if m is None or e.tool_run_id is None or e.tool_run_id.endswith("#recompute"):
            continue
        checked += 1
        try:
            ok = matches(m, recompute(m, arrays))
        except ScienceToolError:
            ok = False
        reproduced += ok
        if not ok:
            failures.append(e.event_id)

    layer_suppliers: dict[str, list[str]] = {}
    for x in evidence:
        if x["window_scoped"] and not x["is_recomputation"]:
            for lay in x["layers"]:
                layer_suppliers.setdefault(lay, [])
                if x["supplier"] not in layer_suppliers[lay]:
                    layer_suppliers[lay].append(x["supplier"])

    missing = [a for a in agent_ids if a not in reviews]
    unscoped = [a for a, r in reviews.items() if not r["cites_window_scoped"]]
    reasons = [f"no review filed by {a}" for a in missing] + [
        f"{a}'s review cites no measurement made on the target window" for a in unscoped
    ]
    findings = {a: r["finding"] for a, r in reviews.items()}
    vals = set(findings.values())
    agreement = "no_reviews" if not vals else (next(iter(vals)) if len(vals) == 1 else "mixed")
    all_missing_evidence = [
        {"agent": a, "text": t} for a, r in reviews.items() for t in r["missing_evidence"]
    ]
    all_follow_up = [{"agent": a, "text": t} for a, r in reviews.items() for t in r["follow_up"]]
    competing = [{"agent": a, "text": t} for a, r in reviews.items() for t in r["alternatives"]]
    usage = {
        k: run.get(k)
        for k in (
            "llm_calls",
            "tool_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "elapsed_s",
            "events",
            "ended_reason",
        )
    }
    metrics = {
        "review_complete": not reasons,
        "n_agents": len(agent_ids),
        "n_reviews_filed": len(reviews),
        "n_incomplete_reviews": len(missing),
        "n_window_scoped_measurements": sum(
            1 for x in evidence if x["window_scoped"] and not x["is_recomputation"]
        ),
        "n_layers_measured_on_window": len(layer_suppliers),
        "n_test_requests": len(requests),
        "n_test_requests_answered": sum(r["answered"] for r in requests),
        "n_challenges": len(challenges),
        "n_revisions": len(revisions),
        "n_cross_agent_citations": len(cross),
        "n_measurements_checked": checked,
        "n_measurements_reproduced": reproduced,
        "numerical_verification_success": (reproduced == checked) if checked else None,
        "usage": usage,
    }
    return {
        "format": REVIEW_FORMAT,
        "label": LABEL,
        "arm": arm,
        "target": target,
        "status": INCOMPLETE if reasons else COMPLETE,
        "incomplete_reasons": reasons,
        "agent_findings": findings,
        "agreement": agreement,
        "reviews": reviews,
        "evidence": evidence,
        "layers_measured_on_window": layer_suppliers,
        "requests": requests,
        "challenges": challenges,
        "revisions": revisions,
        "cross_agent_citations": cross,
        "missing_evidence": all_missing_evidence,
        "competing_explanations": competing,
        "suggested_follow_up": all_follow_up,
        "numerical_verification": {
            "checked": checked,
            "reproduced": reproduced,
            "failed_event_ids": failures,
        },
        "metrics": metrics,
        "n_events": len(by_id),
    }
