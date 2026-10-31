"""Export an investigation into a reproducible, independently verifiable bundle.

Layout (``runs/<id>/``): the six contract files ``input_manifest.json``, ``events.jsonl``,
``evidence.json``, ``report.md``, ``rankings.csv``, ``verification.json``; plus
``llm_calls.jsonl`` (the retained model transcript with provider-reported usage) and ``source/``
(the minimal permitted source package, see ``reports/package.py``).

Design rule: ``evidence.json``, ``rankings.csv`` and ``report.md`` are PURE FUNCTIONS of
(events, recorded tool measurements, package geometry/manifest). The verifier re-derives them and
compares, so any edit that is not reproduced from the events and measurements is detected. Nothing
here invents activity: events and measurements are exported exactly as recorded.
"""

from __future__ import annotations

import csv
import io
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iceclaw.contracts import EventKind, EvidenceEvent, InputManifest, RunBudget, SourceMode
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.reports import package as pkg
from iceclaw.swarm.store import EvidenceStore

# --- export contract versions -------------------------------------------------------------
# iceclaw-bundle/1 + iceclaw-derive/1: the original P6 bundle (full transcript only; rankings.csv
#   without tie columns; report section 10 without the tie/attention wording). Still verified,
#   with the original logic, so existing bundles stay valid.
# iceclaw-bundle/2 + iceclaw-derive/2 (written by this version): adds `transcript_profile`
#   (full-private | public-redacted) and `transcript_redactions` to the manifest, tie columns in
#   rankings.csv, and the explicit "investigation attention, not an ice-prospectivity ranking"
#   wording in the report. The verifier re-derives evidence/rankings/report, so a derivation change
#   must bump DERIVATION_VERSION and keep the old logic reachable (see ``derive_files``).
BUNDLE_FORMAT_V1 = "iceclaw-bundle/1"
BUNDLE_FORMAT_V2 = "iceclaw-bundle/2"
BUNDLE_FORMAT = BUNDLE_FORMAT_V2
SUPPORTED_BUNDLE_FORMATS = (BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2)
DERIVATION_V1 = "iceclaw-derive/1"
DERIVATION_V2 = "iceclaw-derive/2"
DERIVATION_VERSION = DERIVATION_V2
SUPPORTED_DERIVATIONS = (DERIVATION_V1, DERIVATION_V2)
FORMAT_DERIVATION = {BUNDLE_FORMAT_V1: DERIVATION_V1, BUNDLE_FORMAT_V2: DERIVATION_V2}
PROFILE_FULL = "full-private"
PROFILE_PUBLIC = "public-redacted"
OPAQUE_KEYS = ("protected_data", "encrypted_content")  # provider-opaque reasoning payloads
REDACTED_MARKER = "[redacted: opaque provider reasoning data]"
CONTRACT_FILES = (
    "input_manifest.json",
    "events.jsonl",
    "evidence.json",
    "report.md",
    "rankings.csv",
    "verification.json",
)
INTEGRITY_FILES = ("events.jsonl", "evidence.json", "report.md", "rankings.csv", "llm_calls.jsonl")

CLASS_PRED = "calculation_on_nasa_ibm_model_output"
CLASS_OBS = "calculation_on_lunar_observation"
LABELS = {
    "nasa_ibm_model_prediction": "NASA-IBM MODEL PREDICTION (a prospectivity index from a "
    "regression model; not an observation, not a calibrated probability, not ice detection)",
    "lunar_observation": "LUNAR OBSERVATION DATA (SomBench-distributed instrument-derived layers)",
    CLASS_PRED: "DETERMINISTIC CALCULATION on NASA-IBM model output",
    CLASS_OBS: "DETERMINISTIC CALCULATION on lunar observation data",
    "ai_hypothesis": "AI HYPOTHESIS / interpretation by a language-model agent (not a measurement)",
    "runtime_note": "RUNTIME NOTE (neutral supervisor record)",
}
PIXEL_INDEX = "pixel index (relative to window origin)"
DISCLAIMER = (
    "**Research prioritization only.** The NASA-IBM quantity analysed here is a *prospectivity "
    "index*, not a measured ice abundance, not a calibrated probability and not a detection. "
    "Nothing in this report detects, confirms or discovers lunar ice."
)
LIMITATIONS = [
    "The NASA-IBM prediction is the model output published in the MOONATLAS science-smoke "
    "example (source_mode upstream_published_example); it is NOT a fresh IceClaw inference and "
    "the checkpoint hash is the upstream-published value, not verified locally.",
    "One region only (PARTIAL_ONE_REGION): the other south test tiles have observations but no "
    "verified prediction.",
    "Observation units for TMAX (K), LPSR (fraction) and SLOPE (deg) rest on upstream naming "
    "and anchor-pixel agreement; the GeoTIFFs carry no unit tags.",
    "Thresholds passed to tools (e.g. cold-trap K, steep-slope degrees) are analyst-chosen "
    "screening parameters, not validated physical constants.",
    "The prediction and the observation layers share upstream modelling assumptions; agreement "
    "between them is not independent validation.",
    "Agent text is language-model output. It can be re-checked against the cited deterministic "
    "measurements but its authenticity (that a model wrote it) cannot be proven offline.",
]
FIXED_LAYERS = {
    "thermal_psr_consistency": ["pixelwise_temperature", "pixelwise_shadow_mask"],
    "terrain_slope_summary": ["pixelwise_terrain"],
}


# ================================================================================ run record
@dataclass
class RunRecord:
    investigation_id: str
    region_id: str
    events: list[EvidenceEvent]
    measurements: dict[str, dict[str, Any]]  # tool_run_id -> {event_id, agent_id, record}
    llm_calls: list[dict[str, Any]]
    summary: dict[str, Any]
    mode: str  # "live" | "recorded-export"
    provider: str
    model: str
    code_revision: str
    notes: list[str] = field(default_factory=list)


def current_code_revision(root: Path | None = None) -> str:
    """``<git sha>`` or ``<git sha>+dirty`` (uncommitted changes) or ``unknown``."""
    try:
        cwd = str(root) if root else None
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=cwd
        ).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record_from_store(
    store: EvidenceStore,
    *,
    summary: dict[str, Any],
    mode: str,
    provider: str,
    model: str,
    code_revision: str,
    notes: Iterable[str] = (),
) -> RunRecord:
    meas = {
        r["tool_run_id"]: {
            "event_id": r["event_id"],
            "agent_id": r["agent_id"],
            "record": r["payload"],
        }
        for r in store.tool_runs()
    }
    events = list(store)
    return RunRecord(
        investigation_id=summary["investigation_id"],
        region_id=summary["region_id"],
        events=events,
        measurements=meas,
        llm_calls=store.llm_calls(),
        summary=summary,
        mode=mode,
        provider=provider,
        model=model,
        code_revision=code_revision,
        notes=list(notes),
    )


def find_run_files(run_dir: Path) -> tuple[Path, Path]:
    """(events.sqlite, run_summary.json) for a P5-style run dir or a P6 run's internal/ dir."""
    for base in (run_dir, run_dir / "internal"):
        db, js = base / "events.sqlite", base / "run_summary.json"
        if db.is_file() and js.is_file():
            return db, js
    raise FileNotFoundError(f"no events.sqlite + run_summary.json in {run_dir}")


def record_from_run_dir(
    run_dir: Path, *, mode: str = "recorded-export", notes: Iterable[str] = ()
) -> RunRecord:
    """Read a recorded run's SQLite history WITHOUT modifying it (opened read-only)."""
    db, js = find_run_files(Path(run_dir))
    summary = json.loads(js.read_text())
    ro = EvidenceStore.open_readonly(db)
    rows = ro.llm_calls()
    provider = rows[0]["provider"] if rows else summary.get("provider", "unknown")
    model = rows[0]["model"] if rows else summary.get("model", "unknown")
    return record_from_store(
        ro,
        summary={**summary, "provider": provider, "model": model},
        mode=mode,
        provider=provider,
        model=model,
        code_revision=current_code_revision(),
        notes=notes,
    )


# =================================================================================== evidence
def layers_of(rec: dict[str, Any]) -> list[str]:
    if rec["tool_id"] in FIXED_LAYERS:
        return list(FIXED_LAYERS[rec["tool_id"]])
    m = rec["parameters"].get("modality")
    return [m] if m else []


def evidence_class(rec: dict[str, Any]) -> str:
    obs = any(str(f).startswith("sombench:") for f in rec["source_file_ids"])
    return CLASS_OBS if obs else CLASS_PRED


def field_unit(tool: str, key: str, unit: str | None) -> str:
    """Correct unit for ONE field of a (possibly mixed-unit) measurement dictionary."""
    unit = unit or "unknown"
    if tool in ("valid_pixel_summary", "quantiles", "boundary_check", "gradient_stats"):
        return unit
    if tool in ("missingness", "unit_interval_check"):
        return "pixels"
    if tool == "discontinuity_candidates":
        m = re.search(r"\*_gradient in (\S+)", unit)
        grad = m.group(1) if m else "unknown"
        if key.endswith("_gradient"):
            return grad
        if key.endswith("_count"):
            return "pixels"
        return PIXEL_INDEX
    if tool == "peak_location":
        return "pixels" if key == "edge_distance_px" else PIXEL_INDEX
    if tool == "thermal_psr_consistency":
        return (
            "K" if key.endswith("_k") else "pixels" if key.endswith("_count") else "dimensionless"
        )
    if tool == "terrain_slope_summary":
        return "deg" if key.endswith("_deg") else "dimensionless"
    return unit


def field_table(rec: dict[str, Any]) -> list[dict[str, Any]]:
    v = rec["value"]
    if v is None:
        return []
    items = v.items() if isinstance(v, dict) else [("value", v)]
    return [
        {"name": k, "value": x, "unit": field_unit(rec["tool_id"], k, rec.get("unit"))}
        for k, x in items
    ]


def measurement_entry(e: EvidenceEvent, m: dict[str, Any]) -> dict[str, Any]:
    r = m["record"]
    cls = evidence_class(r)
    return {
        "event_id": e.event_id,
        "tool_run_id": r["tool_run_id"],
        "agent_id": e.agent_id,
        "event_kind": e.event_kind.value,
        "tool_id": r["tool_id"],
        "evidence_class": cls,
        "evidence_class_label": LABELS[cls],
        "layers": layers_of(r),
        "parameters": {
            k: v for k, v in r["parameters"].items() if v is not None and k != "unavailable_detail"
        },
        "source_file_ids": r["source_file_ids"],
        "source_sha256": r["source_sha256"],
        "resolution_m": r["resolution_m"],
        "valid_pixel_count": r["valid_pixel_count"],
        "unavailable_reason": r["unavailable_reason"],
        "reproducibility_tolerance_relative": r["tolerance"],
        "implementation_version": r["implementation_version"],
        "footprint_lat_lon": r["footprint"],
        "fields": field_table(r),
        "record": r,
    }


_NUM = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?![\w])")


def _floats(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _floats(x)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _floats(x)


def check_numbers(text: str, allowed: list[float]) -> dict[str, Any]:
    """Which decimal / >=3-digit numbers in ``text`` are found among ``allowed`` values.

    A token matches if it equals an allowed value rounded to the token's own decimals (honest
    rounding), or (before a '%') the value x100. Short integers (< 100) are ignored.
    """
    clean = re.sub(r"evt-\d+", " ", text)
    clean = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", clean)  # 65,536 -> 65536
    checked, unmatched = 0, []
    for m in _NUM.finditer(clean):
        raw = m.group(0).replace("−", "-")
        digits = raw.lstrip("-")
        has_dec = "." in digits
        if not has_dec and len(digits) < 3:
            continue
        t = float(raw)
        d = len(digits.split(".")[1]) if has_dec else 0
        tol = 0.5 * 10**-d * 1.0001
        pct = clean[m.end() : m.end() + 1] == "%"
        cands = [a * 100 if pct else a for a in allowed] + [-a for a in allowed]
        checked += 1
        if not any(abs(a - t) <= tol + 1e-12 * abs(a) for a in cands):
            unmatched.append(raw)
    return {"checked": checked, "matched": checked - len(unmatched), "unmatched": unmatched}


def _closure(
    event_id: str, ev: dict[str, EvidenceEvent], evidence_ids: set[str], order: dict[str, int]
):
    seen: set[str] = set()
    stack = [event_id]
    while stack:
        cur = stack.pop()
        node = ev.get(cur)
        if node is None:  # dangling parent (tampered log): the verifier reports it separately
            continue
        for p in node.parent_event_ids:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return sorted((p for p in seen if p in evidence_ids), key=lambda x: order[x])


NUMERIC_KINDS = (
    EventKind.CLAIM,
    EventKind.REVISION,
    EventKind.CHALLENGE,
    EventKind.COMPLETION,
)


def request_text(kind: str, text: str) -> str:
    """Readable form of a test_request's JSON body (other kinds pass through unchanged)."""
    if kind != "test_request":
        return text
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(d, dict) or "tool" not in d:
        return text
    params = ", ".join(f"{k}={v}" for k, v in sorted((d.get("params") or {}).items()))
    return (
        f"asks {d.get('to')} to run {d.get('tool')}"
        + (f" on {d['modality']}" if d.get("modality") else "")
        + (f" with {params}" if params else "")
        + (f" in window {d['window']}" if d.get("window") else "")
        + f". Rationale: {d.get('rationale', '')}"
    )


STATEMENT_KINDS = (
    EventKind.CLAIM,
    EventKind.REVISION,
    EventKind.CHALLENGE,
    EventKind.COMPLETION,
    EventKind.TEST_REQUEST,
    EventKind.FAILURE,
)


def build_evidence(rec: RunRecord) -> dict[str, Any]:
    ev = {e.event_id: e for e in rec.events}
    order = {e.event_id: i for i, e in enumerate(rec.events)}
    by_event = {m["event_id"]: m for m in rec.measurements.values()}
    measurements = [
        measurement_entry(e, by_event[e.event_id]) for e in rec.events if e.event_id in by_event
    ]
    m_by_id = {m["event_id"]: m for m in measurements}
    evidence_ids = set(m_by_id)
    statements = []
    for e in rec.events:
        if e.event_kind not in STATEMENT_KINDS:
            continue
        runtime = e.agent_id == "supervisor"
        cited = _closure(e.event_id, ev, evidence_ids, order)
        allowed: list[float] = []
        for c in cited:
            allowed += list(_floats([f["value"] for f in m_by_id[c]["fields"]]))
            allowed += list(_floats(m_by_id[c]["parameters"]))
            allowed += list(_floats([m_by_id[c]["valid_pixel_count"], m_by_id[c]["resolution_m"]]))
        text = e.claim_or_request or ""
        statements.append(
            {
                "event_id": e.event_id,
                "agent_id": e.agent_id,
                "kind": e.event_kind.value,
                "epistemic_label": LABELS["runtime_note" if runtime else "ai_hypothesis"],
                "text": text,
                "status": e.hypothesis_status.value if e.hypothesis_status else None,
                "parent_event_ids": e.parent_event_ids,
                "cited_evidence_event_ids": cited,
                "cited_source_file_ids": sorted(
                    {f for c in cited for f in m_by_id[c]["source_file_ids"]}
                ),
                "alternative_explanations": e.alternative_explanations,
                "limitations": e.limitations,
                # only interpretive statements are number-checked; a test_request's numbers are
                # REQUESTED parameters (inputs), not claims
                "numeric_check": check_numbers(text, allowed)
                if (not runtime and e.event_kind in NUMERIC_KINDS)
                else None,
            }
        )
    exchange = [
        {
            "event_id": e.event_id,
            "agent_id": e.agent_id,
            "kind": e.event_kind.value,
            "parent_event_ids": e.parent_event_ids,
            "tool_id": e.tool_id,
            "text": (e.claim_or_request or "")
            if e.event_kind is EventKind.TEST_REQUEST
            else (e.claim_or_request or "")[:400],
        }
        for e in rec.events
        if e.event_kind
        in (EventKind.CHALLENGE, EventKind.TEST_REQUEST, EventKind.TEST_RESULT, EventKind.REVISION)
    ]
    kinds: dict[str, int] = {}
    agents: dict[str, int] = {}
    for e in rec.events:
        kinds[e.event_kind.value] = kinds.get(e.event_kind.value, 0) + 1
        agents[e.agent_id] = agents.get(e.agent_id, 0) + 1
    return {
        "format": "iceclaw-evidence/1",
        "epistemic_labels": LABELS,
        "measurements": measurements,
        "statements": statements,
        "exchange": exchange,
        "summary": {
            "events_by_kind": dict(sorted(kinds.items())),
            "events_by_agent": dict(sorted(agents.items())),
            "n_measurements": len(measurements),
            "n_statements": len(statements),
            "challenges": kinds.get("challenge", 0),
            "revisions": kinds.get("revision", 0),
            "test_requests": kinds.get("test_request", 0),
            "no_challenge_or_revision": not (kinds.get("challenge") or kinds.get("revision")),
        },
    }


# =================================================================================== rankings
RANK_NOTE_V1 = (
    "research follow-up priority from a deterministic coverage rule (how many distinct agents' "
    "evidence covers or names the location); NOT a physical probability of ice"
)
RANK_NOTE_V2 = (
    "investigation-attention order: how many distinct agents' evidence covers or names the "
    "location. NOT an independently validated ice-prospectivity ranking and NOT a physical "
    "probability of ice; tied targets are listed alphabetically by target_id and the order "
    "among them carries no importance"
)
RANK_NOTE = RANK_NOTE_V2
TIE_BREAK_TEXT = "alphabetical by target_id (no importance implied)"
RANK_COLUMNS_V1 = [
    "rank",
    "target_id",
    "kind",
    "layer",
    "row",
    "col",
    "window",
    "latitude",
    "longitude",
    "n_distinct_agents",
    "n_evidence_events",
    "evidence_event_ids",
    "basis",
    "note",
]
RANK_COLUMNS_V2 = [*RANK_COLUMNS_V1, "tie_group_size", "tie_break"]
RANK_COLUMNS = RANK_COLUMNS_V2


def _window_offset(entry: dict[str, Any]) -> tuple[int, int]:
    w = entry["parameters"].get("window")
    return (int(w[0]), int(w[2])) if w else (0, 0)


def build_rankings(
    evidence: dict[str, Any], projection: dict[str, Any], grid: dict[str, Any], version: int = 2
) -> list[dict[str, Any]]:
    """Targets named by the recorded measurements, ranked by cross-agent coverage.

    Rule: a target (pixel or window) is COVERED by every measurement that reports it, whose
    window contains it, or that reports a point within 8 px. Rank by distinct covering agents,
    then covering events, then target id. This is an INVESTIGATION-ATTENTION order: it counts how
    much independent scrutiny a location received. It is not an independently validated
    ice-prospectivity ranking, not physical support for ice, and no score is a probability of ice.
    Targets with identical (agents, events) counts are TIED and appear alphabetically by target
    id, which carries no importance. ``version`` 1 reproduces the original P6 derivation (no tie
    columns, original note); version 2 adds ``tie_group_size`` / ``tie_break`` and the v2 note.
    """
    ms = [m for m in evidence["measurements"] if m["unavailable_reason"] is None]
    targets: dict[str, dict[str, Any]] = {}

    def add(tid, kind, layer, row, col, window, event_id):
        t = targets.setdefault(
            tid,
            {"kind": kind, "layer": layer, "row": row, "col": col, "window": window, "gen": set()},
        )
        t["gen"].add(event_id)

    for m in ms:
        oy, ox = _window_offset(m)
        layer = ",".join(m["layers"])
        f = {x["name"]: x["value"] for x in m["fields"]}
        w = m["parameters"].get("window")
        if w:
            add(
                f"window:{w[0]}-{w[1]}:{w[2]}-{w[3]}",
                "examined window",
                layer,
                None,
                None,
                list(w),
                m["event_id"],
            )
        if m["tool_id"] == "peak_location":
            r, c = int(f["row"]) + oy, int(f["col"]) + ox
            add(f"peak:{layer}:{r}:{c}", "raw maximum", layer, r, c, None, m["event_id"])
        if m["tool_id"] == "discontinuity_candidates":
            i = 0
            while f"cand{i}_row" in f:
                r, c = int(f[f"cand{i}_row"]) + oy, int(f[f"cand{i}_col"]) + ox
                add(
                    f"discontinuity:{layer}:{r}:{c}",
                    "gradient discontinuity candidate",
                    layer,
                    r,
                    c,
                    None,
                    m["event_id"],
                )
                i += 1

    def center(t):
        if t["window"]:
            w = t["window"]
            return (w[0] + w[1]) / 2.0, (w[2] + w[3]) / 2.0
        return t["row"] + 0.5, t["col"] + 0.5

    rows = []
    for tid, t in targets.items():
        r0, c0 = center(t)
        support = set(t["gen"])
        for m in ms:
            oy, ox = _window_offset(m)
            w = m["parameters"].get("window")
            if w and w[0] <= r0 < w[1] and w[2] <= c0 < w[3]:
                support.add(m["event_id"])
            f = {x["name"]: x["value"] for x in m["fields"]}
            pts = []
            if m["tool_id"] == "peak_location":
                pts.append((f["row"] + oy + 0.5, f["col"] + ox + 0.5))
            if m["tool_id"] == "discontinuity_candidates":
                i = 0
                while f"cand{i}_row" in f:
                    pts.append((f[f"cand{i}_row"] + oy + 0.5, f[f"cand{i}_col"] + ox + 0.5))
                    i += 1
            if any(abs(pr - r0) <= 8 and abs(pc - c0) <= 8 for pr, pc in pts):
                support.add(m["event_id"])
        agents = sorted({m["agent_id"] for m in ms if m["event_id"] in support})
        lat, lon = ma.pixel_to_lat_lon(projection, grid, c0, r0)
        rows.append(
            {
                "target_id": tid,
                "kind": t["kind"],
                "layer": t["layer"],
                "row": "" if t["row"] is None else t["row"],
                "col": "" if t["col"] is None else t["col"],
                "window": ""
                if not t["window"]
                else "{}:{},{}:{}".format(*t["window"][:2], *t["window"][2:]),
                "latitude": f"{lat:.6f}",
                "longitude": f"{lon:.6f}",
                "n_distinct_agents": len(agents),
                "n_evidence_events": len(support),
                "evidence_event_ids": ";".join(sorted(support)),
                "basis": f"covered or named by agents: {', '.join(agents)}",
                "note": RANK_NOTE_V1 if version == 1 else RANK_NOTE_V2,
            }
        )
    rows.sort(key=lambda r: (-r["n_distinct_agents"], -r["n_evidence_events"], r["target_id"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    if version >= 2:
        sizes: dict[tuple[int, int], int] = {}
        for r in rows:
            key = (r["n_distinct_agents"], r["n_evidence_events"])
            sizes[key] = sizes.get(key, 0) + 1
        for r in rows:
            n = sizes[(r["n_distinct_agents"], r["n_evidence_events"])]
            r["tie_group_size"] = n
            r["tie_break"] = TIE_BREAK_TEXT if n > 1 else ""
    return rows


def rankings_csv(rows: list[dict[str, Any]], version: int = 2) -> str:
    cols = RANK_COLUMNS_V1 if version == 1 else RANK_COLUMNS_V2
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in cols})
    return buf.getvalue()


# ===================================================================================== report
def _v(x: Any) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _quote(text: str) -> str:
    return " ".join(text.split())


def _fields(m: dict[str, Any]) -> dict[str, Any]:
    return {f["name"]: f["value"] for f in m["fields"]}


def _bullet(m: dict[str, Any], geo: dict[str, Any]) -> str | None:
    """Deterministic one-sentence reading of a prediction-layer measurement (no interpretation)."""
    f, t, ev = _fields(m), m["tool_id"], m["event_id"]
    w = m["parameters"].get("window")
    where = f" in window {w}" if w else ""
    if t == "unit_interval_check":
        return (
            f"{_v(f['outside_unit_interval_pixels'])} of {m['valid_pixel_count']} valid pixels{where} have raw "
            f"values outside [0, 1] ({_v(f['below_zero_pixels'])} below 0, {_v(f['above_one_pixels'])} above 1); "
            f"raw regression values are preserved, not clipped. ({ev})"
        )
    if t == "valid_pixel_summary":
        return (
            f"raw values{where} range {_v(f['min'])} to {_v(f['max'])} (mean {_v(f['mean'])}, "
            f"median {_v(f['median'])}) [{m['record']['unit']}]. ({ev})"
        )
    if t == "missingness":
        return (
            f"{_v(f['valid_pixels'])} of {_v(f['total_pixels'])} pixels{where} are valid; "
            f"{_v(f['uncovered_pixels'])} uncovered, {_v(f['nonfinite_pixels'])} non-finite. ({ev})"
        )
    if t == "peak_location":
        oy, ox = _window_offset(m)
        r, c = int(f["row"]) + oy, int(f["col"]) + ox
        lat, lon = ma.pixel_to_lat_lon(geo["projection"], geo["grid"], c + 0.5, r + 0.5)
        return (
            f"the raw maximum{where} is at row {r}, col {c} (pixel centre {lat:.4f} deg, {lon:.4f} deg), "
            f"{_v(f['edge_distance_px'])} px from the border. ({ev})"
        )
    if t == "boundary_check":
        return (
            f"mean in the outer {m['parameters'].get('edge_width_px')}-px band is {_v(f['edge_mean'])} versus "
            f"{_v(f['interior_mean'])} in the interior (difference {_v(f['edge_minus_interior'])}); a screening "
            f"statistic, not proof of an edge artifact. ({ev})"
        )
    if t == "gradient_stats":
        return (
            f"gradient magnitude{where}: mean {_v(f['mean'])}, p95 {_v(f['p95'])}, max {_v(f['max'])} "
            f"[{m['record']['unit']}]. ({ev})"
        )
    if t == "discontinuity_candidates":
        return (
            f"{_v(f['n_candidates_count'])} pixels{where} exceed the robust gradient threshold "
            f"{_v(f['threshold_gradient'])} (k_mad {m['parameters'].get('k_mad')}); these are screening "
            f"candidates, not confirmed seams. ({ev})"
        )
    if t == "quantiles":
        qs = ", ".join(f"{k}={_v(v)}" for k, v in f.items())
        return f"quantiles{where}: {qs}. ({ev})"
    return None


def render_report(
    manifest: dict[str, Any],
    evidence: dict[str, Any],
    rankings: list[dict[str, Any]],
    geo: dict[str, Any],
) -> str:
    L: list[str] = []
    a = L.append
    pkgm = manifest["source_package"]
    mu = manifest["model_usage"]
    mode = manifest["mode"]
    a(f"# IceClaw research report: {manifest['region_id']}")
    a("")
    a("> " + DISCLAIMER)
    a("")
    mode_text = {
        "live": "**LIVE-GENERATED bundle** (the agents ran with real model calls)",
        "recorded-export": "**RECORDED RUN, exported from a saved history** (no model call was "
        "made to produce this bundle)",
    }.get(mode, "**SCRIPTED TEST RUN** (a fake model drove the agents; NOT a real model's output)")
    a(f"- Investigation `{manifest['investigation_id']}`; mode: {mode_text}")
    a(
        f"- Agents: four specialists on Microsoft Agent Framework, model `{manifest['llm_provider']}/{manifest['llm_model']}`"
    )
    a(
        f"- Code revision: `{manifest['code_revision']}`; bundle format `{manifest['bundle_format']}`"
    )
    a(
        f"- Model usage (provider-reported): {mu['calls']} calls, {mu['input_tokens']} input / "
        f"{mu['output_tokens']} output / {mu['total_tokens']} total tokens; cost: **{mu['cost']['status']}** "
        "(no verified model rate configured)"
    )
    a("")
    a("## 1. Question")
    a("")
    a(
        "Does the NASA-IBM ice-prospectivity prediction for this tile look internally ordinary and "
        "consistent with the lunar observation layers (temperature, permanent shadow, slope), what "
        "alternative explanations remain, and where should follow-up work look first? This is a "
        "research-prioritization question, not a search for ice."
    )
    a("")
    a("## 2. How to read this report")
    a("")
    a("Every statement carries one of four epistemic labels:")
    a("")
    for k in (
        "nasa_ibm_model_prediction",
        "lunar_observation",
        CLASS_PRED,
        CLASS_OBS,
        "ai_hypothesis",
    ):
        a(f"- **{k}**: {LABELS[k]}")
    a("")
    a("## 3. Available inputs")
    a("")
    a("| Layer | Kind | Unit | Source file id | SHA256 |")
    a("| --- | --- | --- | --- | --- |")
    for fid, f in sorted(pkgm["files"].items(), key=lambda kv: kv[1]["modality"]):
        label = (
            "NASA-IBM model prediction"
            if f["kind"] == "nasa_ibm_model_prediction"
            else "lunar observation"
        )
        a(f"| {f['modality']} | {label} | {f['unit']} | `{fid}` | `{f['sha256'][:16]}...` |")
    a("")
    a(
        f"Source mode: `{manifest['source_mode']}`; dataset `{manifest['source_dataset']}` "
        f"(split `{manifest['source_split']}`, revision `{manifest['source_revision'][:12]}`); checkpoint "
        f"`{manifest['prediction_checkpoint_id'].split('@')[0]}` (hash is the upstream-published value). "
        "Tile-level scalars and single-anchor metadata are not arrays and were not used as arrays. "
        "Layers with unestablished units were excluded from every tool."
    )
    a("")
    a("## 4. Observed numeric measurements (deterministic tools)")
    a("")
    for cls, title in (
        (CLASS_OBS, "4a. Calculations on lunar observation data"),
        (CLASS_PRED, "4b. Calculations on NASA-IBM model output"),
    ):
        a(f"### {title}")
        a("")
        subset = [m for m in evidence["measurements"] if m["evidence_class"] == cls]
        if not subset:
            a("_No measurement of this class was recorded._")
            a("")
        for m in subset:
            params = ", ".join(
                f"{k}={v}" for k, v in sorted(m["parameters"].items()) if k != "shadow_modality"
            )
            a(f"#### {m['event_id']} - {m['agent_id']} - `{m['tool_id']}`")
            a(
                f"{m['evidence_class_label']}; layers: {', '.join(m['layers'])}; parameters: {params}"
            )
            if m["unavailable_reason"]:
                a(f"Result: **unavailable** ({m['unavailable_reason']}). No value is guessed.")
                a("")
                continue
            a(
                f"Valid pixels: {m['valid_pixel_count']}; resolution {_v(m['resolution_m'])} m; "
                f"reproducibility tolerance {m['reproducibility_tolerance_relative']:g} (relative); "
                f"sources: {', '.join(f'`{s}`' for s in m['source_file_ids'])}"
            )
            a("")
            a("| Field | Value | Unit |")
            a("| --- | --- | --- |")
            for f in m["fields"]:
                a(f"| {f['name']} | {_v(f['value'])} | {f['unit']} |")
            a("")
    a("## 5. Model-prediction observations and possible inconsistencies")
    a("")
    a(
        "Screening readings of the recorded calculations on the NASA-IBM output (no interpretation added):"
    )
    a("")
    bullets = []
    for m in evidence["measurements"]:
        if m["evidence_class"] == CLASS_PRED and not m["unavailable_reason"]:
            b = _bullet(m, geo)
            if b:
                bullets.append(b)
    for b in bullets or ["_No calculation on the model output was recorded._"]:
        a(f"- {b}")
    a("")
    a(f"Label for every bullet: {LABELS[CLASS_PRED]}.")
    a("")
    a("## 6. What the AI specialists concluded (AI hypotheses)")
    a("")
    a(
        f"Label for this section: {LABELS['ai_hypothesis']}. Text is quoted from the event log; each "
        "statement lists the deterministic measurement events it cites."
    )
    a("")
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for s in evidence["statements"]:
        if s["agent_id"] != "supervisor":
            by_agent.setdefault(s["agent_id"], []).append(s)
    for agent in sorted(by_agent):
        a(f"### {agent}")
        a("")
        for s in by_agent[agent]:
            a(
                f"- **{s['event_id']}** ({s['kind']}{', status ' + s['status'] if s['status'] else ''}; "
                f"cites {', '.join(s['cited_evidence_event_ids']) or 'no measurement'}): "
                f"{_quote(request_text(s['kind'], s['text']))}"
            )
            nc = s["numeric_check"]
            if nc and nc["checked"]:
                if nc["unmatched"]:
                    a(
                        f"  - numeric check: {nc['matched']}/{nc['checked']} quoted numbers found in cited "
                        f"evidence; NOT found: {', '.join(nc['unmatched'])}"
                    )
                else:
                    a(f"  - numeric check: all {nc['checked']} quoted numbers match cited evidence")
        a("")
    a("## 7. Cross-agent exchange")
    a("")
    ex = evidence["exchange"]
    sm = evidence["summary"]
    if sm["no_challenge_or_revision"]:
        a(
            "**No challenge and no revision occurred in this run.** No conflict arose, so none is "
            "reported or implied."
        )
    for x in ex:
        parents = ", ".join(x["parent_event_ids"]) or "none"
        a(
            f"- {x['event_id']} {x['agent_id']} **{x['kind']}** (parents: {parents}): "
            f"{_quote(request_text(x['kind'], x['text']))[:400]}"
        )
    if not ex:
        a("- (no challenges, test requests, test results or revisions)")
    a("")
    a("## 8. Alternative explanations (as stated by the agents)")
    a("")
    seen = set()
    for s in evidence["statements"]:
        for alt in s["alternative_explanations"]:
            key = (s["agent_id"], alt)
            if key not in seen:
                seen.add(key)
                a(f"- {s['agent_id']} ({s['event_id']}): {_quote(alt)}")
    if not seen:
        a("- (none stated)")
    a("")
    a("## 9. Outstanding uncertainties")
    a("")
    for lim in manifest["limitations"]:
        a(f"- {lim}")
    seen = set()
    for s in evidence["statements"]:
        for lim in s["limitations"]:
            if lim not in seen and s["agent_id"] != "supervisor":
                seen.add(lim)
                a(f"- ({s['agent_id']}, {s['event_id']}) {_quote(lim)}")
    a("")
    v2 = manifest.get("derivation_version") == DERIVATION_V2
    if not v2:  # iceclaw-derive/1: original wording, kept byte-identical so old bundles verify
        a("## 10. Research follow-up priority")
        a("")
        a(
            f"{DISCLAIMER} The ranking below is a {RANK_NOTE_V1}. It orders locations that the "
            "recorded measurements named (raw peak, gradient-discontinuity candidates, examined "
            "windows) by how many distinct agents' evidence covers or names them (scrutiny, not "
            "physical support)."
        )
        a("")
        a("| Rank | Target | Kind | Lat, lon (deg) | Agents | Evidence events |")
        a("| --- | --- | --- | --- | --- | --- |")
        for r in rankings[:12]:
            a(
                f"| {r['rank']} | `{r['target_id']}` | {r['kind']} | {r['latitude']}, "
                f"{r['longitude']} | {r['n_distinct_agents']} | {r['evidence_event_ids']} |"
            )
        if not rankings:
            a("| - | (no locatable target was named by the measurements) | | | | |")
    else:
        a(
            "## 10. Research follow-up priority (investigation attention, not an ice-prospectivity ranking)"
        )
        a("")
        a(DISCLAIMER)
        a("")
        a(
            "**How to read this table.** It is an *investigation-attention* order: it lists the "
            "locations that the recorded measurements named (raw peak, gradient-discontinuity "
            "candidates, examined windows) and orders them by how many distinct agents' evidence "
            "covers or names each one. That reflects where scrutiny landed. It is **not an "
            "independently validated ice-prospectivity ranking**, it is not physical support for "
            "ice, and it is not a probability of ice."
        )
        a("")
        a(
            "**Ties.** Targets with the same agent and event counts are *tied*. Tied targets are "
            "listed **alphabetically by target id**, so their relative order carries **no "
            "importance**: for example, the raw maximum can appear below discontinuity candidates "
            "purely because of its name. Use the evidence event ids, not the rank number, to "
            "judge a target."
        )
        a("")
        a("| Rank | Target | Kind | Lat, lon (deg) | Agents | Evidence events | Tie |")
        a("| --- | --- | --- | --- | --- | --- | --- |")
        for r in rankings[:12]:
            tie = f"tied x{r['tie_group_size']} (alphabetical)" if r["tie_group_size"] > 1 else "-"
            a(
                f"| {r['rank']} | `{r['target_id']}` | {r['kind']} | {r['latitude']}, "
                f"{r['longitude']} | {r['n_distinct_agents']} | {r['evidence_event_ids']} | {tie} |"
            )
        if not rankings:
            a("| - | (no locatable target was named by the measurements) | | | | | |")
    a("")
    a("Next steps proposed by the agents (AI hypotheses; see their closing statements above):")
    a("")
    nxt = [
        s
        for s in evidence["statements"]
        if s["kind"] == "completion" and s["agent_id"] != "supervisor"
    ]
    for s in nxt:
        a(f"- {s['agent_id']} ({s['event_id']}): {_quote(s['text'])}")
    if not nxt:
        a("- (no closing statements)")
    a("")
    a("## 11. Provenance, reproducibility and verification")
    a("")
    a(
        f"- Source package: `{pkgm['embedded_path']}/` (content address `{pkgm['package_sha256']}`); "
        + (
            "offline portable: the exact source arrays are included."
            if pkgm["portable"]
            else "NOT embedded: verification requires the declared source package."
        )
    )
    a("- Attribution: " + " ".join(manifest["attribution"]))
    a(
        f"- Run: ended `{manifest['run'].get('ended_reason')}`; events {manifest['run'].get('events')}; "
        f"tool calls {manifest['run'].get('tool_calls')}; model calls {manifest['run'].get('llm_calls')}."
    )
    if manifest.get("notes"):
        for n in manifest["notes"]:
            a(f"- Note: {n}")
    if v2:
        prof = manifest.get("transcript_profile")
        red = manifest.get("transcript_redactions") or {}
        if prof == PROFILE_PUBLIC:
            a(
                f"- Transcript (`llm_calls.jsonl`): **{prof}**. {red.get('fields_redacted', 0)} opaque "
                f"provider reasoning field(s) ({red.get('characters_removed', 0)} characters) were "
                "replaced by a marker; the private transcript (sha256 "
                f"`{str(red.get('private_transcript_sha256'))[:16]}...`) is not included. Events, "
                "measurements, source package, agent-visible text and token usage are unchanged."
            )
        else:
            a(
                f"- Transcript (`llm_calls.jsonl`): **{prof}**: complete, including the provider's "
                "opaque reasoning payloads; intended for private audit. Use "
                "`python -m iceclaw.cli publish` to create a public-redacted bundle."
            )
    hint = (
        manifest.get("bundle_dir") or manifest["investigation_id"]
        if v2
        else manifest["investigation_id"]
    )
    a(f"- Verify without any API key or GPU: `python -m iceclaw.cli verify --run {hint}`")
    a("")
    return "\n".join(L) + "\n"


# ===================================================================================== bundle
def _usage(rec: RunRecord) -> dict[str, Any]:
    rows = rec.llm_calls
    by: dict[str, dict[str, int]] = {}
    for r in rows:
        a = by.setdefault(r["agent_id"], {"calls": 0, "input": 0, "output": 0, "total": 0})
        a["calls"] += 1
        a["input"] += r["input_tokens"] or 0
        a["output"] += r["output_tokens"] or 0
        a["total"] += r["total_tokens"] or 0
    return {
        "source": "provider-reported usage stored per call",
        "provider": rec.provider,
        "model": rec.model,
        "calls": len(rows),
        "ok_calls": sum(1 for r in rows if r["ok"]),
        "failed_calls": sum(1 for r in rows if not r["ok"]),
        "input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "total_tokens": sum(r["total_tokens"] or 0 for r in rows),
        "by_agent": dict(sorted(by.items())),
        "per_call": [
            {
                k: r[k]
                for k in (
                    "seq",
                    "agent_id",
                    "started_utc",
                    "finished_utc",
                    "ok",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                )
            }
            for r in rows
        ],
        "cost": {
            "status": "unpriced",
            "note": "no independently verified model rate is configured",
        },
    }


def build_manifest(
    rec: RunRecord,
    pm: dict[str, Any],
    *,
    format_version: int = 2,
    profile: str = PROFILE_FULL,
    redactions: dict[str, Any] | None = None,
    bundle_dir: str | None = None,
) -> dict[str, Any]:
    reg = pm["region"]
    pred = next(f for f in pm["files"] if f["kind"] == "nasa_ibm_model_prediction")
    budget = {k: rec.summary["budget"][k] for k in RunBudget.model_fields}
    tools = [
        {
            "tool_run_id": r["record"]["tool_run_id"],
            "tool_id": r["record"]["tool_id"],
            "agent_id": r["agent_id"],
            "event_id": r["event_id"],
            "parameters": {
                k: v
                for k, v in r["record"]["parameters"].items()
                if v is not None and k != "unavailable_detail"
            },
        }
        for r in rec.measurements.values()
    ]
    im = InputManifest(
        investigation_id=rec.investigation_id,
        region_id=rec.region_id,
        source_mode=SourceMode(reg["source_mode"]),
        source_dataset=reg["source_dataset"],
        source_split=reg["source_split"],
        source_revision=reg["source_revision"],
        source_sha256=reg["upstream_sectors_sha256"],
        prediction_sha256=pred["sha256"],
        prediction_checkpoint_id=reg["prediction_checkpoint_id"],
        llm_provider=rec.provider,
        llm_model=rec.model,
        code_revision=rec.code_revision,
        visible_modalities=sorted(f["modality"] for f in pm["files"]),
        budget=RunBudget(**budget),
        tools_executed=tools,
        limitations=LIMITATIONS,
    )
    s = rec.summary
    d = im.model_dump(mode="json")
    d.update(
        {
            "bundle_format": BUNDLE_FORMAT_V1 if format_version == 1 else BUNDLE_FORMAT_V2,
            "derivation_version": DERIVATION_V1 if format_version == 1 else DERIVATION_V2,
            "mode": rec.mode,
            "created_utc": datetime.now(UTC).isoformat(),
            "source_package": {
                "embedded_path": "source",
                "package_sha256": pm["package_sha256"],
                "portable": True,
                "files": {
                    f["file_id"]: {
                        "modality": f["modality"],
                        "kind": f["kind"],
                        "unit": f["unit"],
                        "sha256": f["sha256"],
                        "file": f["file"],
                    }
                    for f in pm["files"]
                },
            },
            "attribution": pm["attribution_and_modification_notice"],
            "model_usage": _usage(rec),
            "prompt_policy": s.get("limits")
            or {"note": "recorded before P6: unbounded digest (P5 prompt policy); not reproduced"},
            "run": {
                k: s.get(k)
                for k in (
                    "ended_reason",
                    "elapsed_s",
                    "events",
                    "tool_calls",
                    "llm_calls",
                    "rounds",
                    "exit_reasons",
                    "problems",
                )
            },
            "notes": rec.notes,
            "integrity": {},
        }
    )
    if format_version >= 2:
        d["transcript_profile"] = profile
        d["transcript_redactions"] = redactions or {
            "fields_redacted": 0,
            "characters_removed": 0,
            "note": "complete transcript; contains provider-opaque reasoning payloads",
        }
        d["bundle_dir"] = bundle_dir
    return d


def _w(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8", newline="\n")
    return pkg.sha256_bytes(text.encode("utf-8"))


def derive_files(rec: RunRecord, manifest: dict[str, Any], pm: dict[str, Any]) -> dict[str, Any]:
    """Everything derivable from (events, measurements, manifest, package): used by export AND verify."""
    version = 1 if manifest.get("derivation_version") == DERIVATION_V1 else 2
    evidence = build_evidence(rec)
    geo = {"projection": pm["geometry"]["projection"], "grid": pm["geometry"]["grid"]}
    rankings = build_rankings(evidence, geo["projection"], geo["grid"], version)
    view = {k: v for k, v in manifest.items() if k != "integrity"}
    return {
        "evidence": evidence,
        "rankings": rankings,
        "rankings_csv": rankings_csv(rankings, version),
        "report": render_report(view, evidence, rankings, geo),
    }


@dataclass
class ExportResult:
    out_dir: Path
    verification: Any  # VerificationReport


def export_bundle(
    rec: RunRecord,
    out_dir: Path,
    spec: pkg.SourceSpec | None = None,
    *,
    format_version: int = 2,
    profile: str = PROFILE_FULL,
    redactions: dict[str, Any] | None = None,
) -> ExportResult:
    """Write the bundle; run the offline verifier on it and store the result in verification.json.

    ``format_version=1`` writes the original P6 contract (kept so it stays testable); the default
    writes ``iceclaw-bundle/2``. ``profile`` is ``full-private`` (complete transcript) or
    ``public-redacted`` (opaque provider reasoning payloads already removed from ``rec``).
    """
    from iceclaw.reports.verify import verify_bundle

    if profile not in (PROFILE_FULL, PROFILE_PUBLIC):
        raise ValueError(f"unknown transcript profile {profile!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = out_dir / "source"
    if not (src_dir / pkg.MANIFEST_NAME).is_file():
        if spec is None:
            raise ValueError("no source package present and no SourceSpec to write one")
        pkg.write_package(spec, src_dir)
    pm = pkg.read_manifest(src_dir)
    manifest = build_manifest(
        rec,
        pm,
        format_version=format_version,
        profile=profile,
        redactions=redactions,
        bundle_dir=out_dir.name,
    )
    derived = derive_files(rec, manifest, pm)
    integrity: dict[str, str] = {}
    events_text = "".join(e.model_dump_json() + "\n" for e in rec.events)
    integrity["events.jsonl"] = _w(out_dir / "events.jsonl", events_text)
    integrity["evidence.json"] = _w(
        out_dir / "evidence.json", json.dumps(derived["evidence"], indent=1)
    )
    integrity["rankings.csv"] = _w(out_dir / "rankings.csv", derived["rankings_csv"])
    integrity["report.md"] = _w(out_dir / "report.md", derived["report"])
    llm_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rec.llm_calls)
    integrity["llm_calls.jsonl"] = _w(out_dir / "llm_calls.jsonl", llm_text)
    manifest["integrity"] = integrity
    _w(out_dir / "input_manifest.json", json.dumps(manifest, indent=1))
    _w(out_dir / "verification.json", json.dumps({"pending": True}))
    report = verify_bundle(out_dir)
    _w(out_dir / "verification.json", json.dumps(report.to_dict(), indent=1))
    return ExportResult(out_dir, report)


# ================================================================================ public export
class PublishRefused(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _redact(obj: Any, stats: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in OPAQUE_KEYS and isinstance(v, str) and v and v != REDACTED_MARKER:
                stats["fields"] += 1
                stats["chars"] += len(v)
                stats["keys"].add(k)
                out[k] = REDACTED_MARKER
            else:
                out[k] = _redact(v, stats)
        return out
    if isinstance(obj, list):
        return [_redact(x, stats) for x in obj]
    return obj


def redact_llm_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy of the transcript rows with provider-opaque reasoning payloads replaced by a marker.

    Only the values of ``protected_data`` / ``encrypted_content`` change; roles, text, tool calls,
    tool results, ids, timestamps and token usage are untouched, so totals and every agent-visible
    word are preserved. The input rows (the private transcript) are never modified.
    """
    stats: dict[str, Any] = {"fields": 0, "chars": 0, "keys": set()}
    out = []
    for r in rows:
        nr = dict(r)
        for col in ("request_json", "response_json"):
            if nr.get(col):
                nr[col] = json.dumps(_redact(json.loads(nr[col]), stats), default=str)
        out.append(nr)
    return out, {
        "fields_redacted": stats["fields"],
        "characters_removed": stats["chars"],
        "keys": sorted(stats["keys"]),
    }


def publish_bundle(src: Path, out_dir: Path) -> ExportResult:
    """Create a PUBLIC-REDACTED bundle from a verified bundle, without any model call.

    The source (private) bundle is only read. The public bundle has the same events, measurement
    records and byte-identical source package; its transcript has the provider's opaque
    reasoning payloads replaced by a marker; it has no ``internal/`` history. It is a new
    ``iceclaw-bundle/2`` export that records the hash of the private transcript it was made from.
    """
    import shutil

    from iceclaw.reports.verify import record_from_bundle, verify_bundle

    src, out_dir = Path(src), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise PublishRefused("OUT_DIR_EXISTS", f"{out_dir} is not empty")
    rep = verify_bundle(src)
    if not rep.ok:
        raise PublishRefused(
            "SOURCE_BUNDLE_NOT_VERIFIED",
            "refusing to publish an unverified bundle: " + ", ".join(c.id for c in rep.failures),
        )
    manifest_bytes = (src / "input_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("transcript_profile") == PROFILE_PUBLIC:
        raise PublishRefused("ALREADY_PUBLIC", f"{src.name} is already a public-redacted bundle")
    embedded = src / manifest["source_package"]["embedded_path"]
    if not (embedded / pkg.MANIFEST_NAME).is_file():
        raise PublishRefused("NO_EMBEDDED_SOURCE", "the bundle has no embedded source package")
    rec = record_from_bundle(src, manifest)
    private_bytes = (src / "llm_calls.jsonl").read_bytes()
    rec.llm_calls, stats = redact_llm_rows(rec.llm_calls)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(embedded, out_dir / "source")
    redactions = {
        **stats,
        "marker": REDACTED_MARKER,
        "private_transcript_sha256": pkg.sha256_bytes(private_bytes),
        "private_transcript_bytes": len(private_bytes),
        "source_bundle_dir": src.name,
        "source_bundle_format": manifest["bundle_format"],
        "source_bundle_manifest_sha256": pkg.sha256_bytes(manifest_bytes),
    }
    rec.notes = [
        *rec.notes,
        f"Public bundle published (no model call) from the private bundle '{src.name}' "
        f"(manifest sha256 {redactions['source_bundle_manifest_sha256'][:16]}...): provider-opaque "
        "reasoning payloads were removed from the transcript; events, measurements and the source "
        "package are verified copies.",
    ]
    return export_bundle(
        rec, out_dir, None, format_version=2, profile=PROFILE_PUBLIC, redactions=redactions
    )
