"""Offline verification of an investigation bundle. No model call, no API key, no GPU.

What is checked (each is an independent, named check; any failed ERROR check => exit code 1):

* structure and per-file SHA256 integrity recorded in the input manifest;
* events: valid, unique ids, every parent exists and PRECEDES its child, challenge/revision
  reference rules (challenge another agent's event; revise your own claim with newer evidence);
* source package (embedded, external content-addressed, or local): every file hash, the
  package identity (``package_sha256``), grid/units/resolution/footprint consistency;
* **recomputation**: every recorded deterministic measurement is re-run from the exact source
  arrays and compared within its declared tolerance;
* event text vs measurement: the numbers an event displays are those of its measurement;
* ``evidence.json``, ``rankings.csv`` and ``report.md`` are RE-DERIVED from the events and
  measurements and must equal the saved files (edits that are not reproduced are detected);
* usage totals and budgets; absence of forbidden content (PRO, keys, absolute paths, ``.env``);
* affirmative ice-discovery wording (warning) and quoted numbers not found in cited evidence
  (warning).

Limits (stated, not hidden): this proves the numbers follow from the source data and that the
bundle is internally consistent. It cannot prove that a language model wrote the agent text.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from iceclaw.contracts import EventKind, EvidenceEvent, InputManifest
from iceclaw.reports import export as ex
from iceclaw.reports import package as pkg
from iceclaw.science.model_audit import matches, recompute
from iceclaw.swarm.runtime import measurement_event_text, recompute_event_text

MODE_EMBEDDED = "embedded source package (offline portable)"
MODE_EXTERNAL = "external content-addressed source package"
MODE_LOCAL = "requires local source data"
EVIDENCE_KINDS = (EventKind.MEASUREMENT, EventKind.TEST_RESULT)


@dataclass
class Check:
    id: str
    ok: bool
    severity: str  # "error" | "warning" | "info"
    detail: str = ""


@dataclass
class VerificationReport:
    bundle: str
    mode: str = "unresolved"
    checks: list[Check] = field(default_factory=list)

    def add(self, id: str, ok: bool, detail: str = "", severity: str = "error") -> bool:
        self.checks.append(Check(id, ok, severity, detail))
        return ok

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.severity == "error")

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "warning" and not c.ok]

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.severity == "error" and not c.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source_mode": self.mode,
            "n_checks": len(self.checks),
            "n_failed": len(self.failures),
            "n_warnings": len(self.warnings),
            "checks": [c.__dict__ for c in self.checks],
            "note": "recorded at export time; re-run `verify` for a fresh result",
        }

    def render(self) -> str:
        lines = [f"verification of {self.bundle}", f"source: {self.mode}"]
        for c in self.checks:
            tag = "PASS" if c.ok else ("FAIL" if c.severity == "error" else "WARN")
            lines.append(f"  [{tag}] {c.id}" + (f": {c.detail}" if c.detail else ""))
        lines.append(
            f"RESULT: {'VERIFIED' if self.ok else 'FAILED'} "
            f"({len(self.checks) - len(self.failures) - len(self.warnings)} passed, "
            f"{len(self.failures)} failed, {len(self.warnings)} warning(s))"
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------------- loaders
def record_from_bundle(b: Path, manifest: dict[str, Any]) -> ex.RunRecord:
    """Rebuild a RunRecord from bundle files only (events + embedded measurement records)."""
    events = [
        EvidenceEvent.model_validate_json(x)
        for x in (b / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    evidence = json.loads((b / "evidence.json").read_text(encoding="utf-8"))
    meas = {
        m["record"]["tool_run_id"]: {
            "event_id": m["event_id"],
            "agent_id": m["agent_id"],
            "record": m["record"],
        }
        for m in evidence["measurements"]
    }
    llm = [
        json.loads(x) for x in (b / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if x
    ]
    return ex.RunRecord(
        investigation_id=manifest["investigation_id"],
        region_id=manifest["region_id"],
        events=events,
        measurements=meas,
        llm_calls=llm,
        summary={
            **manifest.get("run", {}),
            "budget": manifest["budget"],
            "limits": manifest.get("prompt_policy"),
        },
        mode=manifest["mode"],
        provider=manifest["llm_provider"],
        model=manifest["llm_model"],
        code_revision=manifest["code_revision"],
        notes=manifest.get("notes", []),
    )


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- source resolution
def resolve_source(
    rep: VerificationReport,
    b: Path,
    manifest: dict[str, Any],
    source_package: Path | None,
    local_settings: Any | None,
) -> tuple[Path | None, Path | None]:
    """(package dir, tempdir to clean up). Sets ``rep.mode``."""
    want = manifest["source_package"]["package_sha256"]
    if source_package is not None:
        rep.mode = MODE_EXTERNAL
        return Path(source_package), None
    embedded = b / manifest["source_package"]["embedded_path"]
    if (embedded / pkg.MANIFEST_NAME).is_file():
        rep.mode = MODE_EMBEDDED
        return embedded, None
    if local_settings is not None:
        from iceclaw.data.catalog import discover

        rep.mode = MODE_LOCAL
        tmp = Path(tempfile.mkdtemp(prefix="iceclaw-local-src-"))
        try:
            cat = discover(local_settings)
            spec = pkg.spec_from_catalog_record(cat.get(manifest["region_id"]))
            pkg.write_package(spec, tmp)
            return tmp, tmp
        except Exception as e:  # catalog / package failure is a verification failure, not a crash
            rep.add(
                "source_available", False, f"local source data unusable: {type(e).__name__}: {e}"
            )
            shutil.rmtree(tmp, ignore_errors=True)
            return None, None
    rep.mode = MODE_LOCAL
    rep.add(
        "source_available",
        False,
        f"bundle declares source package {want[:16]}... but none is embedded; this bundle "
        "requires local source data: pass --source-package PATH or --local-source",
    )
    return None, None


# ------------------------------------------------------------------------------- checks
FORBIDDEN_PATTERNS = {
    "expert target PRO": re.compile(r"\bPRO\b"),
    # a credential starts at a token boundary; "sk-" in the middle of an opaque base64 blob
    # (e.g. the provider's protected reasoning data) is chance, not a key
    "API-key-like string": re.compile(
        r"(?<![A-Za-z0-9_\-])sk-(?:[A-Za-z0-9]{2,12}-)?[A-Za-z0-9_\-]{20,}"
    ),
    "absolute path": re.compile(r"(?<![\w.])(/home/|/Users/|/root/|[A-Za-z]:\\\\)"),
    ".env reference": re.compile(r"\.env\b"),
    "sealed/evaluator material": re.compile(r"sealed_key|reference\.npy|fault_id"),
}
ICE_CLAIM = re.compile(
    r"\b(?:confirmed|detected|discovered|found|proved|proven)\b[^.]{0,30}\b(?:lunar\s+)?(?:water\s+)?ice\b"
    r"|\bice\s+(?:was|is|has been)\s+(?:confirmed|detected|discovered|found)\b",
    re.I,
)
NEGATION = re.compile(
    r"\b(?:not|no|never|nor|neither|without|cannot|can't|isn't|doesn't|does not|do not)\b", re.I
)


def _scan_text(text: str) -> list[str]:
    return [name for name, pat in FORBIDDEN_PATTERNS.items() if pat.search(text)]


def verify_bundle(
    bundle: str | Path, source_package: str | Path | None = None, local_settings: Any | None = None
) -> VerificationReport:
    b = Path(bundle)
    rep = VerificationReport(str(b))
    return _verify(rep, b, Path(source_package) if source_package else None, local_settings)


def _verify(
    rep: VerificationReport, b: Path, source_package: Path | None, local_settings: Any | None
):
    # 1. structure --------------------------------------------------------------------
    required = [*ex.CONTRACT_FILES, "llm_calls.jsonl"]
    missing = [f for f in required if not (b / f).is_file()]
    if not rep.add(
        "files_present", not missing, f"missing: {missing}" if missing else f"{len(required)} files"
    ):
        return rep
    try:
        manifest = _read_json(b / "input_manifest.json")
        InputManifest.model_validate({k: manifest[k] for k in InputManifest.model_fields})
        ok_fmt = manifest.get("bundle_format") in ex.SUPPORTED_BUNDLE_FORMATS
        rep.add(
            "manifest_schema",
            ok_fmt,
            "" if ok_fmt else f"unknown bundle format {manifest.get('bundle_format')!r}",
        )
        if not ok_fmt:
            return rep
    except (ValidationError, KeyError, json.JSONDecodeError) as e:
        rep.add("manifest_schema", False, f"{type(e).__name__}: {str(e)[:200]}")
        return rep
    fmt = manifest["bundle_format"]
    got = manifest.get("derivation_version")
    expected = ex.FORMAT_DERIVATION[fmt]
    same = got == expected
    rep.add(
        "derivation_version",
        same,
        f"{got} (bundle format {fmt})"
        if same
        else f"bundle format {fmt} requires derivation {expected!r} but the bundle declares {got!r}; "
        f"supported: {list(ex.SUPPORTED_DERIVATIONS)}; use a matching IceClaw version or re-export "
        "the run",
    )
    if not same:
        return rep

    # 2. file integrity -----------------------------------------------------------------
    bad = []
    for name, want in manifest["integrity"].items():
        p = b / name
        if not p.is_file() or pkg.sha256_bytes(p.read_bytes()) != want:
            bad.append(name)
    rep.add(
        "integrity_hashes",
        not bad and set(manifest["integrity"]) == set(ex.INTEGRITY_FILES),
        f"modified or missing: {bad}"
        if bad
        else f"{len(manifest['integrity'])} files match the manifest",
    )

    # 2b. transcript profile (iceclaw-bundle/2) -----------------------------------------------
    if fmt == ex.BUNDLE_FORMAT_V2:
        _check_transcript_profile(rep, b, manifest)

    # 3. events --------------------------------------------------------------------------
    try:
        events = [
            EvidenceEvent.model_validate_json(x)
            for x in (b / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        evidence = _read_json(b / "evidence.json")
    except (ValidationError, json.JSONDecodeError) as e:
        rep.add("events_valid", False, f"{type(e).__name__}: {str(e)[:200]}")
        return rep
    ids = [e.event_id for e in events]
    rep.add(
        "events_valid",
        len(set(ids)) == len(ids) and bool(events),
        f"{len(events)} events, unique ids",
    )
    pos = {e.event_id: i for i, e in enumerate(events)}
    byid = {e.event_id: e for e in events}
    broken = [
        f"{e.event_id}->{p}"
        for e in events
        for p in e.parent_event_ids
        if p not in pos or pos[p] >= pos[e.event_id]
    ]
    rep.add(
        "event_parents",
        not broken,
        f"missing or non-preceding parents: {broken[:6]}"
        if broken
        else "every parent exists and precedes its child",
    )
    rule_bad: list[str] = []
    for e in events:
        if e.event_kind is EventKind.CHALLENGE and e.parent_event_ids:
            t = byid.get(e.parent_event_ids[0])
            if t is None or t.agent_id == e.agent_id:
                rule_bad.append(f"{e.event_id}: challenge must target another agent's event")
        if e.event_kind is EventKind.REVISION and e.parent_event_ids:
            old = byid.get(e.parent_event_ids[0])
            newer = [
                p
                for p in e.parent_event_ids[1:]
                if p in byid
                and byid[p].event_kind in EVIDENCE_KINDS
                and old is not None
                and pos[p] > pos[old.event_id]
            ]
            if (
                old is None
                or old.agent_id != e.agent_id
                or old.event_kind not in (EventKind.CLAIM, EventKind.REVISION)
            ):
                rule_bad.append(
                    f"{e.event_id}: revision must revise the same agent's claim/revision"
                )
            elif not newer:
                rule_bad.append(f"{e.event_id}: revision cites no evidence newer than the claim")
        if e.event_kind is EventKind.TEST_RESULT and e.parent_event_ids:
            req = byid.get(e.parent_event_ids[0])
            if req is None or req.event_kind is not EventKind.TEST_REQUEST:
                rule_bad.append(f"{e.event_id}: test_result must answer a test_request")
    rep.add(
        "reference_rules",
        not rule_bad,
        "; ".join(rule_bad[:4]) or "challenge/revision/result references are valid",
    )

    # 4. measurement <-> event linkage ----------------------------------------------------
    meas = {m["event_id"]: m for m in evidence.get("measurements", [])}
    linked_bad = []
    for e in events:
        if e.event_kind in EVIDENCE_KINDS:
            m = meas.get(e.event_id)
            if (
                m is None
                or m["tool_run_id"] != e.tool_run_id
                or m["agent_id"] != e.agent_id
                or m["tool_id"] != e.tool_id
            ):
                linked_bad.append(e.event_id)
    extra = sorted(set(meas) - {e.event_id for e in events if e.event_kind in EVIDENCE_KINDS})
    rep.add(
        "measurement_events_linked",
        not linked_bad and not extra,
        f"unlinked events {linked_bad}, orphan records {extra}"
        if (linked_bad or extra)
        else f"{len(meas)} measurements linked",
    )

    # 5. source package -------------------------------------------------------------------
    pdir, tmp = resolve_source(rep, b, manifest, source_package, local_settings)
    try:
        if pdir is None:
            return rep
        return _verify_with_source(rep, b, manifest, evidence, events, meas, pdir)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def _verify_with_source(rep, b, manifest, evidence, events, meas, pdir) -> VerificationReport:
    try:
        loaded = pkg.load_package(pdir)
    except pkg.PackageError as e:
        rep.add("source_package_integrity", False, str(e))
        return rep
    pm, arrays = loaded.manifest, loaded.arrays
    rep.add("source_package_integrity", True, f"{len(pm['files'])} source files match their hashes")
    decl = manifest["source_package"]
    ident = pm["package_sha256"] == decl["package_sha256"]
    rep.add(
        "source_package_identity",
        ident,
        "content address matches the bundle"
        if ident
        else f"package {pm['package_sha256'][:16]}... != declared {decl['package_sha256'][:16]}...",
    )
    file_bad = [
        fid
        for fid, f in decl["files"].items()
        if not any(x["file_id"] == fid and x["sha256"] == f["sha256"] for x in pm["files"])
    ]
    pred = next((f for f in pm["files"] if f["kind"] == pkg.KIND_PREDICTION), None)
    rep.add(
        "source_hashes_declared",
        not file_bad and pred is not None and pred["sha256"] == manifest["prediction_sha256"],
        f"declared file hashes differ: {file_bad}"
        if file_bad
        else "per-file and prediction hashes match the manifest",
    )

    # geometry / units consistency
    geo = pm["geometry"]
    units = {f["file_id"]: f["unit"] for f in pm["files"]}
    shas = {f["file_id"]: f["sha256"] for f in pm["files"]}
    incons: list[str] = []
    for m in evidence["measurements"]:
        r = m["record"]
        if abs((r["resolution_m"] or 0) - geo["resolution_m"]) > 1e-9:
            incons.append(
                f"{m['event_id']}: resolution {r['resolution_m']} != {geo['resolution_m']}"
            )
        for fid, sha in r["source_sha256"].items():
            if shas.get(fid) != sha:
                incons.append(f"{m['event_id']}: source {fid} hash differs from the package")
        if (
            not r["parameters"].get("window")
            and r["footprint"]
            and [list(c) for c in r["footprint"]]
            != [list(c) for c in geo["footprint_lat_lon_NW_NE_SE_SW"]]
        ):
            incons.append(f"{m['event_id']}: footprint differs from the package footprint")
        layer_units = {units.get(f) for f in r["source_file_ids"]}
        u = r.get("unit") or ""
        if (
            len(layer_units) == 1
            and next(iter(layer_units))
            and not u.startswith(("mixed", "pixels"))
            and not u.startswith(next(iter(layer_units)))
        ):
            incons.append(f"{m['event_id']}: unit {u!r} != layer unit {next(iter(layer_units))!r}")
    rep.add(
        "geometry_units_consistent",
        not incons,
        "; ".join(incons[:4]) or "grid, resolution, footprint, units and source hashes agree",
    )

    # 6. recomputation ----------------------------------------------------------------------
    from iceclaw.contracts import ToolMeasurement

    fails, n = [], 0
    for m in evidence["measurements"]:
        rec = dict(m["record"])
        base_id = rec["tool_run_id"].split("#")[0]
        try:
            recorded = ToolMeasurement.model_validate({**rec, "tool_run_id": base_id})
            again = recompute(recorded, arrays)
            n += 1
            ok = matches(recorded, again) and again.tool_run_id == base_id
        except Exception as e:
            ok = False
            fails.append(f"{m['event_id']} ({type(e).__name__}: {str(e)[:80]})")
            continue
        if not ok:
            fails.append(
                f"{m['event_id']} {recorded.tool_id}: recorded result does not reproduce from the source"
            )
    rep.add(
        "recompute_measurements",
        not fails and n == len(evidence["measurements"]),
        "; ".join(fails[:4])
        if fails
        else f"{n} deterministic measurements reproduced within their declared tolerances from the source arrays",
    )

    # 7. event text vs measurement ------------------------------------------------------------
    text_bad = []
    for e in events:
        m = meas_for(e, evidence)
        if m is None:
            continue
        r = m["record"]
        tid = r["tool_run_id"]
        if "#recompute" in tid:
            orig = tid.split("#")[0]
            allowed = (recompute_event_text(orig, True), recompute_event_text(orig, False))
            if (e.claim_or_request or "") not in allowed:
                text_bad.append(e.event_id)
        elif (e.claim_or_request or "") != measurement_event_text(
            ToolMeasurement.model_validate(r)
        ):
            text_bad.append(e.event_id)
    rep.add(
        "event_text_matches_measurement",
        not text_bad,
        f"event text differs from its measurement: {text_bad[:6]}"
        if text_bad
        else "displayed numbers equal the recorded measurements",
    )

    # 8. re-derive evidence / rankings / report ----------------------------------------------------
    rec = record_from_bundle(b, manifest)
    try:
        derived = ex.derive_files(rec, manifest, pm)
    except Exception as e:  # inconsistent inputs must be REPORTED, never crash the verifier
        rep.add(
            "evidence_rederived",
            False,
            f"cannot re-derive evidence: {type(e).__name__}: {str(e)[:120]}",
        )
        return rep
    saved_evidence = evidence
    same_ev = json.loads(json.dumps(derived["evidence"])) == saved_evidence
    rep.add(
        "evidence_rederived",
        same_ev,
        "evidence.json equals the re-derivation from events + measurements"
        if same_ev
        else "evidence.json differs from what the events and measurements imply",
    )
    same_rank = (b / "rankings.csv").read_text(encoding="utf-8") == derived["rankings_csv"]
    rep.add(
        "rankings_rederived",
        same_rank,
        "rankings.csv equals its re-derivation"
        if same_rank
        else "rankings.csv differs from its re-derivation",
    )
    same_rep = (b / "report.md").read_text(encoding="utf-8") == derived["report"]
    rep.add(
        "report_rederived",
        same_rep,
        "report.md equals its re-derivation"
        if same_rep
        else "report.md differs from its re-derivation",
    )
    unmatched = [
        (s["event_id"], s["numeric_check"]["unmatched"])
        for s in derived["evidence"]["statements"]
        if s["numeric_check"] and s["numeric_check"]["unmatched"]
    ]
    rep.add(
        "numeric_claims_in_cited_evidence",
        not unmatched,
        f"{len(unmatched)} statement(s) quote numbers not found in their cited measurements: {unmatched[:4]}"
        if unmatched
        else "all quoted numbers match cited measurements",
        severity="warning",
    )

    # 9. usage and budgets -----------------------------------------------------------------------
    mu = manifest["model_usage"]
    rows = rec.llm_calls
    sums_ok = (
        mu["calls"] == len(rows)
        and mu["total_tokens"] == sum(r["total_tokens"] or 0 for r in rows)
        and mu["input_tokens"] == sum(r["input_tokens"] or 0 for r in rows)
        and mu["output_tokens"] == sum(r["output_tokens"] or 0 for r in rows)
        and len(mu["per_call"]) == len(rows)
    )
    bud = manifest["budget"]
    agent_events = [e for e in events if e.agent_id != "supervisor"]
    within = (
        len(agent_events) <= bud["max_events"]
        and len(rec.measurements) <= bud["max_tool_calls"]
        and len(rows) <= bud["max_llm_calls"]
    )
    rep.add(
        "usage_and_budgets",
        sums_ok and within,
        f"{len(rows)} model calls, {mu['total_tokens']} tokens (provider-reported); within budget"
        if sums_ok and within
        else "token totals or budget ceilings are inconsistent with the recorded calls",
    )
    rep.add(
        "cost_unpriced",
        mu["cost"]["status"] == "unpriced",
        "no fabricated token price",
        severity="warning",
    )

    # 10. forbidden content & honesty -----------------------------------------------------------------
    hits: list[str] = []
    for name in (
        "input_manifest.json",
        "events.jsonl",
        "evidence.json",
        "report.md",
        "rankings.csv",
        "llm_calls.jsonl",
    ):
        for h in _scan_text((b / name).read_text(encoding="utf-8")):
            hits.append(f"{name}: {h}")
    names = (
        [p.name for p in (b / manifest["source_package"]["embedded_path"]).rglob("*")]
        if (b / manifest["source_package"]["embedded_path"]).is_dir()
        else []
    )
    for n in names:
        if re.search(r"(^|_)PRO\.|reference|sealed|\.env|ice-sectors", n):
            hits.append(f"source/{n}: forbidden file")
    rep.add(
        "no_forbidden_content",
        not hits,
        "; ".join(hits[:5]) or "no PRO, key-like string, absolute path, .env or sealed material",
    )
    text = (b / "report.md").read_text(encoding="utf-8")
    banner = ex.DISCLAIMER in text and "prospectivity index" in text
    rep.add(
        "report_disclaimer",
        banner,
        "prospectivity-index disclaimer present" if banner else "required disclaimer missing",
    )
    claims = []
    for e in events:
        if e.agent_id == "supervisor" or not e.claim_or_request:
            continue
        for m in ICE_CLAIM.finditer(e.claim_or_request):
            window = e.claim_or_request[max(0, m.start() - 60) : m.end() + 5]
            if not NEGATION.search(window):
                claims.append(f"{e.event_id} ({e.agent_id}): ...{m.group(0)}...")
    rep.add(
        "no_affirmative_ice_claim",
        not claims,
        "; ".join(claims[:3]) or "no agent statement affirms ice detection",
        severity="warning",
    )
    return rep


_BASE64ISH = re.compile(r"[A-Za-z0-9_\-+/=]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _opaque_in(obj: Any, where: str, out: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ex.OPAQUE_KEYS and isinstance(v, str) and v and v != ex.REDACTED_MARKER:
                out.append(f"{where}.{k} holds {len(v)} opaque characters")
            elif isinstance(v, str) and len(v) >= 1000 and _BASE64ISH.fullmatch(v):
                out.append(f"{where}.{k} holds a {len(v)}-character opaque blob")
            else:
                _opaque_in(v, f"{where}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _opaque_in(v, f"{where}[{i}]", out)


def opaque_violations(text: str) -> list[str]:
    """Provider-opaque payloads present in a transcript (empty = clean)."""
    out: list[str] = []
    for n, line in enumerate(text.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(f"line {n} is not JSON")
            continue
        for col in ("request_json", "response_json"):
            raw = row.get(col)
            if not raw:
                continue
            try:
                _opaque_in(json.loads(raw), f"line{n}.{col}", out)
            except json.JSONDecodeError:
                out.append(f"line {n} {col} is not JSON")
    return out


def _check_transcript_profile(rep: VerificationReport, b: Path, manifest: dict[str, Any]) -> None:
    prof = manifest.get("transcript_profile")
    red = manifest.get("transcript_redactions")
    ok = prof in (ex.PROFILE_FULL, ex.PROFILE_PUBLIC) and isinstance(red, dict)
    rep.add("transcript_profile", ok, str(prof) if ok else f"invalid transcript profile {prof!r}")
    if not ok or prof != ex.PROFILE_PUBLIC:
        return
    problems = opaque_violations((b / "llm_calls.jsonl").read_text(encoding="utf-8"))
    meta_ok = (
        isinstance(red.get("fields_redacted"), int)
        and bool(_SHA256.fullmatch(str(red.get("private_transcript_sha256", ""))))
        and red.get("marker") == ex.REDACTED_MARKER
    )
    rep.add(
        "public_transcript_redacted",
        not problems and meta_ok,
        "; ".join(problems[:3])
        if problems
        else (
            f"no opaque provider data; {red.get('fields_redacted')} field(s) were redacted from "
            f"the private transcript sha256 {str(red.get('private_transcript_sha256'))[:16]}..."
            if meta_ok
            else "redaction metadata is missing or malformed"
        ),
    )
    internal = b / "internal"
    rep.add(
        "public_bundle_no_internal_history",
        not internal.exists(),
        "no internal/ run history" if not internal.exists() else "internal/ must not be published",
    )


def meas_for(e: EvidenceEvent, evidence: dict[str, Any]) -> dict[str, Any] | None:
    if e.event_kind not in EVIDENCE_KINDS:
        return None
    return next((m for m in evidence["measurements"] if m["event_id"] == e.event_id), None)
