"""P6: bundle export, offline verification, tamper detection, replay, service and CLI.

Most tests use a SYNTHETIC source package (real GeoTIFF/WebP files written with Rasterio/Pillow)
and scripted agents, so they need no downloaded data and no API key. Real-data tests are at the
end and skip if the data are absent. A fake model proves the pipeline, not live operation.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from bundle_helpers import (
    copy_bundle,
    failed_ids,
    rehash,
    rich_policies,
    scripted_service,
    synthetic_spec,
)

from iceclaw.cli import main
from iceclaw.contracts import InputManifest
from iceclaw.reports import export as ex
from iceclaw.reports import package as pkg
from iceclaw.reports.verify import MODE_EMBEDDED, MODE_EXTERNAL, MODE_LOCAL, verify_bundle

EXPECTED_TOP = {
    "input_manifest.json", "events.jsonl", "evidence.json", "report.md", "rankings.csv",
    "verification.json", "llm_calls.jsonl", "source", "internal",
}  # fmt: skip


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("p6")
    svc = scripted_service(root / "runs", root / "src")
    res = svc.investigate("synthetic-region", run_id="inv-bundle-test")
    return svc, res, root


@pytest.fixture
def bundle(built, tmp_path):
    _, res, _ = built
    return copy_bundle(res.run_dir, tmp_path / "bundle")


def load(b: Path, name: str):
    return json.loads((b / name).read_text())


# =============================================================================== export
def test_scripted_investigation_exports_a_verified_bundle(built):
    _, res, _ = built
    assert {p.name for p in res.run_dir.iterdir()} == EXPECTED_TOP
    assert res.ok and res.verification.ok and res.verification.mode == MODE_EMBEDDED
    assert not res.verification.failures and not res.verification.warnings
    assert res.summary["ended_reason"] == "QUIESCENT" and res.summary["problems"] == []
    assert (res.run_dir / "internal" / "events.sqlite").is_file()  # full history retained
    saved = load(res.run_dir, "verification.json")
    assert saved["ok"] and saved["n_failed"] == 0 and saved["source_mode"] == MODE_EMBEDDED


def test_input_manifest_keeps_the_contract_and_records_provenance(built):
    _, res, _ = built
    m = load(res.run_dir, "input_manifest.json")
    InputManifest.model_validate({k: m[k] for k in InputManifest.model_fields})  # contract intact
    assert m["bundle_format"] == "iceclaw-bundle/2" and m["mode"] == "scripted-test"
    assert m["derivation_version"] == "iceclaw-derive/2"
    assert m["transcript_profile"] == "full-private" and m["bundle_dir"] == "inv-bundle-test"
    assert m["source_mode"] == "upstream_published_example" and len(m["prediction_sha256"]) == 64
    assert m["llm_provider"] == "scripted" and m["code_revision"]
    assert set(m["visible_modalities"]) == {
        "nasa_ibm_prediction_raw", "pixelwise_shadow_mask", "pixelwise_temperature", "pixelwise_terrain"
    }  # fmt: skip
    assert m["source_package"]["portable"] and len(m["source_package"]["package_sha256"]) == 64
    assert set(m["integrity"]) == set(ex.INTEGRITY_FILES)
    assert m["limitations"] and m["budget"]["max_llm_calls"] == 28
    tools = {t["tool_id"] for t in m["tools_executed"]}
    assert {"thermal_psr_consistency", "terrain_slope_summary", "peak_location"} <= tools
    for t in m["tools_executed"]:
        assert t["parameters"] is not None and t["tool_run_id"] and t["agent_id"]


def test_model_usage_comes_from_recorded_calls_and_cost_is_unpriced(built):
    _, res, _ = built
    m = load(res.run_dir, "input_manifest.json")
    mu = m["model_usage"]
    rows = [json.loads(x) for x in (res.run_dir / "llm_calls.jsonl").read_text().splitlines()]
    assert mu["calls"] == len(rows) == res.summary["llm_calls"]
    assert mu["total_tokens"] == sum(r["total_tokens"] for r in rows) == res.summary["total_tokens"]
    assert mu["cost"]["status"] == "unpriced" and "usd" not in json.dumps(mu).lower()
    assert sum(a["calls"] for a in mu["by_agent"].values()) == mu["calls"]
    assert len(mu["per_call"]) == mu["calls"]
    assert res.summary["tokens_by_agent"] and res.summary["cost"]["status"] == "unpriced"
    assert all("request_json" in r and r["provider"] == "scripted" for r in rows)  # transcript kept


def test_evidence_separates_model_prediction_observation_calculation_and_ai_hypothesis(built):
    _, res, _ = built
    ev = load(res.run_dir, "evidence.json")
    classes = {m["tool_id"] + ":" + m["evidence_class"] for m in ev["measurements"]}
    assert "thermal_psr_consistency:calculation_on_lunar_observation" in classes
    assert "terrain_slope_summary:calculation_on_lunar_observation" in classes
    assert "peak_location:calculation_on_nasa_ibm_model_output" in classes
    for m in ev["measurements"]:  # class is driven by the source file, not by the agent
        obs = all(f.startswith("sombench:") for f in m["source_file_ids"])
        assert (m["evidence_class"] == "calculation_on_lunar_observation") == obs
        assert m["record"]["tool_run_id"] and m["record"]["source_sha256"]
    assert ev["statements"] and all(
        "AI HYPOTHESIS" in s["epistemic_label"] or "RUNTIME NOTE" in s["epistemic_label"]
        for s in ev["statements"]
    )
    assert {"nasa_ibm_model_prediction", "lunar_observation"} <= set(ev["epistemic_labels"])
    claim = next(s for s in ev["statements"] if s["kind"] == "claim" and s["agent_id"] == "thermal")
    assert claim["cited_evidence_event_ids"] and claim["cited_source_file_ids"]


def test_mixed_unit_measurements_carry_the_correct_unit_per_field():
    assert ex.field_unit("thermal_psr_consistency", "psr_mean_k", "mixed") == "K"
    assert ex.field_unit("thermal_psr_consistency", "psr_count", "mixed") == "pixels"
    assert (
        ex.field_unit("thermal_psr_consistency", "psr_below_cold_trap_fraction", "mixed")
        == "dimensionless"
    )
    assert ex.field_unit("terrain_slope_summary", "p95_deg", "mixed") == "deg"
    assert ex.field_unit("terrain_slope_summary", "steep_fraction", "mixed") == "dimensionless"
    disc = "mixed: *_count/_row/_col in pixels; *_gradient in prospectivity_index/m"
    assert (
        ex.field_unit("discontinuity_candidates", "cand0_gradient", disc) == "prospectivity_index/m"
    )
    assert (
        ex.field_unit("discontinuity_candidates", "threshold_gradient", disc)
        == "prospectivity_index/m"
    )
    assert ex.field_unit("discontinuity_candidates", "n_candidates_count", disc) == "pixels"
    assert ex.field_unit("discontinuity_candidates", "cand2_row", disc) == ex.PIXEL_INDEX
    assert ex.field_unit("peak_location", "row", "pixels") == ex.PIXEL_INDEX
    assert ex.field_unit("peak_location", "edge_distance_px", "pixels") == "pixels"
    assert ex.field_unit("valid_pixel_summary", "max", "K") == "K"
    assert ex.field_unit("gradient_stats", "p95", "deg/m") == "deg/m"
    assert ex.field_unit("missingness", "valid_pixels", "pixels") == "pixels"


def test_exported_fields_have_per_field_units_in_the_bundle(built):
    _, res, _ = built
    ev = load(res.run_dir, "evidence.json")
    th = next(m for m in ev["measurements"] if m["tool_id"] == "thermal_psr_consistency")
    units = {f["name"]: f["unit"] for f in th["fields"]}
    assert units["psr_mean_k"] == "K" and units["psr_count"] == "pixels"
    assert units["psr_below_cold_trap_fraction"] == "dimensionless"
    assert th["record"]["unit"].startswith("mixed")  # the single-unit record is preserved verbatim
    dc = next(m for m in ev["measurements"] if m["tool_id"] == "discontinuity_candidates")
    for f in dc["fields"]:
        assert f["unit"] != dc["record"]["unit"] or f["unit"] == "pixels"  # never the mixed string


def test_report_sections_labels_and_no_ice_claims(built):
    _, res, _ = built
    text = (res.run_dir / "report.md").read_text()
    for h in ("## 1. Question", "## 3. Available inputs", "## 4. Observed numeric measurements",
              "## 5. Model-prediction observations", "## 6. What the AI specialists concluded",
              "## 7. Cross-agent exchange", "## 8. Alternative explanations",
              "## 9. Outstanding uncertainties", "## 10. Research follow-up priority",
              "## 11. Provenance"):  # fmt: skip
        assert h in text, h
    assert ex.DISCLAIMER in text and "SCRIPTED TEST RUN" in text
    assert "NASA-IBM MODEL PREDICTION" in text and "LUNAR OBSERVATION DATA" in text
    assert "AI HYPOTHESIS" in text and "DETERMINISTIC CALCULATION" in text
    assert "NOT a physical probability of ice" in text or "not a calibrated probability" in text
    assert "python -m iceclaw.cli verify --run inv-bundle-test" in text
    assert "cost: **unpriced**" in text
    import re

    assert not re.search(r"\bPRO\b", text)


def test_report_states_plainly_when_no_challenge_or_revision_happened(built):
    _, res, _ = built
    text = (res.run_dir / "report.md").read_text()
    assert "No challenge and no revision occurred in this run" in text
    assert load(res.run_dir, "evidence.json")["summary"]["no_challenge_or_revision"] is True


def test_rankings_are_deterministic_labelled_and_not_probabilities(built):
    _, res, _ = built
    lines = (res.run_dir / "rankings.csv").read_text().splitlines()
    assert lines[0].split(",") == ex.RANK_COLUMNS
    import csv

    rows = list(csv.DictReader((res.run_dir / "rankings.csv").read_text().splitlines()))
    assert rows and [int(r["rank"]) for r in rows] == list(range(1, len(rows) + 1))
    keys = [
        (-int(r["n_distinct_agents"]), -int(r["n_evidence_events"]), r["target_id"]) for r in rows
    ]
    assert keys == sorted(keys)
    assert all("NOT a physical probability of ice" in r["note"] for r in rows)
    assert any(r["target_id"].startswith("window:4-20:4-20") for r in rows)
    assert not any("prob" in c.lower() for c in ex.RANK_COLUMNS)
    # same inputs -> byte-identical re-derivation
    geo = pkg.read_manifest(res.run_dir / "source")["geometry"]
    ev = load(res.run_dir, "evidence.json")
    rederived = ex.rankings_csv(ex.build_rankings(ev, geo["projection"], geo["grid"]))
    assert (res.run_dir / "rankings.csv").read_text() == rederived


def test_build_rankings_coverage_rule_on_crafted_evidence():
    def meas(eid, agent, tool, fields, window=None):
        params = {"modality": "nasa_ibm_prediction_raw"}
        if window:
            params["window"] = window
        return {"event_id": eid, "agent_id": agent, "tool_id": tool, "layers": ["nasa_ibm_prediction_raw"],
                "parameters": params, "unavailable_reason": None,
                "fields": [{"name": k, "value": v, "unit": "x"} for k, v in fields.items()]}  # fmt: skip

    ev = {"measurements": [
        meas("evt-1", "a", "peak_location", {"row": 10.0, "col": 10.0, "edge_distance_px": 10.0}),
        meas("evt-2", "b", "valid_pixel_summary", {"max": 1.0}, window=[0, 20, 0, 20]),
        meas("evt-3", "c", "peak_location", {"row": 50.0, "col": 50.0, "edge_distance_px": 5.0}),
    ]}  # fmt: skip
    proj = {"kind": "polar-stereographic", "latitudeOfOriginDeg": -90, "centralMeridianDeg": 0,
            "scaleFactor": 0.994, "falseEastingM": 500000.0, "falseNorthingM": 500000.0, "radiusM": 1737400.0}  # fmt: skip
    grid = {"xMinM": 320000.0, "yMaxM": 740000.0, "pixelSizeXM": 240.0, "pixelSizeYM": 240.0}
    rows = ex.build_rankings(ev, proj, grid)
    by = {r["target_id"]: r for r in rows}
    p1 = by["peak:nasa_ibm_prediction_raw:10:10"]
    assert (
        p1["n_distinct_agents"] == 2 and p1["evidence_event_ids"] == "evt-1;evt-2"
    )  # peak + window
    assert by["peak:nasa_ibm_prediction_raw:50:50"]["n_distinct_agents"] == 1
    assert rows[0]["target_id"] in (p1["target_id"], "window:0-20:0-20")  # 2-agent targets first
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert all("probability" not in r["basis"] for r in rows)


def test_check_numbers_rules():
    allowed = [65536.0, 8.73401, -107.774, 0.065367, 110.0]
    r = ex.check_numbers(
        "all 65,536 pixels; mean 8.734; delta −107.774 K; 110 K; 6.54% ok", allowed
    )
    assert (
        r["unmatched"] == [] and r["checked"] >= 5
    )  # thousands separator, rounding, unicode minus, %
    bad = ex.check_numbers("mean 9.123 and 4242 pixels", allowed)
    assert bad["unmatched"] == ["9.123", "4242"]
    assert (
        ex.check_numbers("evt-0001 and 32x32 and k_mad=5", allowed)["checked"] == 0
    )  # ids / short ints


def test_event_log_is_exported_exactly_as_recorded(built):
    _, res, _ = built
    sql = [
        json.loads(e.model_dump_json())
        for e in __import__("iceclaw.swarm.store", fromlist=["x"]).EvidenceStore.open_readonly(
            res.run_dir / "internal" / "events.sqlite"
        )
    ]
    out = [json.loads(x) for x in (res.run_dir / "events.jsonl").read_text().splitlines()]
    assert out == sql  # no reconstruction, no reordering, no added or dropped events


# ======================================================================= verification
def test_verify_passes_and_exit_code_zero_via_cli(built, capsys):
    _, res, _ = built
    rep = verify_bundle(res.run_dir)
    assert rep.ok and rep.mode == MODE_EMBEDDED
    assert main(["verify", "--run", str(res.run_dir)]) == 0
    out = capsys.readouterr().out
    assert (
        "RESULT: VERIFIED" in out
        and "recompute_measurements" in out
        and "embedded source package" in out
    )
    assert main(["verify", "--run", str(res.run_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_verify_unknown_run_exit_2(capsys):
    assert main(["verify", "--run", "/nonexistent/run"]) == 2
    assert "RUN_NOT_FOUND" in capsys.readouterr().err


def test_tampered_source_raster_is_detected_and_cli_exits_nonzero(bundle, capsys):
    f = next((bundle / "source" / "files").glob("*TMAX.tif"))
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 0xFF
    f.write_bytes(bytes(b))
    rep = verify_bundle(bundle)
    assert not rep.ok and "source_package_integrity" in failed_ids(rep)
    assert main(["verify", "--run", str(bundle)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_source_swapped_and_package_rehashed_still_fails_recomputation(bundle):
    """Attacker replaces the slope raster, fixes the package manifest AND the bundle manifest."""
    import numpy as np
    from bundle_helpers import write_tif

    f = next((bundle / "source" / "files").glob("*SLOPE.tif"))
    write_tif(
        f,
        np.full((32, 32), 3.0),
        json.loads((bundle / "source" / "package_manifest.json").read_text())["geometry"]["grid"],
    )
    pm_path = bundle / "source" / "package_manifest.json"
    pm = json.loads(pm_path.read_text())
    for x in pm["files"]:
        if x["file"].endswith("SLOPE.tif"):
            x["sha256"], x["size"] = pkg.sha256_bytes(f.read_bytes()), f.stat().st_size
    body = {k: v for k, v in pm.items() if k != "package_sha256"}
    pm["package_sha256"] = pkg.sha256_bytes(pkg.canonical(body).encode())
    pm_path.write_text(json.dumps(pm))
    m = load(bundle, "input_manifest.json")
    m["source_package"]["package_sha256"] = pm["package_sha256"]
    for fid, v in m["source_package"]["files"].items():
        if fid.endswith("SLOPE.tif"):
            v["sha256"] = next(x["sha256"] for x in pm["files"] if x["file_id"] == fid)
    (bundle / "input_manifest.json").write_text(json.dumps(m, indent=1))
    rep = verify_bundle(bundle)
    assert not rep.ok
    assert failed_ids(rep) & {"recompute_measurements", "geometry_units_consistent"}


def test_edited_recorded_number_fails_even_when_hashes_are_recomputed(bundle):
    p = bundle / "evidence.json"
    ev = json.loads(p.read_text())
    th = next(m for m in ev["measurements"] if m["tool_id"] == "thermal_psr_consistency")
    th["record"]["value"]["psr_mean_k"] += 5.0
    next(f for f in th["fields"] if f["name"] == "psr_mean_k")["value"] += 5.0
    p.write_text(json.dumps(ev, indent=1))
    assert "integrity_hashes" in failed_ids(verify_bundle(bundle))  # naive edit: caught by hash
    rehash(bundle, "evidence.json")
    rep = verify_bundle(bundle)
    assert "integrity_hashes" not in failed_ids(rep)
    assert "recompute_measurements" in failed_ids(rep)  # smart edit: caught by recomputation


def test_fabricated_event_parent_is_detected(bundle, capsys):
    p = bundle / "events.jsonl"
    lines = p.read_text().splitlines()
    ev = json.loads(lines[-2])
    ev.update(event_id="evt-9001", event_kind="challenge", parent_event_ids=["evt-9999"],
              claim_or_request="fabricated challenge", agent_id="skeptic")  # fmt: skip
    lines.insert(len(lines) - 1, json.dumps(ev))
    p.write_text("\n".join(lines) + "\n")
    rehash(bundle, "events.jsonl")  # even with matching hashes
    rep = verify_bundle(bundle)
    assert not rep.ok and "event_parents" in failed_ids(rep)
    assert main(["verify", "--run", str(bundle)]) == 1


def test_event_that_cites_a_later_event_as_parent_is_detected(bundle):
    p = bundle / "events.jsonl"
    lines = p.read_text().splitlines()
    first = json.loads(lines[0])
    first["parent_event_ids"] = [json.loads(lines[3])["event_id"]]
    first["event_kind"] = "claim"
    first["claim_or_request"] = "backdated"
    first.pop("tool_run_id", None)
    first["tool_id"] = None
    lines[0] = json.dumps(first)
    p.write_text("\n".join(lines) + "\n")
    rehash(bundle, "events.jsonl")
    assert "event_parents" in failed_ids(verify_bundle(bundle))


def test_removed_event_breaks_parent_chain(bundle):
    p = bundle / "events.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[1:]) + "\n")  # drop the first measurement
    rehash(bundle, "events.jsonl")
    assert failed_ids(verify_bundle(bundle)) & {"event_parents", "measurement_events_linked"}


def test_edited_agent_text_is_detected_by_rederivation(bundle):
    p = bundle / "events.jsonl"
    lines = p.read_text().splitlines()
    for i, ln in enumerate(lines):
        e = json.loads(ln)
        if e["event_kind"] == "claim" and e["agent_id"] == "thermal":
            e["claim_or_request"] += " Ice was confirmed beneath the crater."
            lines[i] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n")
    rehash(bundle, "events.jsonl")
    rep = verify_bundle(bundle)
    assert "evidence_rederived" in failed_ids(rep) and "report_rederived" in failed_ids(rep)


def test_edited_report_or_rankings_are_detected_even_when_rehashed(bundle):
    r = bundle / "report.md"
    r.write_text(r.read_text().replace("psr_mean_k |", "psr_mean_k |  ", 1) + "\nEdited.\n")
    c = bundle / "rankings.csv"
    lines = c.read_text().splitlines()
    c.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
    rehash(bundle, "report.md", "rankings.csv")
    rep = verify_bundle(bundle)
    assert {"report_rederived", "rankings_rederived"} <= failed_ids(rep)


def test_metadata_tampering_units_and_grid(bundle):
    pm_path = bundle / "source" / "package_manifest.json"
    pm = json.loads(pm_path.read_text())
    for x in pm["files"]:
        if x["modality"] == "pixelwise_temperature":
            x["unit"] = "degC"  # inconsistent unit
    pm["geometry"]["footprint_lat_lon_NW_NE_SE_SW"][0][0] += 0.25  # inconsistent footprint
    body = {k: v for k, v in pm.items() if k != "package_sha256"}
    pm["package_sha256"] = pkg.sha256_bytes(pkg.canonical(body).encode())
    pm_path.write_text(json.dumps(pm))
    m = load(bundle, "input_manifest.json")
    m["source_package"]["package_sha256"] = pm["package_sha256"]
    (bundle / "input_manifest.json").write_text(json.dumps(m, indent=1))
    rep = verify_bundle(bundle)
    assert not rep.ok and failed_ids(rep) & {"geometry_units_consistent", "recompute_measurements"}


def test_package_identity_is_content_addressed(bundle):
    pm_path = bundle / "source" / "package_manifest.json"
    pm = json.loads(pm_path.read_text())
    pm["region"]["source_split"] = "train"  # edit without fixing package_sha256
    pm_path.write_text(json.dumps(pm))
    rep = verify_bundle(bundle)
    assert "source_package_integrity" in failed_ids(rep)
    assert "MANIFEST_TAMPERED" in " ".join(c.detail for c in rep.failures)


def test_usage_tampering_budget_and_forbidden_content(bundle):
    m = load(bundle, "input_manifest.json")
    m["model_usage"]["total_tokens"] += 1000
    (bundle / "input_manifest.json").write_text(json.dumps(m, indent=1))
    assert "usage_and_budgets" in failed_ids(verify_bundle(bundle))


def test_forbidden_content_is_flagged(bundle):
    p = bundle / "events.jsonl"
    lines = p.read_text().splitlines()
    e = json.loads(lines[-2])
    e["claim_or_request"] = (
        "see the PRO target at /home/someone/data and key sk-ABCDEFGHIJKLMNOP123456"
    )
    lines[-2] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n")
    rehash(bundle, "events.jsonl")
    rep = verify_bundle(bundle)
    assert "no_forbidden_content" in failed_ids(rep)
    d = next(c.detail for c in rep.failures if c.id == "no_forbidden_content")
    assert "PRO" in d and "API-key-like" in d and "absolute path" in d


def test_missing_manifest_or_file_is_reported(bundle):
    (bundle / "evidence.json").unlink()
    rep = verify_bundle(bundle)
    assert not rep.ok and failed_ids(rep) == {"files_present"}
    (bundle / "input_manifest.json").write_text("{not json")
    (bundle / "evidence.json").write_text("{}")
    assert "manifest_schema" in failed_ids(verify_bundle(bundle))


def test_affirmative_ice_claims_warn_but_negations_do_not(bundle):
    p = bundle / "events.jsonl"
    lines = p.read_text().splitlines()
    for i, ln in enumerate(lines):
        e = json.loads(ln)
        if e["event_kind"] == "claim" and e["agent_id"] == "geology":
            e["claim_or_request"] = (
                "Slopes are moderate. Water ice was detected in the shadow. This does not confirm ice."
            )
            lines[i] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n")
    rep = verify_bundle(bundle)
    warn = [c for c in rep.warnings if c.id == "no_affirmative_ice_claim"]
    assert warn and "detected" in warn[0].detail and "confirm" not in warn[0].detail


# ------------------------------------------------------------- portability / source modes
def test_bundle_without_embedded_source_is_labelled_requires_local_source(bundle, capsys):
    shutil.rmtree(bundle / "source")
    rep = verify_bundle(bundle)
    assert not rep.ok and rep.mode == MODE_LOCAL
    assert "source_available" in failed_ids(rep)
    assert "requires local source data" in rep.render()
    assert main(["verify", "--run", str(bundle)]) == 1


def test_external_content_addressed_source_package(built, bundle):
    _, res, _ = built
    external = bundle.parent / "external_pkg"
    shutil.copytree(res.run_dir / "source", external)
    shutil.rmtree(bundle / "source")
    rep = verify_bundle(bundle, source_package=external)
    assert rep.ok and rep.mode == MODE_EXTERNAL
    # a different package with a different content address is rejected
    f = next((external / "files").glob("*LPSR.tif"))
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    rep2 = verify_bundle(bundle, source_package=external)
    assert not rep2.ok and "source_package_integrity" in failed_ids(rep2)


def test_independent_process_verifies_without_key_catalog_or_gpu(built, tmp_path):
    """A second Python process, clean env, other cwd, no data roots: verification only."""
    _, res, _ = built
    iso = tmp_path / "downloaded"
    shutil.copytree(res.run_dir, iso, ignore=shutil.ignore_patterns("internal"))
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "ICECLAW_SKIP_DOTENV": "1",
        "ICECLAW_MOONATLAS_ROOTS": "/nonexistent",
        "ICECLAW_DATA_ROOT": "/nonexistent",
        "ICECLAW_SOMBENCH_ROOT": "/nonexistent",
    }
    assert "OPENAI_API_KEY" not in env
    cp = subprocess.run([sys.executable, "-m", "iceclaw.cli", "verify", "--run", str(iso)],
                        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)  # fmt: skip
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert (
        "RESULT: VERIFIED" in cp.stdout
        and "embedded source package (offline portable)" in cp.stdout
    )
    # and the same process fails (exit 1) on a tampered copy
    f = next((iso / "source" / "files").glob("*SLOPE.tif"))
    f.write_bytes(f.read_bytes()[:-2] + b"zz")
    cp2 = subprocess.run([sys.executable, "-m", "iceclaw.cli", "verify", "--run", str(iso)],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)  # fmt: skip
    assert cp2.returncode == 1 and "FAILED" in cp2.stdout


def test_source_package_never_contains_evaluator_material(built):
    _, res, _ = built
    names = [p.name for p in (res.run_dir / "source").rglob("*") if p.is_file()]
    assert sorted(names) == ["package_manifest.json", "synthetic-prediction-raw.webp",
                             "synthetic_LPSR.tif", "synthetic_SLOPE.tif", "synthetic_TMAX.tif"]  # fmt: skip
    pm = pkg.read_manifest(res.run_dir / "source")
    assert any("expert target" in x for x in pm["excluded_by_policy"])
    assert "ice-sectors.json" not in " ".join(names)
    assert pm["region"]["upstream_sectors_note"].startswith("NOT included")
    with pytest.raises(pkg.PackageError) as e:
        spec = synthetic_spec(res.run_dir.parent / "src_forbidden")
        bad = Path(res.run_dir.parent / "src_forbidden" / "synthetic_PRO.tif")
        shutil.copy(spec.files[1].path, bad)
        spec.files[1] = pkg.SourceFile(spec.files[1].modality, spec.files[1].kind, bad,
                                       spec.files[1].file_id, "K", {}, "x")  # fmt: skip
        pkg.write_package(spec, res.run_dir.parent / "out_forbidden")
    assert e.value.code == "FORBIDDEN_SOURCE"


# ============================================================================== replay
def test_replay_uses_saved_events_only_and_makes_no_model_call(built, monkeypatch, capsys):
    svc, res, _ = built
    from iceclaw.agents import provider

    def boom(*a, **k):
        raise AssertionError("replay must not construct any model client")

    monkeypatch.setattr(provider, "build_openai_client", boom)
    monkeypatch.setattr(provider.ScriptedChatClient, "__init__", boom)
    monkeypatch.setattr(provider.InstrumentedOpenAIChatClient, "__init__", boom)
    r = svc.replay(res.run_dir)
    assert r.model_calls_made == 0 and r.n_events == res.summary["events"]
    assert "RECORDED RUN" in r.header and "no model calls are made" in r.header
    assert len(r.lines) == r.n_events and all("[evt-" in line for line in r.lines)
    assert "thermal" in r.lines[0] or "geology" in r.lines[0] or "prediction_auditor" in r.lines[0]
    assert main(["replay", "--run", str(res.run_dir)]) == 0
    out = capsys.readouterr().out
    assert "RECORDED RUN" in out and "0 model calls made" in out
    assert main(["investigate", "--region", "synthetic-region", "--replay", str(res.run_dir)]) == 0
    assert "RECORDED RUN" in capsys.readouterr().out


def test_replay_of_a_bundle_matches_replay_of_its_sqlite_history(built):
    svc, res, _ = built
    a = svc.replay(res.run_dir)  # bundle events.jsonl
    b = svc.replay(res.run_dir / "internal")  # sqlite only
    assert a.lines == b.lines and a.n_events == b.n_events


# ======================================================================= export recorded
def test_export_recorded_run_is_read_only_and_verifies(built):
    svc, res, root = built
    internal = res.run_dir / "internal"
    before = {p.name: pkg.sha256_bytes(p.read_bytes()) for p in internal.iterdir()}
    out = svc.export_recorded(internal, out_id="inv-exported-copy")
    assert before == {p.name: pkg.sha256_bytes(p.read_bytes()) for p in internal.iterdir()}
    assert out.ok and out.verification.mode == MODE_EMBEDDED
    m = load(out.run_dir, "input_manifest.json")
    assert m["mode"] == "recorded-export" and any(
        "without any new model call" in n for n in m["notes"]
    )
    assert (out.run_dir / "events.jsonl").read_text() == (res.run_dir / "events.jsonl").read_text()
    assert "RECORDED RUN" in (out.run_dir / "report.md").read_text()
    assert main(["export", "--run", "/nonexistent"]) == 2


# ====================================================================== service / CLI
def test_investigate_without_key_gives_clear_error_and_exit_2(capsys):
    assert main(["investigate", "--region", "ice-s-0001-0002"]) == 2
    err = capsys.readouterr().err
    assert "LIVE MODE UNAVAILABLE" in err and "MISSING_API_KEY" in err


def test_cli_and_service_share_one_implementation():
    import inspect

    from iceclaw import cli, services

    src = inspect.getsource(cli)
    assert "InvestigationService" in src
    for forbidden in ("run_tool(", "SwarmRuntime(", "verify_bundle(", "export_bundle("):
        assert forbidden not in src  # the CLI only calls the service
    assert hasattr(services.InvestigationService, "investigate")
    assert hasattr(services.InvestigationService, "verify")


def test_input_token_ceiling_stops_the_run_and_the_partial_bundle_still_verifies(tmp_path):
    svc = scripted_service(tmp_path / "runs", tmp_path / "src")
    res = svc.investigate("synthetic-region", run_id="inv-capped", max_input_tokens=50)
    assert res.summary["ended_reason"] == "BUDGET_EXHAUSTED"
    assert any(
        (r or "").startswith("BUDGET_INPUT_TOKENS") for r in res.summary["exit_reasons"].values()
    )
    m = load(res.run_dir, "input_manifest.json")
    assert m["prompt_policy"]["max_input_tokens"] == 50
    assert res.verification.ok  # truncated runs still produce a consistent, verifiable bundle
    assert m["model_usage"]["calls"] < 20


def test_region_gate_for_unknown_or_unpaired_regions(tmp_path):
    from iceclaw.config import Settings
    from iceclaw.data.catalog import RegionNotSupported
    from iceclaw.services import InvestigationService

    svc = InvestigationService(Settings(data_root=Path("/none"), runs_root=tmp_path / "r",
                                        moonatlas_roots=(Path("/none"),)), client_factory=lambda r, a: None)  # fmt: skip
    with pytest.raises(RegionNotSupported):
        svc.investigate("ice-s-9999-9999")
    assert not list((tmp_path / "r").glob("*/source"))  # nothing was written for a refused region
    _ = rich_policies


# =========================================================================== real data
REPO = Path(__file__).resolve().parents[1]
SMOKE = Path(
    os.environ.get(
        "ICECLAW_MOONATLAS_SMOKE_ROOT", REPO / ".upstream/moonatlas-science/examples/science-smoke"
    )
)
SOM = Path(
    os.environ.get("ICECLAW_SOMBENCH_TEST_ROOT", REPO / "data/sombench/prospectivity_dataset")
)
P5_RUN = REPO / "runs" / "inv-20261031T192018Z-p5live"
needs_real = pytest.mark.skipif(
    not ((SMOKE / "manifest.json").is_file() and (SOM / "download_manifest.json").is_file()),
    reason="real MOONATLAS / SomBench data not present",
)


def real_settings(runs: Path):
    from iceclaw.config import Settings

    return Settings(
        data_root=Path("/none"), runs_root=runs, moonatlas_roots=(SMOKE,), sombench_root=SOM
    )


@needs_real
def test_real_region_package_decodes_identically_to_the_catalog_and_excludes_pro(tmp_path):
    import numpy as np

    from iceclaw.data.catalog import discover

    cat = discover(real_settings(tmp_path))
    spec = pkg.spec_from_catalog_record(cat.get("ice-s-0001-0002"))
    m = pkg.write_package(spec, tmp_path / "pkg")
    loaded = pkg.load_package(tmp_path / "pkg")
    cat_arrays = cat.load_arrays("ice-s-0001-0002")
    for name, layer in cat_arrays.layers.items():
        got = loaded.arrays.layers[name]
        assert np.array_equal(got.values, layer.values) and np.array_equal(
            got.covered, layer.covered
        )
        assert (
            got.sha256 == layer.sha256 and got.unit == layer.unit and got.file_id == layer.file_id
        )
    names = sorted(p.name for p in (tmp_path / "pkg" / "files").iterdir())
    assert names == ["ice-s-0001-0002-raw.webp", "patch_0001_0002_S_80S_LPSR.tif",
                     "patch_0001_0002_S_80S_SLOPE.tif", "patch_0001_0002_S_80S_TMAX.tif"]  # fmt: skip
    assert "PRO" not in " ".join(names) and "ice-sectors" not in json.dumps(m)
    text = json.dumps(m)
    import re

    assert not re.search(r"\bPRO\b", text) and "reference_path" not in text
    assert "CC BY 4.0" in text and "not a fresh IceClaw inference" in text
    assert pkg.write_package(spec, tmp_path / "pkg2")["package_sha256"] == m["package_sha256"]
    assert sum(p.stat().st_size for p in (tmp_path / "pkg" / "files").iterdir()) < 1_000_000


@needs_real
def test_real_region_scripted_investigation_exports_and_verifies(tmp_path):
    from iceclaw.agents.provider import ScriptedChatClient
    from iceclaw.services import InvestigationService

    pol = rich_policies()

    def factory(rt, aid):
        c = ScriptedChatClient(pol[aid], aid)
        c.bind(rt, aid)
        return c

    svc = InvestigationService(real_settings(tmp_path / "runs"), client_factory=factory)
    res = svc.investigate("ice-s-0001-0002", run_id="inv-real-scripted")
    assert res.ok, res.verification.render()
    ev = load(res.run_dir, "evidence.json")
    th = next(m for m in ev["measurements"] if m["tool_id"] == "thermal_psr_consistency")
    vals = {f["name"]: f["value"] for f in th["fields"]}
    assert vals["psr_mean_k"] == pytest.approx(139.21508, abs=1e-4)  # the P4 real number
    assert vals["psr_minus_non_psr_k"] == pytest.approx(-107.77419, abs=1e-4)
    assert set(th["source_file_ids"]) == {"sombench:patch_0001_0002_S_80S_LPSR.tif",
                                          "sombench:patch_0001_0002_S_80S_TMAX.tif"}  # fmt: skip
    assert verify_bundle(res.run_dir).ok
    # observation-only tiles stay refused
    from iceclaw.data.catalog import PredictionUnavailable

    with pytest.raises(PredictionUnavailable):
        svc.investigate("ice-s-0003-0002")


@needs_real
@pytest.mark.skipif(not P5_RUN.is_dir(), reason="the recorded P5 live run is not present")
def test_the_preserved_p5_live_run_exports_verifies_and_stays_unchanged(tmp_path):
    before = {p.name: pkg.sha256_bytes(p.read_bytes()) for p in P5_RUN.iterdir() if p.is_file()}
    from iceclaw.services import InvestigationService

    svc = InvestigationService(real_settings(tmp_path / "runs"), client_factory=None)
    out = svc.export_recorded(P5_RUN, out_id="inv-p5-export")
    assert before == {
        p.name: pkg.sha256_bytes(p.read_bytes()) for p in P5_RUN.iterdir() if p.is_file()
    }
    assert out.ok, out.verification.render()
    m = load(out.run_dir, "input_manifest.json")
    assert m["llm_provider"] == "openai" and m["model_usage"]["calls"] == 22
    assert m["model_usage"]["total_tokens"] == 91049 and m["mode"] == "recorded-export"
    ev = load(out.run_dir, "evidence.json")
    assert len(ev["measurements"]) == 15 and ev["summary"]["challenges"] == 0
    assert ev["summary"]["events_by_kind"] == {"claim": 4, "completion": 5, "measurement": 15}
    rep = verify_bundle(out.run_dir)
    assert rep.ok and not rep.warnings, rep.render()
    assert svc.replay(out.run_dir).n_events == 24


# ====================================================== verifier robustness regressions
def test_key_detector_ignores_chance_sk_inside_opaque_base64_but_flags_real_looking_keys():
    from iceclaw.reports.verify import FORBIDDEN_PATTERNS, _scan_text

    blob = "A" * 1900 + "nTuh9Ysk-" + "Qr7_x" * 40 + "B" * 300  # provider protected_data-like
    assert "API-key-like string" not in _scan_text(f'{{"protected_data": "{blob}"}}')
    for key in ("sk-" + "a1B2" * 12, "sk-proj-" + "a1B2" * 12, "token=sk-" + "a1B2" * 12):
        assert "API-key-like string" in _scan_text(f"prefix {key} suffix")
    assert "API-key-like string" not in _scan_text("risk-assessment-methodology-for-screening")
    assert FORBIDDEN_PATTERNS["expert target PRO"].search("the PRO layer")


def test_derivation_version_is_stamped_and_enforced(bundle):
    m = load(bundle, "input_manifest.json")
    assert m["derivation_version"] == ex.DERIVATION_VERSION
    m["derivation_version"] = "iceclaw-derive/0"
    (bundle / "input_manifest.json").write_text(json.dumps(m, indent=1))
    rep = verify_bundle(bundle)
    assert not rep.ok and failed_ids(rep) == {"derivation_version"}
    assert "re-export" in rep.failures[0].detail


def test_verify_never_crashes_on_a_scrambled_event_log(bundle):
    p = bundle / "events.jsonl"
    p.write_text("\n".join(reversed(p.read_text().splitlines())) + "\n")
    rehash(bundle, "events.jsonl")
    rep = verify_bundle(bundle)  # must report, not raise
    assert not rep.ok and "event_parents" in failed_ids(rep)
    (bundle / "events.jsonl").write_text('{"event_id": "x"}\n')
    rehash(bundle, "events.jsonl")
    assert not verify_bundle(bundle).ok


def test_verify_write_refreshes_verification_json(bundle, capsys):
    (bundle / "verification.json").write_text('{"stale": true}')
    assert main(["verify", "--run", str(bundle), "--write"]) == 0
    assert load(bundle, "verification.json")["ok"] is True
    assert "wrote" in capsys.readouterr().out


# ===================================== richer exchanges: request->result and challenge->revision
def _run(tmp_path, policies, run_id):
    svc = scripted_service(tmp_path / "runs", tmp_path / "src", policies)
    return svc.investigate("synthetic-region", run_id=run_id)


def test_test_request_and_result_bundle_is_verified_readable_and_not_number_checked(tmp_path):
    from agent_helpers import EVT
    from bundle_helpers import first_event_id, one_tool_policy

    base = rich_policies()
    thermal_first = one_tool_policy("thermal_psr_consistency", cold_trap_k=110.0)
    asked = {"done": False}

    def skeptic(ctx):
        claims = [m[0] for m in EVT.findall(ctx.last_user_text) if m[2] == "claim"]
        if claims and not asked["done"]:
            asked["done"] = True
            return [("request_test", {"to_agent": "thermal", "tool_id": "thermal_psr_consistency",
                                      "rationale": "check the shadow contrast near the peak at 150 K",
                                      "params": {"cold_trap_k": 150.0, "psr_threshold": 0.5},
                                      "window": [2, 18, 2, 18], "cites": claims[:1]})]  # fmt: skip
        return "nothing further"

    def thermal(ctx):
        reqs = [m[0] for m in EVT.findall(ctx.last_user_text) if m[2] == "test_request"]
        if reqs:
            if not ctx.last_tool_results:
                return [("run_science_tool", {"tool_id": "thermal_psr_consistency",
                                              "in_response_to": reqs[-1], "cold_trap_k": 150.0,
                                              "psr_threshold": 0.5, "window": [2, 18, 2, 18]})]  # fmt: skip
            return "answered"
        return thermal_first(ctx)

    res = _run(tmp_path, {**base, "skeptic": skeptic, "thermal": thermal}, "inv-request")
    assert res.ok, res.verification.render()
    ev = load(res.run_dir, "evidence.json")
    assert ev["summary"]["test_requests"] == 1
    kinds = [x["kind"] for x in ev["exchange"]]
    assert "test_request" in kinds and "test_result" in kinds
    req = next(s for s in ev["statements"] if s["kind"] == "test_request")
    assert req["numeric_check"] is None  # requested parameters are inputs, not claims
    text = (res.run_dir / "report.md").read_text()
    assert "asks thermal to run thermal_psr_consistency" in text and "Rationale:" in text
    assert '"rationale"' not in text  # no raw JSON in the readable report
    assert "No challenge and no revision occurred" in text  # requests are not challenges
    _ = first_event_id


def test_challenge_and_linked_revision_bundle_verifies_and_is_reported(tmp_path):
    from test_agents import BASIC, skeptic_challenging, thermal_reacting

    pol = {**BASIC, "thermal": thermal_reacting(), "skeptic": skeptic_challenging()}
    res = _run(tmp_path, pol, "inv-debate")
    assert res.ok, res.verification.render()
    ev = load(res.run_dir, "evidence.json")
    assert ev["summary"]["challenges"] == 1 and ev["summary"]["revisions"] == 1
    assert ev["summary"]["no_challenge_or_revision"] is False
    text = (res.run_dir / "report.md").read_text()
    assert "No challenge and no revision occurred" not in text
    assert "**challenge**" in text and "**revision**" in text
    events = [json.loads(x) for x in (res.run_dir / "events.jsonl").read_text().splitlines()]
    claim = next(e for e in events if e["event_kind"] == "claim" and e["agent_id"] == "thermal")
    rev = next(e for e in events if e["event_kind"] == "revision")
    ch = next(e for e in events if e["event_kind"] == "challenge")
    assert (
        rev["parent_event_ids"][0] == claim["event_id"]
        and ch["event_id"] in rev["parent_event_ids"]
    )
    # a revision stripped of its new evidence is caught by the verifier
    b = copy_bundle(res.run_dir, tmp_path / "tampered")
    p = b / "events.jsonl"
    lines = p.read_text().splitlines()
    for i, ln in enumerate(lines):
        e = json.loads(ln)
        if e["event_kind"] == "revision":
            e["parent_event_ids"] = e["parent_event_ids"][:1]  # drop new evidence + challenge
            lines[i] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n")
    rehash(b, "events.jsonl")
    assert "reference_rules" in failed_ids(verify_bundle(b))
