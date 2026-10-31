"""P6 corrections: public-redacted bundles, versioned export contract, ranking labelling.

A scripted fake model emits provider-style opaque reasoning items (``protected_data``) so the
transcripts here really contain them. Synthetic source data; no model is called anywhere.
Real-data tests at the end skip if the preserved runs are absent.
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from bundle_helpers import copy_bundle, failed_ids, rehash, scripted_service

from iceclaw.agents.provider import OPAQUE_FIXTURE
from iceclaw.cli import main
from iceclaw.reports import export as ex
from iceclaw.reports.verify import MODE_EMBEDDED, opaque_violations, verify_bundle


def tree_hashes(d: Path) -> dict[str, str]:
    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def load(b: Path, name: str):
    return json.loads((b / name).read_text())


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    root = tmp_path_factory.mktemp("pub")
    svc = scripted_service(root / "runs", root / "src", emit_reasoning=True)
    res = svc.investigate("synthetic-region", run_id="inv-private")
    return svc, res, root


@pytest.fixture(scope="module")
def public(private):
    svc, res, _ = private
    before = tree_hashes(res.run_dir)
    pub = svc.publish(res.run_dir)
    assert tree_hashes(res.run_dir) == before  # the private run is untouched
    return svc, res, pub


# ============================================================ the private bundle is complete
def test_private_bundle_keeps_the_full_transcript_including_opaque_data(private):
    _, res, _ = private
    text = (res.run_dir / "llm_calls.jsonl").read_text()
    assert OPAQUE_FIXTURE in text  # the authentic private transcript is not altered
    assert opaque_violations(text)  # ...and the detector sees it
    assert res.ok
    assert load(res.run_dir, "input_manifest.json")["transcript_profile"] == "full-private"
    assert (res.run_dir / "internal" / "events.sqlite").is_file()


# ================================================================================= publishing
def test_publish_makes_a_verified_public_bundle_without_opaque_provider_data(public):
    _, res, pub = public
    assert pub.run_id == "inv-private-public" and pub.ok
    assert pub.verification.mode == MODE_EMBEDDED
    ids = {c.id for c in pub.verification.checks}
    assert {"transcript_profile", "public_transcript_redacted"} <= ids
    assert "public_bundle_no_internal_history" in ids
    text = (pub.run_dir / "llm_calls.jsonl").read_text()
    assert OPAQUE_FIXTURE not in text and ex.REDACTED_MARKER in text
    assert opaque_violations(text) == []
    assert "function_call" in text and "function_result" in text  # readable parts survive
    assert (pub.run_dir / "llm_calls.jsonl").stat().st_size < (
        res.run_dir / "llm_calls.jsonl"
    ).stat().st_size
    assert not (pub.run_dir / "internal").exists()


def test_published_bundle_preserves_events_measurements_source_and_usage(public):
    _, res, pub = public
    assert (pub.run_dir / "events.jsonl").read_bytes() == (
        res.run_dir / "events.jsonl"
    ).read_bytes()
    a, b = load(res.run_dir, "evidence.json"), load(pub.run_dir, "evidence.json")
    assert a["measurements"] == b["measurements"] and a["statements"] == b["statements"]
    assert tree_hashes(res.run_dir / "source") == tree_hashes(pub.run_dir / "source")
    ma, mb = load(res.run_dir, "input_manifest.json"), load(pub.run_dir, "input_manifest.json")
    assert ma["source_package"]["package_sha256"] == mb["source_package"]["package_sha256"]
    assert ma["model_usage"] == mb["model_usage"]  # tokens, calls and per-call table identical
    ra, rb = lines(res.run_dir / "llm_calls.jsonl"), lines(pub.run_dir / "llm_calls.jsonl")
    assert len(ra) == len(rb) > 0
    for x, y in zip(ra, rb, strict=True):
        for k in x:
            if not k.endswith("_json"):
                assert x[k] == y[k], k  # only the payload JSON differs, and only in opaque values
    redacted, stats = ex.redact_llm_rows(ra)
    assert redacted == rb and stats["fields_redacted"] > 0


def test_public_manifest_versions_the_contract_and_records_what_was_removed(public):
    _, res, pub = public
    m = load(pub.run_dir, "input_manifest.json")
    assert m["bundle_format"] == "iceclaw-bundle/2"
    assert m["derivation_version"] == "iceclaw-derive/2"
    assert m["transcript_profile"] == "public-redacted"
    assert m["bundle_dir"] == "inv-private-public"
    red = m["transcript_redactions"]
    assert red["fields_redacted"] > 0
    assert red["characters_removed"] == red["fields_redacted"] * len(OPAQUE_FIXTURE)
    assert red["keys"] == ["protected_data"] and red["marker"] == ex.REDACTED_MARKER
    priv = (res.run_dir / "llm_calls.jsonl").read_bytes()
    assert red["private_transcript_sha256"] == hashlib.sha256(priv).hexdigest()
    assert red["private_transcript_bytes"] == len(priv)
    assert red["source_bundle_format"] == "iceclaw-bundle/2"
    mp = (res.run_dir / "input_manifest.json").read_bytes()
    assert red["source_bundle_manifest_sha256"] == hashlib.sha256(mp).hexdigest()
    text = (pub.run_dir / "report.md").read_text()
    assert "public-redacted" in text and red["private_transcript_sha256"][:16] in text


def test_public_bundle_verifies_in_a_clean_process_without_key_or_catalog(public, tmp_path):
    _, _, pub = public
    iso = tmp_path / "downloaded"
    shutil.copytree(pub.run_dir, iso)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "ICECLAW_MOONATLAS_ROOTS": "/nonexistent",
        "ICECLAW_SOMBENCH_ROOT": "/nonexistent",
    }
    cp = subprocess.run(
        [sys.executable, "-m", "iceclaw.cli", "verify", "--run", str(iso)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "VERIFIED" in cp.stdout and "public_transcript_redacted" in cp.stdout


def test_reinserting_opaque_data_or_history_or_bad_metadata_fails_verification(public, tmp_path):
    _, _, pub = public
    b = copy_bundle(pub.run_dir, tmp_path / "t1")
    p = b / "llm_calls.jsonl"
    p.write_text(p.read_text().replace(ex.REDACTED_MARKER, OPAQUE_FIXTURE, 1))
    rehash(b, "llm_calls.jsonl")
    assert "public_transcript_redacted" in failed_ids(verify_bundle(b))

    b = copy_bundle(pub.run_dir, tmp_path / "t2")
    (b / "internal").mkdir()
    assert "public_bundle_no_internal_history" in failed_ids(verify_bundle(b))

    b = copy_bundle(pub.run_dir, tmp_path / "t3")
    m = load(b, "input_manifest.json")
    m["transcript_profile"] = "bogus"
    (b / "input_manifest.json").write_text(json.dumps(m, indent=1))
    assert not verify_bundle(b).ok

    b = copy_bundle(pub.run_dir, tmp_path / "t4")
    m = load(b, "input_manifest.json")
    m["transcript_redactions"]["private_transcript_sha256"] = "zz"
    (b / "input_manifest.json").write_text(json.dumps(m, indent=1))
    assert not verify_bundle(b).ok


def test_publish_refuses_unverified_or_already_public_bundles_and_leaves_nothing(public):
    svc, res, pub = public
    runs = svc.settings.runs_root
    bad = copy_bundle(res.run_dir, runs / "inv-tampered-src")
    f = next((bad / "source" / "files").glob("*TMAX.tif"))
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(ex.PublishRefused):
        svc.publish(bad, out_id="inv-should-not-exist")
    assert not (runs / "inv-should-not-exist").exists()  # no half-made public bundle
    with pytest.raises(ex.PublishRefused):
        svc.publish(pub.run_dir, out_id="inv-twice")
    assert not (runs / "inv-twice").exists()
    shutil.rmtree(bad)


def test_cli_publish(private, tmp_path, monkeypatch, capsys):
    _, res, _ = private
    monkeypatch.setenv("ICECLAW_RUNS_ROOT", str(tmp_path / "runs"))
    assert main(["publish", "--run", str(res.run_dir), "--out", "inv-cli-public"]) == 0
    out = capsys.readouterr().out
    assert "public-redacted" in out and "VERIFIED" in out
    assert main(["publish", "--run", str(tmp_path / "runs" / "inv-cli-public")]) == 1


# ============================================ rankings: attention order, ties, labels
def test_ties_are_flagged_and_alphabetical_and_v1_is_unchanged(private):
    _, res, _ = private
    ev = load(res.run_dir, "evidence.json")
    geo = load(res.run_dir, "input_manifest.json")["source_package"]
    pm = json.loads((res.run_dir / "source" / "package_manifest.json").read_text())
    proj, grid = pm["geometry"]["projection"], pm["geometry"]["grid"]
    del geo
    v2 = ex.build_rankings(ev, proj, grid, 2)
    v1 = ex.build_rankings(ev, proj, grid, 1)
    assert v2 and [r["target_id"] for r in v1] == [r["target_id"] for r in v2]
    assert all("tie_group_size" not in r and r["note"] == ex.RANK_NOTE_V1 for r in v1)
    groups: dict[tuple, list[str]] = {}
    for r in v2:
        groups.setdefault((r["n_distinct_agents"], r["n_evidence_events"]), []).append(
            r["target_id"]
        )
    for r in v2:
        size = len(groups[(r["n_distinct_agents"], r["n_evidence_events"])])
        assert r["tie_group_size"] == size
        assert (r["tie_break"] == ex.TIE_BREAK_TEXT) == (size > 1)
        assert "NOT an independently validated ice-prospectivity ranking" in r["note"]
    for ids in groups.values():
        assert ids == sorted(ids)  # ties are listed alphabetically
    assert ex.rankings_csv(v1, 1).splitlines()[0].split(",") == ex.RANK_COLUMNS_V1
    assert ex.rankings_csv(v2, 2).splitlines()[0].split(",") == ex.RANK_COLUMNS_V2
    assert ex.RANK_COLUMNS_V2 == [*ex.RANK_COLUMNS_V1, "tie_group_size", "tie_break"]


def test_new_bundles_label_rankings_as_investigation_attention(private):
    _, res, _ = private
    rows = list(csv.DictReader((res.run_dir / "rankings.csv").read_text().splitlines()))
    assert rows and list(rows[0].keys()) == ex.RANK_COLUMNS_V2
    for r in rows:
        assert "investigation-attention order" in r["note"]
        assert "NOT an independently validated ice-prospectivity ranking" in r["note"]
    text = (res.run_dir / "report.md").read_text()
    sec = text.split("## 10. ")[1].split("## 11. ")[0]
    assert "investigation attention, not an ice-prospectivity ranking" in sec
    assert "independently validated ice-prospectivity ranking" in sec
    assert "alphabetically by target id" in sec and "no importance" in sec
    assert "| Tie |" in sec and ex.DISCLAIMER in sec


# ================================================================= legacy /1 stays valid
def _v1_bundle(res, root: Path) -> Path:
    rec = ex.record_from_run_dir(res.run_dir / "internal", mode="scripted-test")
    out = root / "legacy-v1"
    shutil.copytree(res.run_dir / "source", out / "source")
    ex.export_bundle(rec, out, None, format_version=1)
    return out


def test_original_v1_contract_still_verifies_with_the_original_logic(private, tmp_path):
    _, res, _ = private
    out = _v1_bundle(res, tmp_path)
    rep = verify_bundle(out)
    assert rep.ok, rep.render()
    m = load(out, "input_manifest.json")
    assert (m["bundle_format"], m["derivation_version"]) == (
        "iceclaw-bundle/1",
        "iceclaw-derive/1",
    )
    assert "transcript_profile" not in m
    header = (out / "rankings.csv").read_text().splitlines()[0].split(",")
    assert header == ex.RANK_COLUMNS_V1
    text = (out / "report.md").read_text()
    assert "investigation attention" not in text and "| Tie |" not in text


def test_format_and_derivation_must_match_and_unknown_versions_are_refused(private, tmp_path):
    _, res, _ = private
    out = _v1_bundle(res, tmp_path)
    combos = (
        ("iceclaw-bundle/1", "iceclaw-derive/2"),
        ("iceclaw-bundle/2", "iceclaw-derive/1"),
        ("iceclaw-bundle/1", "iceclaw-derive/0"),
        ("iceclaw-bundle/9", "iceclaw-derive/1"),
    )
    for i, (fmt, der) in enumerate(combos):
        b = copy_bundle(out, tmp_path / f"m{i}")
        m = load(b, "input_manifest.json")
        m["bundle_format"], m["derivation_version"] = fmt, der
        (b / "input_manifest.json").write_text(json.dumps(m, indent=1))
        assert not verify_bundle(b).ok, (fmt, der)


def test_a_v1_bundle_can_be_published_as_v2(private, tmp_path):
    svc, res, _ = private
    out = _v1_bundle(res, tmp_path)
    before = tree_hashes(out)
    pub = svc.publish(out, out_id="inv-from-v1")
    assert tree_hashes(out) == before and pub.ok
    m = load(pub.run_dir, "input_manifest.json")
    assert m["bundle_format"] == "iceclaw-bundle/2"
    assert m["transcript_redactions"]["source_bundle_format"] == "iceclaw-bundle/1"
    assert m["transcript_redactions"]["fields_redacted"] > 0


# ============================================== the preserved real runs (skip if absent)
REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "runs" / "inv-20261031T201513Z-p6"
P5_BUNDLE = REPO / "runs" / "inv-20261031T192018Z-p5live-p6bundle"


@pytest.mark.skipif(not SAMPLE.is_dir(), reason="the preserved P6 sample bundle is not present")
def test_the_preserved_p6_sample_bundle_is_unchanged_still_verifies_and_publishes(tmp_path):
    from iceclaw.config import Settings
    from iceclaw.services import InvestigationService

    before = tree_hashes(SAMPLE)
    rep = verify_bundle(SAMPLE)
    assert rep.ok and not rep.warnings, rep.render()  # original v1 bundle, original logic
    m = load(SAMPLE, "input_manifest.json")
    assert (m["bundle_format"], m["derivation_version"]) == (
        "iceclaw-bundle/1",
        "iceclaw-derive/1",
    )
    svc = InvestigationService(Settings(data_root=Path("/none"), runs_root=tmp_path / "runs"))
    pub = svc.publish(SAMPLE)
    assert tree_hashes(SAMPLE) == before  # the authentic private bundle is untouched
    assert pub.ok, pub.verification.render()
    pm = load(pub.run_dir, "input_manifest.json")
    red = pm["transcript_redactions"]
    assert red["fields_redacted"] > 0 and red["source_bundle_format"] == "iceclaw-bundle/1"
    assert pm["model_usage"] == m["model_usage"]
    assert (pub.run_dir / "events.jsonl").read_bytes() == (SAMPLE / "events.jsonl").read_bytes()
    assert tree_hashes(pub.run_dir / "source") == tree_hashes(SAMPLE / "source")
    assert opaque_violations((pub.run_dir / "llm_calls.jsonl").read_text()) == []
    assert (pub.run_dir / "llm_calls.jsonl").stat().st_size < (
        SAMPLE / "llm_calls.jsonl"
    ).stat().st_size


@pytest.mark.skipif(not P5_BUNDLE.is_dir(), reason="the P5-derived bundle is not present")
def test_the_p5_derived_bundle_still_verifies():
    rep = verify_bundle(P5_BUNDLE)
    assert rep.ok and not rep.warnings, rep.render()
