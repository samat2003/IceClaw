"""P8 workbench: view-models, single-flight live runner, and the Streamlit page (AppTest).

Synthetic scripted data prove the plumbing without any model call or API key. The real-data
tests at the end use the preserved public P6 bundle and the local catalog, and skip if absent.
No test here contacts a model: the live path is exercised with the scripted provider, and the
real-client path is only checked up to its configuration pre-flight.
"""

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pytest
import streamlit as st
from bundle_helpers import copy_bundle, rehash, scripted_service
from streamlit.testing.v1 import AppTest

from iceclaw.agents.provider import OPAQUE_FIXTURE
from iceclaw.config import Settings
from iceclaw.reports import export as ex
from iceclaw.services import InvestigationService
from iceclaw.ui import regions as regions_mod
from iceclaw.ui import runner, views

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "app.py")
SAMPLE_PUBLIC = REPO / "runs" / "inv-20261031T201513Z-p6-public"


def sha_tree(d: Path) -> dict[str, str]:
    import hashlib

    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    svc = scripted_service(root / "runs", root / "src", emit_reasoning=True)
    private = svc.investigate("synthetic-region", run_id="inv-ui-private")
    public = svc.publish(private.run_dir, out_id="inv-ui-public")
    return svc, private, public, root


@pytest.fixture(autouse=True)
def _fresh_app_state():
    st.cache_resource.clear()
    st.cache_data.clear()
    runner.forget_managers()
    yield
    runner.forget_managers()


# =================================================================== view-models
def test_only_public_bundles_are_discovered(world):
    svc, _, public, _ = world
    found = views.discover_public_bundles(svc.settings.runs_root)
    assert [b.run_id for b in found] == ["inv-ui-public"]  # the private bundle is never offered
    assert found[0].profile == "public-redacted" and found[0].region_id == "synthetic-region"


def test_bundle_view_exposes_timeline_findings_and_labels(world):
    _, _, public, _ = world
    v = views.load_bundle_view(public.run_dir)
    assert v.is_public and v.region_id == "synthetic-region" and v.timeline
    assert [r.event_id for r in v.timeline] == [e["event_id"] for e in v.events]
    assert all(r.t_offset_s >= 0 for r in v.timeline)
    assert {r.agent_id for r in v.timeline} <= set(views.AGENT_STYLE)
    rows = views.measurement_rows(v.evidence)
    assert rows and {r["category"] for r in rows} <= set(views.CATEGORY_ORDER)
    for r in rows:  # units come from the recorded measurement, never invented
        assert r["unit"] and r["basis"].startswith("DETERMINISTIC CALCULATION")
        if r["category"] == "NASA–IBM prediction":
            assert "model output" in r["basis"]
    temps = [r for r in rows if r["quantity"] == "psr_mean_k"]
    assert temps and temps[0]["unit"] == "K" and temps[0]["category"] == "Thermal and shadow"
    slope = [r for r in rows if r["quantity"] == "median_deg"]
    assert slope and slope[0]["unit"] == "deg" and slope[0]["category"] == "Terrain"
    assert views.statements_by_agent(v.evidence)
    assert views.next_steps(v.evidence)
    assert "validated" in views.ATTENTION_NOTICE and "not physical ice prospectivity" in (
        views.ATTENTION_NOTICE.lower().replace("**", "")
    )
    assert "not detected or confirmed water ice" in views.PROSPECTIVITY_NOTICE


def test_followup_is_only_what_really_happened(world):
    _, _, public, _ = world
    v = views.load_bundle_view(public.run_dir)
    real = {e["event_id"] for e in v.events if e["event_kind"] in views.FOLLOWUP_KINDS}
    assert {r.event_id for r in views.followup_events(v)} == real
    # a bundle with no follow-up events yields an empty list (the page then says none occurred)
    quiet = [e for e in v.events if e["event_kind"] not in views.FOLLOWUP_KINDS]
    assert not [r for r in views.build_timeline(quiet) if r.followup]


def test_layers_are_georeferenced_from_the_verified_source_package(world):
    _, _, public, _ = world
    v = views.load_bundle_view(public.run_dir)
    geom, layers = views.build_layers(views.source_arrays(v))
    assert set(layers) == set(views.LAYER_ORDER)
    h, w = geom.shape
    assert layers["pixelwise_temperature"].values.shape == (h, w) == geom.lat.shape
    assert len(geom.x_km) == w and len(geom.y_km) == h
    assert list(geom.y_km) == sorted(geom.y_km)  # ascending, layers flipped to match
    # pixel (row 0, col 0) is the NW corner: top row of the flipped array, left column
    x0, y0 = views.pixel_xy_km(geom, 0, 0)
    assert x0 == pytest.approx(geom.x_km[0]) and y0 == pytest.approx(geom.y_km[-1])
    assert layers["nasa_ibm_prediction_raw"].style.kind_badge == "MODEL OUTPUT"
    assert layers["pixelwise_terrain"].style.unit == "deg"
    fig = views.make_heatmap_figure(geom, layers["pixelwise_temperature"])
    assert fig.data[0].type == "heatmap" and fig.layout.yaxis.scaleanchor == "x"
    assert "K" in fig.data[0].colorbar.title.text
    tgt = views.rank_targets(geom, v.rankings)
    fig2 = views.make_heatmap_figure(geom, layers["nasa_ibm_prediction_raw"], targets=tgt)
    assert len(fig2.data) == (2 if tgt else 1)
    if tgt:
        assert "not an ice location" in fig2.data[1].hovertemplate


def test_public_zip_contains_no_opaque_blobs_or_private_history(world):
    _, private, public, _ = world
    blob = views.public_bundle_zip(public.run_dir)
    import io

    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    assert all(n.startswith("inv-ui-public/") for n in names)
    assert not any("/internal/" in n for n in names)
    assert {"input_manifest.json", "events.jsonl", "evidence.json", "rankings.csv"} <= {
        n.split("/", 1)[1] for n in names
    }
    for n in names:
        assert OPAQUE_FIXTURE.encode() not in z.read(n)
    llm = z.read("inv-ui-public/llm_calls.jsonl").decode()
    assert ex.REDACTED_MARKER in llm


def test_zip_refuses_private_bundles_and_bundles_with_reinserted_blobs(world, tmp_path):
    _, private, public, _ = world
    with pytest.raises(views.DownloadRefused):
        views.public_bundle_zip(private.run_dir)  # private profile: never downloadable
    b = copy_bundle(public.run_dir, tmp_path / "t")
    p = b / "llm_calls.jsonl"
    p.write_text(p.read_text().replace(ex.REDACTED_MARKER, OPAQUE_FIXTURE, 1))
    with pytest.raises(views.DownloadRefused):
        views.public_bundle_zip(b)
    (tmp_path / "nothing").mkdir()
    with pytest.raises(views.DownloadRefused):
        views.public_bundle_zip(tmp_path / "nothing")


def test_load_bundle_view_reports_missing_files_clearly(world, tmp_path):
    _, _, public, _ = world
    b = copy_bundle(public.run_dir, tmp_path / "broken")
    (b / "evidence.json").unlink()
    with pytest.raises(views.BundleLoadError, match="evidence.json"):
        views.load_bundle_view(b)
    (b / "input_manifest.json").write_text("{not json")
    with pytest.raises(views.BundleLoadError):
        views.load_bundle_view(b)


def test_catalog_view_degrades_gracefully_without_data(tmp_path):
    s = Settings(
        data_root=tmp_path,
        runs_root=tmp_path,
        moonatlas_roots=(tmp_path / "none",),
        sombench_root=tmp_path / "none2",
    )
    cv = regions_mod.load_catalog_view(s)
    assert cv.rows == [] and "No local catalog" in cv.warning and cv.investigable_ids == set()


# ===================================================================== live runner
def slow_manager(root: Path, latency=0.03, **kw):
    svc = scripted_service(root / "runs", root / "src", latency_s=latency, emit_reasoning=True)
    return runner.LiveRunManager(
        root / "runs", lambda: svc, allowed_regions=lambda: {"synthetic-region"}, **kw
    )


def test_a_second_start_while_running_is_refused_and_progress_is_real(tmp_path):
    mgr = slow_manager(tmp_path)
    r1 = mgr.start("synthetic-region")
    assert r1.started and r1.code == "STARTED"
    for _ in range(5):  # refreshes / reruns / double clicks while in flight
        r = mgr.start("synthetic-region")
        assert not r.started and r.code == "ALREADY_RUNNING"
    assert mgr.snapshot().state == runner.RUNNING and mgr.starts_attempted == 1
    seen = 0
    deadline = time.time() + 60
    while mgr.snapshot().state == runner.RUNNING and time.time() < deadline:
        seen = max(seen, len(mgr.progress().events))  # real events from the live history
        time.sleep(0.01)
    mgr.wait(60)
    snap = mgr.snapshot()
    assert snap.state == runner.SUCCEEDED, (snap.error_code, snap.error_message)
    assert snap.ended_reason == "QUIESCENT" and snap.public_run_id == f"{snap.run_id}-public"
    assert mgr.starts_attempted == 1
    final = mgr.progress()
    assert len(final.events) >= seen > 0 and final.model_calls > 0
    # the published bundle verifies and the marker is gone; both runs are on disk
    rep = InvestigationService(Settings(data_root=tmp_path, runs_root=tmp_path / "runs")).verify(
        snap.public_run_id
    )
    assert rep.ok
    assert not (tmp_path / "runs" / runner.MARKER).exists()
    assert [b.run_id for b in views.discover_public_bundles(tmp_path / "runs")] == [
        snap.public_run_id
    ]  # only the public one is offered
    # a finished job can be dismissed and then a new one started deliberately
    mgr.reset()
    assert mgr.snapshot().state == runner.IDLE
    assert mgr.start("synthetic-region").started
    mgr.wait(60)
    assert mgr.starts_attempted == 2


def test_region_without_a_verified_prediction_is_refused_before_any_work(tmp_path):
    mgr = slow_manager(tmp_path)
    r = mgr.start("ice-s-0000-0003")  # an observation-only tile
    assert not r.started and r.code == "REGION_NOT_ALLOWED"
    assert mgr.snapshot().state == runner.IDLE
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_missing_api_configuration_blocks_before_a_run_or_client_exists(tmp_path, monkeypatch):
    from iceclaw.agents import provider

    def boom(*a, **k):
        raise AssertionError("a model client must not be built")

    monkeypatch.setattr(provider, "build_openai_client", boom)
    settings = Settings(data_root=tmp_path, runs_root=tmp_path / "runs", llm_model="gpt-6-luna")
    mgr = runner.LiveRunManager(
        settings.runs_root, lambda: InvestigationService(settings), allowed_regions=lambda: {"r"}
    )
    st_ = mgr.config_status()
    assert not st_["ok"] and st_["code"] == "MISSING_API_KEY" and st_["model"] == "gpt-6-luna"
    assert "OPENAI_API_KEY is not set" in st_["message"] and not st_["message"].startswith(
        "MISSING"
    )
    r = mgr.start("r")
    assert not r.started and r.code == "MISSING_API_KEY"
    assert mgr.snapshot().state == runner.IDLE and not settings.runs_root.exists()


def test_configured_live_path_uses_the_real_service_and_maf_client(tmp_path, monkeypatch):
    """With a (fake, local) key the manager calls the real InvestigationService.investigate and
    that builds the Microsoft Agent Framework OpenAI client; here both are intercepted so nothing
    is sent anywhere."""
    calls = {}

    def fake_investigate(self, region_id, **kw):
        calls["region"], calls["run_id"] = region_id, kw.get("run_id")
        raise RuntimeError("stop here: no model call in tests")

    monkeypatch.setattr(InvestigationService, "investigate", fake_investigate)
    settings = Settings(
        data_root=tmp_path,
        runs_root=tmp_path / "runs",
        llm_model="gpt-6-luna",
        openai_api_key="test-placeholder",
    )
    mgr = runner.LiveRunManager(
        settings.runs_root, lambda: InvestigationService(settings), allowed_regions=lambda: {"r"}
    )
    assert mgr.config_status()["ok"]
    assert mgr.start("r").started
    mgr.wait(30)
    snap = mgr.snapshot()
    assert calls["region"] == "r" and calls["run_id"] == snap.run_id
    assert snap.state == runner.FAILED and "no model call in tests" in snap.error_message
    assert not (settings.runs_root / runner.MARKER).exists()  # guard released even on failure


def test_failure_message_never_contains_key_like_text(tmp_path):
    assert "sk-proj-abcdefgh12345678" not in runner.scrub("bad key sk-proj-abcdefgh12345678 sent")
    svc = scripted_service(tmp_path / "runs", tmp_path / "src")

    def explode(_rid):
        raise ValueError("Incorrect API key provided: sk-live-ABCDEFGH123456789")

    svc.spec_provider = explode
    mgr = runner.LiveRunManager(tmp_path / "runs", lambda: svc, allowed_regions=lambda: {"x"})
    assert mgr.start("x").started
    mgr.wait(30)
    snap = mgr.snapshot()
    assert snap.state == runner.FAILED and "sk-live-ABCDEFGH" not in snap.error_message


def test_a_live_run_recorded_by_another_process_blocks_a_new_start(tmp_path):
    mgr = slow_manager(tmp_path)
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / runner.MARKER).write_text(
        json.dumps({"pid": os.getppid(), "run_id": "inv-other", "region_id": "synthetic-region"})
    )
    r = mgr.start("synthetic-region")
    assert not r.started and r.code == "ALREADY_RUNNING"
    # a stale marker (dead pid) does not block
    (tmp_path / "runs" / runner.MARKER).write_text(json.dumps({"pid": 2**22 + 12345}))
    assert mgr.start("synthetic-region").started
    mgr.wait(60)


# ===================================================================== the page
def page_env(monkeypatch, runs_root: Path):
    monkeypatch.setenv("ICECLAW_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", "/nonexistent-iceclaw")
    monkeypatch.setenv("ICECLAW_SOMBENCH_ROOT", "/nonexistent-iceclaw-sombench")
    monkeypatch.setenv("ICECLAW_DATA_ROOT", "/nonexistent-iceclaw")


def forbid_model_calls(monkeypatch):
    from iceclaw.agents import provider

    def no(*a, **k):
        raise AssertionError("the page must not call a model")

    monkeypatch.setattr(InvestigationService, "investigate", no)
    monkeypatch.setattr(provider, "build_openai_client", no)


def text_of(at, kind) -> str:
    return "\n".join(getattr(e, "value", "") or "" for e in getattr(at, kind))


@pytest.fixture
def saved_runs(world, tmp_path):
    """A runs folder holding only the public bundle (+ the private one, to prove it is hidden)."""
    _, private, public, _ = world
    runs = tmp_path / "runs"
    shutil.copytree(public.run_dir, runs / public.run_id)
    shutil.copytree(private.run_dir, runs / private.run_id)
    return runs


def test_page_loads_recorded_bundle_without_any_model_call(saved_runs, monkeypatch):
    page_env(monkeypatch, saved_runs)
    forbid_model_calls(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    md = text_of(at, "markdown")
    assert "RECORDED INVESTIGATION" in md and "Viewing it makes no model calls" in md
    assert "not detected or confirmed water ice" in md
    assert "ice-prospectivity index" in md
    sel = {s.label: s for s in at.selectbox}
    opts = sel["Saved public investigation"].options
    assert len(opts) == 1 and opts[0].startswith("inv-ui-public")  # private bundle hidden
    assert "inv-ui-private" not in " ".join(opts)
    assert text_of(at, "subheader").count("·") >= 4
    live = next(b for b in at.button if "Investigate" in b.label)
    assert live.disabled  # no confirmation, no key, region not catalogued: cannot start
    assert "LIVE INVESTIGATION completed" not in md


def test_page_verify_button_runs_the_real_offline_verifier(saved_runs, monkeypatch):
    page_env(monkeypatch, saved_runs)
    forbid_model_calls(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert "Not verified in this session" in text_of(at, "info")
    next(b for b in at.button if "Verify findings" in b.label).click()
    at.run()
    assert not at.exception, at.exception
    assert "VERIFIED" in text_of(at, "success")
    df = at.dataframe[-1].value  # the checks table
    assert set(df["outcome"]) == {"✅ PASS"} and "public_transcript_redacted" in set(df["check"])
    assert len(df) >= 26


def test_page_shows_failed_verification_and_disables_download(saved_runs, monkeypatch):
    page_env(monkeypatch, saved_runs)
    forbid_model_calls(monkeypatch)
    b = saved_runs / "inv-ui-public"
    ev = b / "events.jsonl"
    ev.write_text(
        ev.read_text().replace('"claim_or_request":"', '"claim_or_request":"TAMPERED ', 1)
    )
    rehash(b, "events.jsonl")  # attacker keeps hashes consistent: the re-derivation must catch it
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(b_ for b_ in at.button if "Verify findings" in b_.label).click()
    at.run()
    assert not at.exception, at.exception
    assert "VERIFICATION FAILED" in text_of(at, "error")
    df = at.dataframe[-1].value
    assert "❌ FAIL" in set(df["outcome"])
    dl = at.get("download_button")
    assert dl and all(d.proto.disabled for d in dl)


def test_page_download_offered_for_public_bundle_only(saved_runs, monkeypatch):
    page_env(monkeypatch, saved_runs)
    forbid_model_calls(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    dl = at.get("download_button")
    assert dl and not dl[0].proto.disabled
    assert "public bundle" in dl[0].proto.label.lower() and "bundle/2" in dl[0].proto.label
    assert "no provider-opaque reasoning payloads" in text_of(at, "caption")


def test_page_no_saved_bundle_and_no_catalog_is_a_clear_message(tmp_path, monkeypatch):
    page_env(monkeypatch, tmp_path / "empty-runs")
    forbid_model_calls(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert "No regions are available" in text_of(at, "error")


def test_page_live_click_needs_confirmation_and_is_single_flight(saved_runs, monkeypatch, world):
    """The button is wired to LiveRunManager.start: disabled until confirmed, one start per
    click, and ordinary reruns never start another run. Scripted provider; no model call."""
    svc, _, _, root = world
    live_svc = scripted_service(saved_runs, root / "src2", latency_s=0.02, emit_reasoning=True)
    page_env(monkeypatch, saved_runs)
    row = regions_mod.RegionRow("synthetic-region", "PAIRED", True, "")
    monkeypatch.setattr(
        regions_mod, "load_catalog_view", lambda s: regions_mod.CatalogView([row], "")
    )
    mgr = runner.LiveRunManager(
        saved_runs, lambda: live_svc, allowed_regions=lambda: {"synthetic-region"}
    )
    monkeypatch.setattr(runner, "get_manager", lambda *a, **k: mgr)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    go = next(b for b in at.button if "Investigate" in b.label)
    assert go.disabled and mgr.starts_attempted == 0
    for _ in range(3):  # plain reruns / refreshes never start a run
        at.run()
    assert mgr.starts_attempted == 0
    at.checkbox(key="confirm_live").check().run()
    go = next(b for b in at.button if "Investigate" in b.label)
    assert not go.disabled and mgr.starts_attempted == 0  # confirmed, but nothing started yet
    go.click().run()
    assert mgr.starts_attempted == 1
    for _ in range(3):
        at.run()  # reruns while running or after: still exactly one run
    mgr.wait(60)
    assert mgr.starts_attempted == 1
    at.run()
    assert not at.exception, at.exception
    snap = mgr.snapshot()
    assert snap.state == runner.SUCCEEDED
    md = text_of(at, "markdown")
    assert "LIVE INVESTIGATION completed in this app session" in md
    assert next(s for s in at.selectbox if s.label == "Saved public investigation").value == (
        snap.public_run_id
    )
    assert mgr.starts_attempted == 1
    assert "internal" not in {p.name for p in (saved_runs / snap.public_run_id).iterdir()}


# ===================================================================== real preserved data
REAL = pytest.mark.skipif(not SAMPLE_PUBLIC.is_dir(), reason="preserved public P6 bundle absent")


@REAL
def test_real_public_bundle_loads_and_verifies_from_the_page_without_model_calls(monkeypatch):
    monkeypatch.setenv("ICECLAW_RUNS_ROOT", str(REPO / "runs"))
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", "/nonexistent-iceclaw")
    monkeypatch.setenv("ICECLAW_SOMBENCH_ROOT", "/nonexistent-iceclaw-sombench")
    forbid_model_calls(monkeypatch)
    before = sha_tree(SAMPLE_PUBLIC)
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    sel = {s.label: s for s in at.selectbox}
    assert sel["Saved public investigation"].value == "inv-20261031T201513Z-p6-public"
    md = text_of(at, "markdown")
    assert "RECORDED INVESTIGATION" in md and "gpt-6-luna" in md and "74,991 tokens" in md
    assert "139.22 K" in md or any("139.22" in m.value for m in at.metric)
    next(b for b in at.button if "Verify findings" in b.label).click()
    at.run()
    assert "VERIFIED" in text_of(at, "success") and "26 checks passed" in text_of(at, "success")
    assert sha_tree(SAMPLE_PUBLIC) == before  # viewing/verifying changes nothing


@REAL
def test_real_catalog_lists_one_paired_and_eleven_unavailable_tiles(monkeypatch):
    src = REPO / ".upstream" / "moonatlas-science" / "examples" / "science-smoke"
    som = REPO / "data" / "sombench" / "prospectivity_dataset"
    if not (src.is_dir() and som.is_dir()):
        pytest.skip("local catalog data absent")
    monkeypatch.setenv("ICECLAW_RUNS_ROOT", str(REPO / "runs"))
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", str(src))
    monkeypatch.setenv("ICECLAW_SOMBENCH_ROOT", str(som))
    forbid_model_calls(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception, at.exception
    region = next(s for s in at.selectbox if s.label == "Lunar region")
    assert len(region.options) == 12
    assert sum("UNAVAILABLE" in o for o in region.options) == 11
    assert region.value == "ice-s-0001-0002"
    other = next(o for o in region.options if "UNAVAILABLE" in o)
    region.select(other.split("  ")[0]).run()
    assert "unavailable for prediction investigations" in text_of(at, "error")
    assert "no NASA–IBM prediction" in text_of(at, "error")
    assert not at.button  # no Investigate / Verify controls for an unavailable tile


def test_ui_code_never_reads_dotenv_or_uses_other_model_vendors():
    for f in [REPO / "app.py", *sorted((REPO / "src" / "iceclaw" / "ui").glob("*.py"))]:
        t = f.read_text(encoding="utf-8").lower()
        assert "anthropic" not in t and "claude" not in t, f.name
        assert '".env"' not in t and "'.env'" not in t, f.name  # only the existing loader reads it
