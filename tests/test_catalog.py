"""Catalog + tool integration tests against the real published sector ``ice-s-0001-0002``.

Discovery tests copy the real export into a tmp dir (never edit the original); a few deliberately
damage the copy. Skipped (and reported as skipped) if the upstream checkout is missing.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from iceclaw.cli import main
from iceclaw.config import Settings, load_settings
from iceclaw.contracts import (
    AgentRegionContext,
    MeasurementFailure,
    Readiness,
    RegionCatalogEntry,
    RunBudget,
    SourceMode,
)
from iceclaw.data import catalog as cat
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.science import model_audit as audit

REPO = Path(__file__).resolve().parents[1]
SMOKE = Path(
    os.environ.get(
        "ICECLAW_MOONATLAS_SMOKE_ROOT",
        REPO / ".upstream/moonatlas-science/examples/science-smoke",
    )
)
pytestmark = pytest.mark.skipif(
    not (SMOKE / "manifest.json").is_file(), reason="MOONATLAS smoke export not checked out"
)
RID = "ice-s-0001-0002"
RAW_REL = "overlays/ice/ice-s-0001-0002-raw.webp"
BUDGET = RunBudget(max_events=10, max_hops=3, max_tool_calls=5, max_llm_calls=5, timeout_s=60)


def settings_for(*roots: Path, data_root: Path | None = None) -> Settings:
    return Settings(
        data_root=data_root or Path("/nonexistent-iceclaw-data"),
        runs_root=Path("/nonexistent-iceclaw-runs"),
        moonatlas_roots=tuple(roots),
    )


@pytest.fixture(scope="module")
def catalog():
    return cat.discover(settings_for(SMOKE))


@pytest.fixture(scope="module")
def arrays(catalog):
    return catalog.load_arrays(RID)


def copy_export(dst: Path) -> Path:
    (dst / "overlays/ice").mkdir(parents=True)
    for f in ("manifest.json", "processing-manifest.json", "ice-sectors.json"):
        shutil.copy(SMOKE / f, dst / f)
    shutil.copy(SMOKE / RAW_REL, dst / RAW_REL)
    return dst


def refresh_checksum(root: Path, rel: str) -> None:
    pm = root / "processing-manifest.json"
    d = json.loads(pm.read_text())
    d["checksums"][rel] = ma.sha256_file(root / rel)
    pm.write_text(json.dumps(d))


# ------------------------------------------------------------------------- entry mapping
def test_real_sector_loads_as_region_catalog_entry(catalog):
    assert [e.region_id for e in catalog.list_entries()] == [RID]
    e = catalog.get(RID).entry
    src = catalog.get(RID).source
    assert isinstance(e, RegionCatalogEntry)
    assert (e.pole, e.source_split, e.original_tile_id) == (
        "south",
        "test",
        "patch_0001_0002_S_80S",
    )
    assert e.source_mode is SourceMode.UPSTREAM_PUBLISHED_EXAMPLE
    assert e.source_dataset == "sombench-ice-prospectivity-regression"
    assert e.source_revision == "78040503ceaa68107829f48114774112e8ee940d"
    assert e.processing_version == "science-0.3.1"
    assert e.source_sha256 == src.sectors_json_sha256 and len(e.prediction_sha256) == 64
    assert e.prediction_sha256 == ma.sha256_file(SMOKE / RAW_REL)
    assert e.prediction_checkpoint_id == (
        f"ni_lfm_ps8_all_modalities_s42.ckpt@sha256:{src.checkpoint_sha256}"
    )
    assert src.checkpoint_sha256.startswith("94a64dfe7e8c892d")
    assert (e.width, e.height) == (256, 256) and e.spatial_resolution_m == pytest.approx(238.56)
    assert e.reference_path is None  # evaluator-owned, never set by the catalog


def test_entry_crs_affine_and_footprint_are_the_real_geometry(catalog):
    e = catalog.get(RID).entry
    src = catalog.get(RID).source
    assert e.lunar_crs == (
        "+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=500000 +y_0=500000 "
        "+R=1737400 +units=m +no_defs"
    )
    assert e.affine == (238.56, 0.0, 320006.48, 0.0, -238.56, 741064.88)
    assert len(e.footprint) == 4
    # affine + CRS reproduce the footprint corners through the P1 projection code
    a, _, c, _, f_e, f = e.affine
    for (lat, lon), (col, row) in zip(
        e.footprint, ((0, 0), (256, 0), (256, 256), (0, 256)), strict=True
    ):
        x, y = a * col + c, f_e * row + f
        assert x == pytest.approx(src.grid["xMinM"] + col * 238.56)
        assert y == pytest.approx(src.grid["yMaxM"] - row * 238.56)
        la, lo = ma.pixel_to_lat_lon(src.projection, src.grid, col, row)
        assert (la, lo) == pytest.approx((lat, lon), abs=1e-5)


def test_entry_modalities_pixelwise_only_with_unavailable_reason(catalog):
    rec = catalog.get(RID)
    e = rec.entry
    assert set(e.modalities) == {"nasa_ibm_prediction_raw"}
    m = e.modalities["nasa_ibm_prediction_raw"]
    assert m.unit == "prospectivity_index" and m.sha256 == e.prediction_sha256
    # scalars/anchors are NOT layers
    assert not any("anchor" in k or "psr" in k for k in e.modalities)
    kinds = {a.name: a.status for a in rec.availability}
    assert kinds["psr_fraction"] == "tile_scalar"
    assert kinds["anchor_maxTemperatureK"] == "single_anchor_point"
    assert e.readiness is Readiness.PARTIAL
    assert "UNAVAILABLE_MODALITY" in e.readiness_reason
    assert all(c.passed for c in rec.validation.checks)


def test_agent_context_from_real_entry_has_no_private_fields(catalog):
    e = catalog.get(RID).entry
    ctx = e.to_agent_context(sorted(audit.PREDICTION_TOOLS), BUDGET)
    assert isinstance(ctx, AgentRegionContext)
    dumped = ctx.model_dump_json()
    for forbidden in (
        str(SMOKE),
        "reference",
        ".webp",
        "prediction_raw_path",
        "prediction_sha256",
        e.prediction_sha256,
    ):
        assert forbidden not in dumped.lower()
    assert "nasa_ibm_prediction_raw" in ctx.available_modalities


def test_unknown_region_is_out_of_coverage(catalog):
    with pytest.raises(cat.RegionNotSupported) as e:
        catalog.get("ice-s-9999-9999")
    assert e.value.code == "OUT_OF_COVERAGE" and RID in str(e.value)


# -------------------------------------------------------------------------- real tools
def test_real_prediction_audit_matches_upstream_and_is_recomputable(catalog, arrays):
    src = catalog.get(RID).source
    ms = {m.tool_id: m for m in audit.run_prediction_audit(arrays)}
    s = ms["valid_pixel_summary"]
    tol = src.stats_tolerance
    assert s.valid_pixel_count == 65536
    assert s.value["min"] == pytest.approx(src.upstream_stats["min"], abs=tol)
    assert s.value["max"] == pytest.approx(src.upstream_stats["max"], abs=tol)
    assert s.value["mean"] == pytest.approx(src.upstream_stats["mean"], abs=tol)
    assert ms["quantiles"].value["q0.9"] == pytest.approx(src.upstream_stats["p90"], abs=tol)
    assert ms["unit_interval_check"].value["outside_unit_interval_pixels"] == src.clipped_pixels
    peak = ms["peak_location"].value
    assert (peak["row"], peak["col"], peak["edge_distance_px"]) == (
        src.raw_peak["row"],
        src.raw_peak["col"],
        src.raw_peak["edgeDistancePx"],
    )
    miss = ms["missingness"].value
    assert miss["valid_pixels"] == 65536 and miss["uncovered_pixels"] == 0
    for m in ms.values():
        assert m.region_id == RID and m.source_sha256 == {RAW_REL: src.prediction_sha256}
        assert m.resolution_m == pytest.approx(238.56)


def test_independent_recompute_from_original_file_agrees(catalog):
    """Fresh decode of the original source file -> same measurement (what a judge would do)."""
    first = audit.run_tool(catalog.load_arrays(RID), "gradient_stats")
    fresh = cat.discover(settings_for(SMOKE)).load_arrays(RID)
    again = audit.recompute(first, fresh)
    assert audit.matches(first, again) and first.tool_run_id == again.tool_run_id
    # and an independent numpy computation from the raw WebP bytes
    src = catalog.get(RID).source
    from PIL import Image

    with Image.open(SMOKE / RAW_REL) as im:
        rgba = np.asarray(im.convert("RGBA"))
    vals = src.encoding["offset"] + src.encoding["scale"] * (
        rgba[..., 0].astype(np.float64) * 256 + rgba[..., 1]
    )
    gx = (vals[1:-1, 2:] - vals[1:-1, :-2]) / (2 * 238.56)
    gy = (vals[2:, 1:-1] - vals[:-2, 1:-1]) / (2 * 238.56)
    assert first.value["mean"] == pytest.approx(float(np.hypot(gx, gy).mean()), rel=1e-9)


def test_real_region_thermal_psr_terrain_unavailable_despite_anchor_metadata(catalog, arrays):
    assert {"psr_fraction", "anchor_maxTemperatureK", "anchor_slopeDeg"} <= set(arrays.scalars)
    for tool in ("thermal_psr_consistency", "terrain_slope_summary"):
        m = audit.run_tool(arrays, tool)
        assert m.unavailable_reason is MeasurementFailure.UNAVAILABLE_MODALITY
        assert m.value is None
        assert m.source_sha256 == {RAW_REL: catalog.get(RID).source.prediction_sha256}


def test_real_window_and_out_of_coverage(arrays):
    m = audit.run_tool(arrays, "valid_pixel_summary", window=(180, 200, 20, 40))
    assert m.valid_pixel_count == 400 and m.value["max"] <= 1.0158
    bad = audit.run_tool(arrays, "valid_pixel_summary", window=(0, 300, 0, 10))
    assert bad.unavailable_reason is MeasurementFailure.OUT_OF_COVERAGE


# ------------------------------------------------------------------- discovery behaviour
def test_discovery_uses_actual_exports_not_hardcoded_ids(tmp_path):
    root = copy_export(tmp_path / "exp")
    f = root / "ice-sectors.json"
    data = json.loads(f.read_text())
    data[0]["id"] = "ice-s-0009-0009"  # test-only rename of a real sector
    f.write_text(json.dumps(data))
    refresh_checksum(root, "ice-sectors.json")
    c = cat.discover(settings_for(root))
    assert sorted(c.records) == ["ice-s-0009-0009"] and not c.rejections


def test_discovery_finds_exports_below_data_root(tmp_path):
    copy_export(tmp_path / "data" / "moonatlas" / "export-a")
    c = cat.discover(settings_for(data_root=tmp_path / "data"))
    assert sorted(c.records) == [RID]


def test_no_exports_means_empty_catalog(tmp_path):
    c = cat.discover(settings_for(tmp_path / "nothing"))
    assert c.records == {} and c.rejections == []


def test_corrupt_sector_is_rejected_not_listed(tmp_path):
    root = copy_export(tmp_path / "bad")
    p = root / RAW_REL
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    c = cat.discover(settings_for(root))
    assert c.records == {}
    assert [(r.sector_id, r.code) for r in c.rejections] == [(RID, "CHECKSUM_MISMATCH")]


def test_duplicate_region_ids_across_roots_are_rejected(tmp_path):
    a, b = copy_export(tmp_path / "a"), copy_export(tmp_path / "b")
    c = cat.discover(settings_for(a, b))
    assert list(c.records) == [RID]
    assert [r.code for r in c.rejections] == ["DUPLICATE_REGION_ID"]


def test_swapped_raster_after_discovery_fails_array_load(tmp_path):
    root = copy_export(tmp_path / "e")
    c = cat.discover(settings_for(root))
    (root / RAW_REL).write_bytes(b"not an image")
    with pytest.raises(ma.SourceError) as e:
        c.load_arrays(RID)
    assert e.value.code == "CHECKSUM_MISMATCH"


# ------------------------------------------------------------------------------- CLI
def test_cli_list_regions_real_catalog(monkeypatch, capsys):
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", str(SMOKE))
    monkeypatch.setenv("ICECLAW_DATA_ROOT", "/nonexistent-iceclaw-data")
    assert main(["list-regions"]) == 0
    out = capsys.readouterr().out
    assert RID in out and "readiness=partial" in out and "upstream_published_example" in out
    assert "pixelwise_temperature" in out and "UNAVAILABLE" in out


def test_cli_list_regions_json(monkeypatch, capsys):
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", str(SMOKE))
    monkeypatch.setenv("ICECLAW_DATA_ROOT", "/nonexistent-iceclaw-data")
    assert main(["list-regions", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["region_id"] for r in rows] == [RID]
    assert rows[0]["pixelwise_modalities"] == ["nasa_ibm_prediction_raw"]
    assert rows[0]["grid"] == [256, 256]


def test_cli_list_regions_empty_catalog_is_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", str(tmp_path))
    monkeypatch.setenv("ICECLAW_DATA_ROOT", str(tmp_path / "none"))
    assert main(["list-regions"]) == 1
    assert "NO_VALIDATED_REGIONS" in capsys.readouterr().err


def test_settings_parse_moonatlas_roots():
    s = load_settings({"ICECLAW_MOONATLAS_ROOTS": os.pathsep.join(["/a", "/b"])})
    assert s.moonatlas_roots == (Path("/a"), Path("/b"))
    assert load_settings({}).moonatlas_roots == (
        Path(".upstream/moonatlas-science/examples/science-smoke"),
    )
