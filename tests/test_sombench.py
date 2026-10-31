"""P4 tests: authentic SomBench observations, verified pairing with the MOONATLAS prediction,
catalog status, and real thermal/terrain measurements.

Real-data tests are skipped (and reported as skipped) if the downloaded files are absent.
Corruption tests copy the real files into a tmp dir and damage the copy.
"""

import dataclasses
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import rasterio

from iceclaw.cli import main
from iceclaw.config import Settings
from iceclaw.contracts import MeasurementFailure, Readiness, RunBudget, SourceMode
from iceclaw.data import catalog as cat
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.data import sombench_adapter as sa
from iceclaw.science import model_audit as audit

REPO = Path(__file__).resolve().parents[1]
SMOKE = Path(
    os.environ.get(
        "ICECLAW_MOONATLAS_SMOKE_ROOT",
        REPO / ".upstream/moonatlas-science/examples/science-smoke",
    )
)
SOM = Path(
    os.environ.get("ICECLAW_SOMBENCH_TEST_ROOT", REPO / "data/sombench/prospectivity_dataset")
)
PATCH = "patch_0001_0002_S_80S"
RID = "ice-s-0001-0002"
needs_data = pytest.mark.skipif(
    not ((SMOKE / "manifest.json").is_file() and (SOM / "download_manifest.json").is_file()),
    reason="MOONATLAS smoke export or downloaded SomBench files not present",
)


def settings(som: Path | None) -> Settings:
    return Settings(
        data_root=Path("/nonexistent-iceclaw-data"),
        runs_root=Path("/nonexistent-iceclaw-runs"),
        moonatlas_roots=(SMOKE,),
        sombench_root=som,
    )


def tif(layer: str, root: Path = SOM, patch: str = PATCH) -> Path:
    return root / f"{patch}_{layer}.tif"


def read(layer: str, root: Path = SOM) -> np.ndarray:
    with rasterio.open(tif(layer, root)) as ds:
        return ds.read(1).astype(np.float64)


# ------------------------------------------------------------------ pure/synthetic units
def test_parse_patch_id_matches_upstream_rule():
    assert sa.parse_patch_id("patch_0001_0002_S_80S") == ("ice-s-0001-0002", "south")
    assert sa.parse_patch_id("patch_0008_0003_80N") == ("ice-n-0008-0003", "north")
    for bad in ("patch_1_2_S_80S", "foo", "patch_0001_0002_S_80S.tif", "patch_0001_0002_X"):
        with pytest.raises(sa.SombenchError) as e:
            sa.parse_patch_id(bad)
        assert e.value.code == "MALFORMED_PATCH_ID"


def test_orientation_scan_picks_identity_and_detects_flips_synthetic():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(40, 40)).cumsum(0).cumsum(1)  # smooth, asymmetric SYNTHETIC field
    fin = np.ones_like(base, bool)
    scan = sa.orientation_scan(base, base, fin)
    assert scan["rot0"] == pytest.approx(1.0)
    assert max(abs(v) for k, v in scan.items() if k != "rot0") < 0.9
    scan_flip = sa.orientation_scan(np.flipud(base), base, fin)
    best = max(scan_flip, key=lambda k: abs(scan_flip[k]))
    assert best == "rot0+flipud"  # a vertical flip is identified, not silently accepted
    nan_all = sa.orientation_scan(base, base, np.zeros_like(fin))
    assert all(np.isnan(v) for v in nan_all.values())


def test_layer_specs_only_expose_established_units():
    exposed = {n for n, s in sa.LAYER_SPECS.items() if s.exposed_as}
    assert exposed == {"TMAX", "LPSR", "SLOPE"}
    assert {s.unit for n, s in sa.LAYER_SPECS.items() if n in exposed} == {"K", "fraction", "deg"}
    for n, s in sa.LAYER_SPECS.items():
        if n not in exposed:
            assert s.unit is None and s.exposed_as is None, n  # unknown unit => no interpretation
    assert sa.LAYER_SPECS["ASP_SIN_COS"].bands == 2
    assert SourceMode.OBSERVATION_ONLY.value == "observation_only"  # additive enum member


# ----------------------------------------------------------------------- real files
@needs_data
def test_download_manifest_is_pinned_and_every_file_hub_verified():
    m = json.loads((SOM / "download_manifest.json").read_text())
    assert m["revision"] == sa.PINNED_REVISION and m["repo"] == sa.REPO
    files = m["files"]
    assert len(files) >= 12
    for name, rec in files.items():
        assert rec["hub_verification"] in sa.HUB_VERIFIED, name
        assert ma.sha256_file(SOM / name) == rec["sha256"], name
        assert (SOM / name).stat().st_size == rec["size"], name
    assert not any(n.endswith("_PRO.tif") for n in files)  # expert target never lives here
    assert not list(SOM.glob("*_PRO.tif"))


@needs_data
def test_real_patch_files_rasterio_facts():
    p = sa.load_patch(SOM, PATCH)
    assert (p.sector_id, p.pole, p.split) == (RID, "south", "test")
    assert set(p.layers) == set(sa.INPUT_LAYERS)
    for name, lf in p.layers.items():
        assert (lf.height, lf.width) == (256, 256)
        assert lf.dtype == "float32" and lf.nodata is not None and np.isnan(lf.nodata)
        assert lf.count == (2 if name == "ASP_SIN_COS" else 1)  # resolves the README discrepancy
        assert len(lf.sha256) == 64 and lf.hub_verification in sa.HUB_VERIFIED
        assert lf.tags["variable_name"] == name and lf.tags["hemisphere"] == "S"
        assert lf.finite_fraction == 1.0  # this tile has no nodata
    assert p.layers["TMAX"].transform == pytest.approx(
        (238.56, 0.0, 320006.48, 0.0, -238.56, 741064.88), abs=1e-6
    )
    assert p.crs_proj4.startswith("+proj=stere +lat_0=-90 +lon_0=0 +k=0.994")
    assert "+R=1737400" in p.crs_proj4
    assert p.grid["pixelSizeXM"] == pytest.approx(238.56)


@needs_data
def test_real_alignment_with_validated_prediction_all_checks_pass():
    src = ma.load_sector(SMOKE)
    pred = ma.decode_prediction(src).values
    rep = sa.check_alignment(sa.load_patch(SOM, PATCH), src, pred)
    assert rep.ok, [(c.name, c.detail) for c in rep.failures()]
    names = {c.name for c in rep.checks}
    assert {
        "tile_identity", "shape", "crs", "affine", "footprint_corners", "peak_pixel_center",
        "anchor_temperature", "anchor_slope", "anchor_shadow", "psr_fraction",
        "orientation_TMAX", "orientation_LPSR",
    } <= names  # fmt: skip
    t = rep.dihedral["TMAX"]
    assert t["rot0"] == pytest.approx(-0.930, abs=0.005)
    assert max(abs(v) for k, v in t.items() if k != "rot0") < 0.2
    # published anchor values are reproduced from the ORIGINAL arrays at the raw-peak pixel
    row, col = src.raw_peak["row"], src.raw_peak["col"]
    assert round(float(read("TMAX")[row, col]), 1) == 84.9
    assert round(float(read("SLOPE")[row, col]), 2) == 6.56
    assert read("LPSR")[row, col] >= 0.5
    assert round(float(read("LPSR").mean()), 4) == 0.0665


@needs_data
def test_alignment_refuses_flipped_transposed_shifted_or_wrong_tile():
    src = ma.load_sector(SMOKE)
    pred = ma.decode_prediction(src).values
    patch = sa.load_patch(SOM, PATCH)
    for bad_pred in (np.flipud(pred), np.fliplr(pred), pred.T):
        rep = sa.check_alignment(patch, src, np.ascontiguousarray(bad_pred))
        assert not rep.ok
        assert {"orientation_TMAX", "orientation_LPSR"} & {c.name for c in rep.failures()}
    shifted = dataclasses.replace(src, grid={**src.grid, "xMinM": src.grid["xMinM"] + 238.56})
    rep = sa.check_alignment(patch, shifted, pred)
    assert "affine" in {c.name for c in rep.failures()} and not rep.ok
    other = sa.load_patch(SOM, "patch_0003_0002_S_80S")
    rep = sa.check_alignment(other, src, pred)
    assert "tile_identity" in {c.name for c in rep.failures()} and not rep.ok


@needs_data
def test_other_real_tiles_load_with_natural_nodata_masks():
    p = sa.load_patch(SOM, "patch_0000_0003_S_80S")
    assert p.layers["TMAX"].finite_fraction == pytest.approx(0.773, abs=0.001)
    vals, finite = sa.read_layer(p, "TMAX")
    assert vals.shape == (1, 256, 256) and int(np.isnan(vals).sum()) == int((~finite).sum())
    assert (~finite).any()  # NaN is preserved, never filled
    with pytest.raises(sa.SombenchError) as e:
        sa.read_layer(p, "PRO")
    assert e.value.code == "UNKNOWN_LAYER"


PRO_DIR = REPO / "data/evaluation_inputs/prospectivity_target"


@needs_data
@pytest.mark.skipif(
    not (PRO_DIR / f"{PATCH}_PRO.tif").is_file(), reason="evaluator-only PRO not downloaded"
)
def test_evaluator_only_pro_target_agrees_with_published_reference_and_is_isolated():
    from iceclaw.evaluation.fixtures import load_reference

    ref, _ = load_reference(ma.load_sector(SMOKE))
    with rasterio.open(PRO_DIR / f"{PATCH}_PRO.tif") as ds:
        assert ds.count == 1 and ds.shape == (256, 256)
        pro = ds.read(1).astype(np.float64)
    assert float(np.abs(pro - ref).max()) <= 7.7e-6  # published raw encoding quantization/2
    assert PRO_DIR.resolve() != SOM.resolve() and not (SOM / f"{PATCH}_PRO.tif").exists()
    assert not list(SOM.glob("*PRO*"))  # the observation directory holds no expert target


# --------------------------------------------------------------------------- catalog
@pytest.fixture(scope="module")
def paired_catalog():
    return cat.discover(settings(SOM))


@needs_data
def test_catalog_pairs_region_and_marks_it_verified(paired_catalog):
    rec = paired_catalog.get(RID)
    e = rec.entry
    assert rec.status == "PAIRED" and rec.prediction_ready and rec.observation_ready
    assert e.readiness is Readiness.VERIFIED and e.readiness_reason is None
    assert set(e.modalities) == {
        "nasa_ibm_prediction_raw", "pixelwise_temperature", "pixelwise_shadow_mask",
        "pixelwise_terrain",
    }  # fmt: skip
    assert {k: v.unit for k, v in e.modalities.items()} == {
        "nasa_ibm_prediction_raw": "prospectivity_index",
        "pixelwise_temperature": "K",
        "pixelwise_shadow_mask": "fraction",
        "pixelwise_terrain": "deg",
    }
    for k in ("pixelwise_temperature", "pixelwise_shadow_mask", "pixelwise_terrain"):
        assert e.modalities[k].sha256 == ma.sha256_file(Path(e.modalities[k].path))
    assert rec.unit_unknown_layers == ["ASP_SIN_COS", "CUR", "DICE", "LPSR_DEN", "LPSR_DIS"]
    assert e.reference_path is None
    assert all(c.passed for c in rec.validation.checks)
    assert any(c.name == "pairing_orientation_TMAX" for c in rec.validation.checks)
    # tile scalars / anchors are still metadata, never layers
    assert {a.name: a.status for a in rec.availability}["psr_fraction"] == "tile_scalar"
    assert not any(a.status == "unavailable" for a in rec.availability)


@needs_data
def test_catalog_separates_observation_only_tiles(paired_catalog):
    obs = paired_catalog.observation_only
    assert len(obs) == 11 and RID not in obs
    assert all(o.status == "OBSERVATION_ONLY" and not o.prediction_ready for o in obs.values())
    assert all(o.observation_ready for o in obs.values())
    south_test = [
        x for x in (SOM / "test_filtered.txt").read_text().split() if x.endswith("_S_80S")
    ]
    assert set(obs) | {RID} == {sa.parse_patch_id(p)[0] for p in south_test}
    assert not set(obs) & set(paired_catalog.records)  # mutually exclusive
    with pytest.raises(cat.PredictionUnavailable) as e:
        paired_catalog.get("ice-s-0003-0002")
    assert e.value.code == "PREDICTION_UNAVAILABLE"
    with pytest.raises(cat.RegionNotSupported) as e2:
        paired_catalog.get("ice-s-9999-9999")
    assert e2.value.code == "OUT_OF_COVERAGE"
    assert paired_catalog.rejections == []


@needs_data
def test_without_observation_root_region_stays_prediction_only():
    c = cat.discover(settings(None))
    rec = c.get(RID)
    assert rec.status == "PREDICTION_ONLY" and not rec.observation_ready
    assert rec.entry.readiness is Readiness.PARTIAL
    assert set(rec.entry.modalities) == {"nasa_ibm_prediction_raw"}
    assert c.observation_only == {}
    arrays = c.load_arrays(RID)
    for tool in ("thermal_psr_consistency", "terrain_slope_summary"):
        assert audit.run_tool(arrays, tool).unavailable_reason is (
            MeasurementFailure.UNAVAILABLE_MODALITY
        )


@needs_data
def test_agent_context_exposes_new_modalities_but_no_paths_or_reference(paired_catalog):
    rec = paired_catalog.get(RID)
    arrays = paired_catalog.load_arrays(RID)
    tools = cat.available_tool_names(arrays)
    assert "thermal_psr_consistency" in tools and "terrain_slope_summary" in tools
    budget = RunBudget(max_events=5, max_hops=2, max_tool_calls=5, max_llm_calls=5, timeout_s=10)
    ctx = rec.entry.to_agent_context(tools, budget)
    dumped = ctx.model_dump_json()
    for forbidden in (
        str(SOM),
        str(SMOKE),
        ".tif",
        ".webp",
        "reference",
        "prediction_sha256",
        rec.entry.prediction_sha256,
    ):
        assert forbidden not in dumped, forbidden
    assert not re.search(r"PRO", dumped)
    assert "pixelwise_temperature" in ctx.available_modalities
    assert not any(s.endswith("_PRO.tif") for s in ctx.source_ids)


# ------------------------------------------------------------ real measurements, verified
@needs_data
def test_real_thermal_and_terrain_match_independent_numpy(paired_catalog):
    arrays = paired_catalog.load_arrays(RID)
    t = audit.run_tool(arrays, "thermal_psr_consistency")
    g = audit.run_tool(arrays, "terrain_slope_summary")
    tm, lp, sl = read("TMAX"), read("LPSR"), read("SLOPE")
    psr = lp >= 0.5
    # independent NumPy, straight from the GeoTIFF bytes
    assert t.unavailable_reason is None and t.valid_pixel_count == 65536
    assert t.value["psr_count"] == float(psr.sum()) == 4360.0
    assert t.value["psr_mean_k"] == pytest.approx(float(tm[psr].mean()), abs=1e-9)
    assert t.value["non_psr_mean_k"] == pytest.approx(float(tm[~psr].mean()), abs=1e-9)
    assert t.value["psr_minus_non_psr_k"] == pytest.approx(
        float(tm[psr].mean() - tm[~psr].mean()), abs=1e-9
    )
    assert t.value["psr_below_cold_trap_fraction"] == pytest.approx(float((tm[psr] < 110).mean()))
    assert g.value["mean_deg"] == pytest.approx(float(sl.mean()), abs=1e-9)
    assert g.value["p95_deg"] == pytest.approx(float(np.quantile(sl, 0.95)), abs=1e-9)
    assert g.value["max_deg"] == pytest.approx(float(sl.max()), abs=1e-9)
    assert g.value["steep_fraction"] == pytest.approx(float((sl > 20).mean()))
    # pinned real values (documented in docs/SOMBENCH_INGESTION.md)
    assert t.value["psr_mean_k"] == pytest.approx(139.21508, abs=1e-4)
    assert t.value["psr_minus_non_psr_k"] == pytest.approx(-107.77419, abs=1e-4)
    assert g.value["mean_deg"] == pytest.approx(8.734015, abs=1e-5)
    assert g.value["max_deg"] == pytest.approx(29.824226, abs=1e-5)
    # source traceability: the GeoTIFF checksums, units and recomputation
    assert sorted(t.source_sha256.values()) == sorted(
        [ma.sha256_file(tif("TMAX")), ma.sha256_file(tif("LPSR"))]
    )
    assert g.source_sha256 == {f"sombench:{PATCH}_SLOPE.tif": ma.sha256_file(tif("SLOPE"))}
    for m in (t, g):
        assert audit.matches(m, audit.recompute(m, paired_catalog.load_arrays(RID)))
        assert m.tolerance > 0 and m.resolution_m == pytest.approx(238.56)


@needs_data
def test_paired_arrays_preserve_masks_units_and_prediction_results(paired_catalog):
    arrays = paired_catalog.load_arrays(RID)
    for name, unit in (("pixelwise_temperature", "K"), ("pixelwise_shadow_mask", "fraction"),
                       ("pixelwise_terrain", "deg")):  # fmt: skip
        layer = arrays.layers[name]
        assert layer.unit == unit and layer.covered.all() and layer.values.dtype == np.float64
        assert layer.file_id.startswith("sombench:")
    # P2 prediction-only results are unchanged by pairing
    prediction_only = cat.discover(settings(None)).load_arrays(RID)
    for tool in audit.PREDICTION_TOOLS:
        a, b = audit.run_tool(arrays, tool), audit.run_tool(prediction_only, tool)
        assert a.tool_run_id == b.tool_run_id and audit.matches(a, b), tool
    # generic tools run on observation layers with their own units
    s = audit.run_tool(arrays, "valid_pixel_summary", modality="pixelwise_temperature")
    assert s.unit == "K" and s.value["min"] == pytest.approx(83.0304, abs=1e-3)
    # a tile with natural nodata keeps NaN uncovered (not filled) in the arrays
    from iceclaw.data.catalog import observation_arrays

    nat = observation_arrays(sa.load_patch(SOM, "patch_0000_0003_S_80S"))
    tl = nat.layers["pixelwise_temperature"]
    assert (~tl.covered).any() and np.isnan(tl.values[~tl.covered]).all()
    m = audit.run_tool(nat, "missingness", modality="pixelwise_temperature")
    assert m.value["uncovered_pixels"] > 0 and m.value["nonfinite_covered_pixels"] == 0


@needs_data
def test_cli_lists_paired_region_and_observation_only_tiles(monkeypatch, capsys):
    monkeypatch.setenv("ICECLAW_MOONATLAS_ROOTS", str(SMOKE))
    monkeypatch.setenv("ICECLAW_DATA_ROOT", "/nonexistent-iceclaw-data")
    monkeypatch.setenv("ICECLAW_SOMBENCH_ROOT", str(SOM))
    assert main(["list-regions"]) == 0
    out = capsys.readouterr().out
    assert "status=PAIRED" in out and "readiness=verified" in out
    assert "pixelwise_temperature [K]" in out and "pixelwise_terrain [deg]" in out
    assert "NOT exposed (unit unknown)" in out and "11 observation-only" in out
    assert main(["list-regions", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["region_id"] for r in rows] == [RID]
    assert rows[0]["observation_ready"] is True and rows[0]["prediction_ready"] is True
    assert main(["list-regions", "--observation-only", "--json"]) == 0
    obs = json.loads(capsys.readouterr().out)
    assert len(obs) == 11 and all(o["status"] == "OBSERVATION_ONLY" for o in obs)
    assert all(o["prediction_ready"] is False for o in obs)


# ------------------------------------------------------- missing / corrupt / mismatched
@pytest.fixture
def som_copy(tmp_path):
    if not (SOM / "download_manifest.json").is_file():
        pytest.skip("downloaded SomBench files not present")
    dst = tmp_path / "som"
    dst.mkdir()
    for f in ("download_manifest.json", "test_filtered.txt", "train_filtered.txt",
              "val_filtered.txt", "band_statistics.json"):  # fmt: skip
        shutil.copy(SOM / f, dst / f)
    for layer in sa.INPUT_LAYERS:
        shutil.copy(tif(layer), tif(layer, dst))
    sa.load_patch(dst, PATCH)  # the copy itself is valid
    return dst


def refresh(root: Path, layer: str) -> None:
    f = root / "download_manifest.json"
    m = json.loads(f.read_text())
    m["files"][f"{PATCH}_{layer}.tif"]["sha256"] = ma.sha256_file(tif(layer, root))
    f.write_text(json.dumps(m))


def rewrite(
    root: Path, layer: str, *, data=None, transform=None, count=None, dtype=None, tags=None
):
    p = tif(layer, root)
    with rasterio.open(p) as ds:
        arr, prof, t = ds.read(), ds.profile, ds.tags()
    if data is not None:
        arr = data
    if count is not None:
        arr, prof["count"] = arr[:count], count
    if dtype is not None:
        arr, prof["dtype"] = arr.astype(dtype), dtype
    if transform is not None:
        prof["transform"] = transform
    with rasterio.open(p, "w", **prof) as out:
        out.write(arr)
        out.update_tags(**{**t, **(tags or {})})
    refresh(root, layer)


def code(root: Path) -> str:
    with pytest.raises(sa.SombenchError) as e:
        sa.load_patch(root, PATCH)
    return e.value.code


@needs_data
def test_corrupt_missing_layer_file(som_copy):
    tif("SLOPE", som_copy).unlink()
    assert code(som_copy) == "MISSING_FILE"


@needs_data
def test_corrupt_flipped_byte_fails_checksum(som_copy):
    p = tif("TMAX", som_copy)
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    assert code(som_copy) == "CHECKSUM_MISMATCH"


@needs_data
def test_unverified_download_or_wrong_revision_is_refused(som_copy):
    f = som_copy / "download_manifest.json"
    m = json.loads(f.read_text())
    m["files"][f"{PATCH}_TMAX.tif"]["hub_verification"] = "unverified"
    f.write_text(json.dumps(m))
    assert code(som_copy) == "UNVERIFIED_DOWNLOAD"
    m["files"][f"{PATCH}_TMAX.tif"]["hub_verification"] = "lfs-sha256"
    m["revision"] = "0" * 40
    f.write_text(json.dumps(m))
    assert code(som_copy) == "REVISION_MISMATCH"


@needs_data
def test_unexpected_band_count_dtype_and_tags(som_copy):
    rewrite(som_copy, "ASP_SIN_COS", count=1)
    assert code(som_copy) == "UNEXPECTED_BAND_COUNT"
    shutil.copy(tif("ASP_SIN_COS"), tif("ASP_SIN_COS", som_copy))
    refresh(som_copy, "ASP_SIN_COS")
    rewrite(som_copy, "CUR", dtype="float64")
    assert code(som_copy) == "UNEXPECTED_DTYPE"
    shutil.copy(tif("CUR"), tif("CUR", som_copy))
    refresh(som_copy, "CUR")
    rewrite(som_copy, "DICE", tags={"variable_name": "SLOPE"})
    assert code(som_copy) == "LAYER_NAME_MISMATCH"
    shutil.copy(tif("DICE"), tif("DICE", som_copy))
    refresh(som_copy, "DICE")
    rewrite(som_copy, "LPSR", tags={"patch_col": "7"})
    assert code(som_copy) == "TILE_TAG_MISMATCH"


@needs_data
def test_layers_on_different_grids_are_refused_not_resampled(som_copy):
    with rasterio.open(tif("TMAX", som_copy)) as ds:
        t = ds.transform
    rewrite(som_copy, "TMAX", transform=rasterio.Affine(t.a, t.b, t.c + 238.56, t.d, t.e, t.f))
    assert code(som_copy) == "GRID_MISMATCH"


@needs_data
def test_split_membership_is_enforced(som_copy):
    f = som_copy / "test_filtered.txt"
    f.write_text(" ".join(x for x in f.read_text().split() if x != PATCH))
    assert code(som_copy) == "SPLIT_MISMATCH"
    (som_copy / "val_filtered.txt").unlink()
    assert code(som_copy) == "MISSING_FILE"


@needs_data
def test_swapped_file_after_load_fails_at_read_time(som_copy):
    patch = sa.load_patch(som_copy, PATCH)
    shutil.copy(tif("SLOPE"), tif("TMAX", som_copy))
    with pytest.raises(sa.SombenchError) as e:
        sa.read_layer(patch, "TMAX")
    assert e.value.code == "CHECKSUM_MISMATCH"


@needs_data
def test_catalog_rejects_corrupt_observation_and_keeps_prediction_only(som_copy):
    p = tif("TMAX", som_copy)
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    c = cat.discover(settings(som_copy))
    assert c.get(RID).status == "PREDICTION_ONLY"
    assert c.get(RID).entry.readiness is Readiness.PARTIAL
    assert [(r.sector_id, r.code) for r in c.rejections] == [(PATCH, "CHECKSUM_MISMATCH")]
    assert c.observation_only == {}


@needs_data
def test_catalog_does_not_pair_misaligned_observations(som_copy):
    """A vertically flipped (but internally consistent, re-hashed) TMAX must NOT be paired."""
    with rasterio.open(tif("TMAX", som_copy)) as ds:
        flipped = ds.read()[:, ::-1, :]  # flip rows (axis 1); axis 0 is the band axis
    rewrite(som_copy, "TMAX", data=np.ascontiguousarray(flipped))
    c = cat.discover(settings(som_copy))
    rec = c.get(RID)
    assert rec.status == "PREDICTION_ONLY" and not rec.observation_ready
    assert rec.pairing_error and "orientation_TMAX" in rec.pairing_error
    assert rec.entry.readiness is Readiness.PARTIAL
    assert set(rec.entry.modalities) == {"nasa_ibm_prediction_raw"}
    assert [r.code for r in c.rejections] == ["PAIRING_FAILED"]
    arrays = c.load_arrays(RID)
    assert audit.run_tool(arrays, "thermal_psr_consistency").unavailable_reason is (
        MeasurementFailure.UNAVAILABLE_MODALITY
    )
