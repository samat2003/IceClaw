"""P4 tests for the sealed multimodal suite ``p4-multimodal-v1`` (genuine SomBench tiles).

Fault-function tests use the real tiles but assert only array math. The pinned baseline outcomes
are the preliminary, frozen results of a baseline written BEFORE any agent run; two of them are
honest imperfections (one false positive, one pre-registration mismatch) and are pinned as such.
"""

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from iceclaw.config import Settings
from iceclaw.contracts import SourceMode
from iceclaw.data.catalog import discover
from iceclaw.evaluation import baselines, fixtures, score
from iceclaw.evaluation import multimodal as mm

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
SALT = b"p4-unit-test-salt-01"
pytestmark = pytest.mark.skipif(
    not ((SMOKE / "manifest.json").is_file() and (SOM / "download_manifest.json").is_file()),
    reason="MOONATLAS smoke export or downloaded SomBench files not present",
)


@pytest.fixture(scope="module")
def tiles():
    s = Settings(
        data_root=Path("/nonexistent-iceclaw-data"),
        runs_root=Path("/nonexistent-iceclaw-runs"),
        moonatlas_roots=(SMOKE,),
        sombench_root=SOM,
    )
    return mm.load_observation_tiles(discover(s))


@pytest.fixture(scope="module")
def suite(tiles, tmp_path_factory):
    root = tmp_path_factory.mktemp("mm")
    m = mm.build_multimodal_suite(tiles, root / "s", root / "e", salt=SALT, suite_seed=5)
    return m, root / "s", root / "e"


def sid() -> str:
    return mm.SUITE_ID


# ---------------------------------------------------------------------------- source
def test_twelve_independent_genuine_tiles_with_provenance(tiles):
    assert len(tiles) == 12 and len({t.tile_id for t in tiles}) == 12
    assert all(t.split == "test" and t.tile_id.endswith("_S_80S") for t in tiles)
    for t in tiles:
        assert set(t.case.layers) == set(mm.LAYER_ORDER)
        assert t.case.units == {
            "pixelwise_temperature": "K",
            "pixelwise_shadow_mask": "fraction",
            "pixelwise_terrain": "deg",
        }
        assert set(t.file_sha256) == {
            "ASP_SIN_COS", "CUR", "DICE", "LPSR", "LPSR_DEN", "LPSR_DIS", "SLOPE", "TMAX"
        }  # fmt: skip
        assert not t.case.layers[mm.TEMP][0].flags.writeable  # source is immutable
    assert len({t.case.grid["xMinM"] for t in tiles}) > 1  # genuinely different tiles


# ---------------------------------------------------------------------------- faults
def test_fault_functions_do_what_they_say_and_never_touch_the_source(tiles):
    clean = next(t for t in tiles if t.tile_id == "patch_0001_0002_S_80S")
    before = {k: v[0].tobytes() for k, v in clean.case.layers.items()}
    T, S, G = mm.TEMP, mm.SHADOW, mm.TERRAIN
    base = clean.case.layers
    c, _ = mm.apply_obs_fault("temperature_unit_celsius", clean.case, 1)
    assert np.allclose(c.layers[T][0], base[T][0] - 273.15)
    c, _ = mm.apply_obs_fault("slope_unit_radians", clean.case, 1)
    assert np.allclose(c.layers[G][0], np.deg2rad(base[G][0]))
    c, _ = mm.apply_obs_fault("psr_mask_inverted", clean.case, 1)
    assert np.allclose(c.layers[S][0], 1.0 - base[S][0])
    c, p = mm.apply_obs_fault("temperature_nan_block", clean.case, 1)
    assert int(np.isnan(c.layers[T][0]).sum()) == p["size"] ** 2
    assert c.layers[T][1].all()  # coverage mask deliberately NOT updated
    c, _ = mm.apply_obs_fault("slope_layer_empty", clean.case, 1)
    assert np.isnan(c.layers[G][0]).all() and not c.layers[G][1].any()
    c, _ = mm.apply_obs_fault("layers_swapped_temp_slope", clean.case, 1)
    assert np.array_equal(c.layers[T][0], base[G][0]) and np.array_equal(c.layers[G][0], base[T][0])
    c, p = mm.apply_obs_fault("temperature_spatial_roll", clean.case, 1)
    assert np.array_equal(c.layers[T][0], np.roll(base[T][0], p["shift_px"], axis=1))
    assert np.array_equal(c.layers[S][0], base[S][0])  # only temperature moved
    c, _ = mm.apply_obs_fault("temperature_offset_small", clean.case, 1)
    assert np.allclose(c.layers[T][0] - base[T][0], 3.0)
    for fid in mm.MULTIMODAL_V1_FAULT_IDS:  # deterministic per seed
        a, pa = mm.apply_obs_fault(fid, clean.case, 9)
        b, pb = mm.apply_obs_fault(fid, clean.case, 9)
        assert pa == pb and all(
            np.array_equal(a.layers[k][0], b.layers[k][0], equal_nan=True) for k in a.layers
        )
    assert before == {k: v[0].tobytes() for k, v in clean.case.layers.items()}  # source untouched
    with pytest.raises(KeyError):
        mm.apply_obs_fault("nope", clean.case, 0)


def test_every_obs_fault_has_documented_preregistered_expectation():
    for spec, _ in mm.OBS_FAULTS.values():
        assert spec.intent
        if spec.is_fault:
            assert isinstance(spec.expected_detectable, bool) and spec.detectable_rationale
        else:
            assert spec.expected_detectable is None
    assert sum(1 for s, _ in mm.OBS_FAULTS.values() if not s.is_fault) == 4
    assert sum(1 for s, _ in mm.OBS_FAULTS.values() if s.is_fault) == 8


# ------------------------------------------------------------------------- freezing
def test_suite_is_reproducible_versioned_and_separate_from_p3(tiles, tmp_path):
    scen, ev = tmp_path / "s", tmp_path / "e"
    p3 = scen / fixtures.SUITE_ID  # a pre-existing frozen P3 suite directory
    p3.mkdir(parents=True)
    (p3 / "manifest.json").write_text('{"p3": "frozen"}')
    before = (p3 / "manifest.json").read_bytes()
    m1 = mm.build_multimodal_suite(tiles, scen, ev, salt=SALT, suite_seed=5)
    assert (p3 / "manifest.json").read_bytes() == before  # P3 suite untouched
    assert sorted(x.name for x in p3.iterdir()) == ["manifest.json"]
    assert mm.SUITE_ID != fixtures.SUITE_ID and m1.suite_id == mm.SUITE_ID
    m2 = mm.build_multimodal_suite(tiles, tmp_path / "s2", tmp_path / "e2", salt=SALT, suite_seed=5)
    assert m1.body_sha256() == m2.body_sha256()
    assert [c.files for c in m1.cases] == [c.files for c in m2.cases]
    m3 = mm.build_multimodal_suite(tiles, tmp_path / "s3", tmp_path / "e3", salt=SALT, suite_seed=6)
    assert m3.body_sha256() != m1.body_sha256()
    with pytest.raises(fixtures.SuiteAlreadyFrozen):
        mm.build_multimodal_suite(tiles, scen, ev, salt=SALT, suite_seed=5)
    with pytest.raises(ValueError):
        mm.build_multimodal_suite(tiles[:5], tmp_path / "x", tmp_path / "y", salt=SALT)
    with pytest.raises(ValueError):
        mm.build_multimodal_suite(tiles, tmp_path / "z", tmp_path / "z", salt=SALT)


def test_manifest_structure_cases_and_independence(suite):
    m, scen, ev = suite
    assert len(m.cases) == 12 and m.detection_case_ids == [f"case-{i:02d}" for i in range(1, 13)]
    assert m.independence["independent_source_tiles"] == 12
    assert len({c.source_group for c in m.cases}) == 12
    assert m.source["dataset_revision"] == "78040503ceaa68107829f48114774112e8ee940d"
    assert m.secondary_task == {}
    assert mm.verify_multimodal_suite(scen, ev) == []
    sealed = fixtures.load_sealed(ev, m)
    assert sum(sealed.labels.values()) == 8 and len(sealed.labels) == 12
    tiles_used = {t["source_tile"] for t in sealed.cases.values()}
    assert len(tiles_used) == 12  # one distinct tile per case


# -------------------------------------------------------------------------- firewall
def test_agent_visible_data_contains_no_truth_tiles_paths_or_pro(suite):
    m, scen, ev = suite
    sealed = fixtures.load_sealed(ev, m)
    secrets_ = {t["fault_id"] for t in sealed.cases.values()}
    secrets_ |= {t["source_tile"] for t in sealed.cases.values()}
    secrets_ |= {str(ev), str(SOM), str(SMOKE)}
    manifest_text = (scen / sid() / "manifest.json").read_text()
    for tok in sorted(secrets_) + ["fault_id", "is_fault", "expected_detectable", "suite_seed"]:
        assert tok.lower() not in manifest_text.lower(), tok
    blob_parts = []
    for c in m.cases:
        d = scen / sid() / c.case_id
        assert sorted(p.name for p in d.iterdir()) == sorted(c.files)
        blob_parts.append((d / "meta.json").read_text())
        s = mm.load_multimodal_scenario(scen, sid(), c.case_id)
        blob_parts += [
            json.dumps(s.public_dict()),
            s.context.model_dump_json(),
            repr(s.arrays.layers),
        ]
        assert s.context.source_mode is SourceMode.OBSERVATION_ONLY
        assert set(s.context.available_modalities) == set(mm.LAYER_ORDER)
        assert "nasa_ibm_prediction_raw" not in s.arrays.layers  # no prediction in this suite
    blob = "\n".join(blob_parts)
    for tok in sorted(secrets_) + ["reference", "sealed", "salt", "seed", "fault", "expected_"]:
        assert tok.lower() not in blob.lower(), tok
    assert not re.search(r"\bPRO\b", blob) and "_PRO" not in blob
    for root in (scen, ev):  # the expert target is never written by this suite
        assert not [p for p in Path(root).rglob("*") if "PRO" in p.name.split(".")[0].split("_")]
    for p in (scen / sid()).rglob("*.npy"):
        raw = p.read_bytes()
        assert not any(t.encode() in raw for t in secrets_ if not t.startswith("/"))


def test_scenarios_load_with_sealed_root_deleted(suite, tmp_path):
    m, scen, ev = suite
    s2 = tmp_path / "scen_copy"
    shutil.copytree(scen, s2)  # agent side has only the scenarios root
    r = {
        c: mm.run_multimodal_baseline(mm.load_multimodal_scenario(s2, sid(), c)).flagged
        for c in m.detection_case_ids
    }
    r2 = {
        c: mm.run_multimodal_baseline(mm.load_multimodal_scenario(scen, sid(), c)).flagged
        for c in m.detection_case_ids
    }
    assert r == r2


def test_tamper_detection(tiles, tmp_path):
    m = mm.build_multimodal_suite(tiles, tmp_path / "s", tmp_path / "e", salt=SALT, suite_seed=5)
    scen, ev = tmp_path / "s", tmp_path / "e"
    p = scen / sid() / "case-04" / f"{mm.TEMP}.values.npy"
    b = bytearray(p.read_bytes())
    b[-5] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(fixtures.SuiteIntegrityError):
        mm.load_multimodal_scenario(scen, sid(), "case-04")
    assert any("case-04" in x for x in mm.verify_multimodal_suite(scen, ev))
    (scen / sid() / "case-05" / "meta.json").unlink()
    assert any("case-05" in x for x in mm.verify_multimodal_suite(scen, ev))
    key = ev / sid() / "sealed_key.json"
    key.write_text(key.read_text().replace('"is_fault":true', '"is_fault":false', 1))
    with pytest.raises(fixtures.SuiteIntegrityError):
        fixtures.load_sealed(ev, m)


# --------------------------------------------------------------------------- baseline
def test_baseline_deterministic_and_config_hashed(suite):
    m, scen, _ = suite
    run = lambda: {  # noqa: E731
        c: mm.run_multimodal_baseline(mm.load_multimodal_scenario(scen, sid(), c))
        for c in m.detection_case_ids
    }
    a, b = run(), run()
    assert {c: (r.flagged, r.tool_run_ids) for c, r in a.items()} == {
        c: (r.flagged, r.tool_run_ids) for c, r in b.items()
    }
    assert mm.ObsBaselineConfig().sha256() == mm.ObsBaselineConfig().sha256()
    assert mm.ObsBaselineConfig(temp_min_k=1.0).sha256() != mm.ObsBaselineConfig().sha256()


def test_natural_nodata_tile_is_not_flagged_as_a_fault(tiles, tmp_path):
    nat = [t for t in tiles if t.tile_id == "patch_0000_0003_S_80S"]
    assert not nat[0].case.layers[mm.TEMP][1].all()  # genuinely has uncovered pixels
    ids = ("clean_identical",)
    mm.build_multimodal_suite(nat, tmp_path / "s", tmp_path / "e", salt=SALT, fault_ids=ids)
    r = mm.run_multimodal_baseline(mm.load_multimodal_scenario(tmp_path / "s", sid(), "case-01"))
    assert not r.flagged, r.rules


def test_baseline_pinned_preliminary_results_including_honest_imperfections(suite):
    m, scen, ev = suite
    results = {
        c: mm.run_multimodal_baseline(mm.load_multimodal_scenario(scen, sid(), c))
        for c in m.detection_case_ids
    }
    frozen = score.freeze_detections(m, {c: r.flagged for c, r in results.items()})
    rep = score.score_detections(frozen, scen, ev, sid())
    by = {r["fault_id"]: r for r in rep["per_case"]}
    # pinned outcome of THIS frozen suite/seed/config (tile assignment depends on the seed)
    assert (rep["tp"], rep["fp"], rep["fn"], rep["tn"]) == (7, 1, 1, 3)
    assert rep["precision"] == pytest.approx(0.875) and rep["recall"] == pytest.approx(0.875)
    assert rep["f1"] == pytest.approx(0.875) and rep["false_positive_rate"] == pytest.approx(0.25)
    assert rep["n_cases"] == 12 and rep["unit"] == "case"
    assert by["temperature_offset_small"]["outcome"] == "FN"  # pre-registered miss
    # honest imperfection 1: the shadow tolerance (1e-6) equals the control's noise (1e-6 sigma)
    assert by["clean_quantization_noise"]["outcome"] == "FP"
    # honest imperfection 2: a fault pre-registered as undetectable WAS flagged
    assert rep["expectation_check"]["mismatches_vs_preregistered"] == ["temperature_spatial_roll"]
    assert by["temperature_spatial_roll"]["outcome"] == "TP"
    for fid in (
        "temperature_unit_celsius",
        "slope_unit_radians",
        "psr_mask_inverted",
        "temperature_nan_block",
        "slope_layer_empty",
    ):
        assert by[fid]["outcome"] == "TP", fid
    flagged_rules = {
        by_case["fault_id"]: results[by_case["case_id"]].triggered_rules
        for by_case in rep["per_case"]
    }
    assert "temperature_range" in flagged_rules["temperature_unit_celsius"]
    assert "slope_range" in flagged_rules["slope_unit_radians"]
    assert "psr_temperature_sign" in flagged_rules["psr_mask_inverted"]
    assert "missing_data" in flagged_rules["temperature_nan_block"]
    assert "layer_empty" in flagged_rules["slope_layer_empty"]
    assert flagged_rules["clean_quantization_noise"] == ["shadow_range"]
    for clean in ("clean_identical", "clean_identical_second", "clean_float32_roundtrip"):
        assert flagged_rules[clean] == []


def test_p3_baseline_surface_unchanged():
    # the P3 baseline config and rule set are untouched by P4
    cfg = baselines.BaselineConfig()
    assert (cfg.max_plausible, cfg.min_dynamic_range, cfg.footprint_tol_deg) == (1.5, 0.2, 1e-4)
    assert mm.ObsBaselineConfig().shadow_tol == 1e-6
