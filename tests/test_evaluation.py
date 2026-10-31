"""P3 tests: sealed boundary, deterministic faults, frozen suite, baseline and scoring.

Most tests use a SMALL SYNTHETIC source (labelled as such); the final section runs the real
published sector end to end.
"""

import ast
import inspect
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from iceclaw.config import Settings
from iceclaw.contracts import AgentRegionContext
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.evaluation import baselines, faults, fixtures, score
from iceclaw.evaluation.faults import FAULTS, CaseData, apply_fault
from iceclaw.science.model_audit import TOOLS

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "iceclaw"
SMOKE = Path(
    os.environ.get(
        "ICECLAW_MOONATLAS_SMOKE_ROOT",
        REPO / ".upstream/moonatlas-science/examples/science-smoke",
    )
)
SALT = b"unit-test-salt-0123"
PROJ = {
    "kind": "polar-stereographic",
    "latitudeOfOriginDeg": -90,
    "centralMeridianDeg": 0,
    "scaleFactor": 0.994,
    "falseEastingM": 500000.0,
    "falseNorthingM": 500000.0,
    "radiusM": 1737400.0,
}
GRID = {"xMinM": 320000.0, "yMaxM": 740000.0, "pixelSizeXM": 240.0, "pixelSizeYM": 240.0}
SHA = "a" * 64


def synthetic_source(n: int = 64) -> fixtures.SourceBundle:
    """SYNTHETIC smooth field + noise with metadata that is self-consistent."""
    rng = np.random.default_rng(1)
    rr, cc = np.mgrid[0:n, 0:n]
    field = 0.5 + 0.3 * np.sin(rr / 9.0) * np.cos(cc / 13.0) + rng.normal(0, 0.01, (n, n))
    ref = field + rng.normal(0, 0.02, (n, n))  # expert-reference stand-in (synthetic)
    ref[0:16, 0:16] += 0.3  # one strongly disagreeing window (window_px=16)
    corners = ((0, 0), (n, 0), (n, n), (0, n))
    fp = tuple(ma.pixel_to_lat_lon(PROJ, GRID, c, r) for c, r in corners)
    grid = {**GRID, "widthPx": n, "heightPx": n}
    data = CaseData(field, np.ones((n, n), bool), fp, grid, dict(PROJ), 240.0)
    return fixtures.SourceBundle(
        "synthetic-region",
        "south",
        "upstream_published_example",  # enum value only; the data here are SYNTHETIC
        "synthetic-ds",
        "rev0",
        "test",
        SHA,
        data,
        ref,
        SHA,
    )


def build(tmp_path, source=None, seed=7, salt=SALT, **kw):
    source = source or synthetic_source()
    scen, ev = tmp_path / "scenarios", tmp_path / "evaluation"
    manifest = fixtures.build_suite(
        source, scen, ev, suite_seed=seed, salt=salt, window_px=16, top_k=4, **kw
    )
    return source, manifest, scen, ev


def sid():
    return fixtures.SUITE_ID


# ------------------------------------------------------------------------------- faults
def test_faults_are_deterministic_for_a_seed_and_vary_across_seeds():
    src = synthetic_source().data
    for fid in FAULTS:
        a, pa = apply_fault(fid, src, 123)
        b, pb = apply_fault(fid, src, 123)
        assert pa == pb
        assert np.array_equal(a.values, b.values, equal_nan=True) and a.footprint == b.footprint
    a, pa = apply_fault("value_nan_block", src, 1)
    b, pb = apply_fault("value_nan_block", src, 2)
    assert pa != pb


def test_faults_never_modify_the_source():
    bundle = synthetic_source()
    before = (bundle.data.values.tobytes(), bundle.data.covered.tobytes(), dict(bundle.data.grid))
    for fid in FAULTS:
        apply_fault(fid, bundle.data, 5)
    after = (bundle.data.values.tobytes(), bundle.data.covered.tobytes(), dict(bundle.data.grid))
    assert before == after
    with pytest.raises(ValueError):  # source arrays are read-only
        bundle.data.values[0, 0] = 9.0


def test_clean_controls_are_unchanged_within_declared_tolerance():
    src = synthetic_source().data
    same, _ = apply_fault("clean_identical", src, 0)
    assert np.array_equal(same.values, src.values) and same.footprint == src.footprint
    f32, _ = apply_fault("clean_float32_roundtrip", src, 0)
    assert np.abs(f32.values - src.values).max() < 1e-7
    noisy, _ = apply_fault("clean_quantization_noise", src, 0)
    assert np.abs(noisy.values - src.values).max() < 1e-5


def test_fault_effects_match_their_declared_intent():
    src = synthetic_source().data
    x100, _ = apply_fault("value_unit_scale_x100", src, 0)
    assert np.allclose(x100.values, src.values * 100)
    nan, p = apply_fault("value_nan_block", src, 0)
    assert int(np.isnan(nan.values).sum()) == p["size"] ** 2 and nan.covered.all()
    sat, p = apply_fault("value_saturation_block", src, 0)
    assert (sat.values == 5.0).sum() == p["size"] ** 2
    off, _ = apply_fault("value_offset_small", src, 0)
    assert np.allclose(off.values - src.values, 0.05)
    roll, p = apply_fault("value_spatial_roll", src, 0)
    assert np.array_equal(roll.values, np.roll(src.values, p["shift_px"], axis=1))
    fp, _ = apply_fault("metadata_footprint_shift", src, 0)
    assert fp.footprint[0][0] == pytest.approx(src.footprint[0][0] + 0.4)
    g, _ = apply_fault("metadata_grid_origin_shift", src, 0)
    assert g.grid["xMinM"] == src.grid["xMinM"] + 25 * src.grid["pixelSizeXM"]
    res, _ = apply_fault("metadata_resolution_unit_error", src, 0)
    assert res.resolution_m == pytest.approx(src.resolution_m * 0.1)
    with pytest.raises(KeyError):
        apply_fault("nope", src, 0)


def test_every_fault_is_documented_with_a_preregistered_expectation():
    for spec, _ in FAULTS.values():
        assert spec.intent
        if spec.is_fault:
            assert isinstance(spec.expected_detectable, bool) and spec.detectable_rationale
        else:
            assert spec.expected_detectable is None
    clean = [s for s, _ in FAULTS.values() if not s.is_fault]
    assert len(clean) >= 1
    assert any(not s.expected_detectable for s, _ in FAULTS.values() if s.is_fault)


# -------------------------------------------------------------------------- frozen suite
def test_suite_is_reproducible_with_same_seed_and_salt(tmp_path):
    _, m1, s1, _ = build(tmp_path / "a")
    _, m2, s2, _ = build(tmp_path / "b")
    assert m1.body_sha256() == m2.body_sha256()
    assert [c.files for c in m1.cases] == [c.files for c in m2.cases]
    _, m3, _, _ = build(tmp_path / "c", seed=8)
    assert m3.body_sha256() != m1.body_sha256()
    _, m4, _, _ = build(tmp_path / "d", salt=b"another-salt-0000")
    assert m4.sealed_commitment != m1.sealed_commitment


def test_case_ids_are_stable_opaque_and_counted_per_case(tmp_path):
    _, m, _, _ = build(tmp_path)
    ids = m.detection_case_ids
    assert ids == [f"case-{i:02d}" for i in range(1, len(faults.P3_SUITE_V1_FAULT_IDS) + 1)]
    assert len(m.cases) == len(ids) + 1  # + the ranking case
    assert m.independence["independent_source_tiles"] == 1
    assert m.independence["statistical_unit"] == "case"
    assert len({c.case_id for c in m.cases}) == len(m.cases)


def test_manifest_hash_file_and_tamper_detection(tmp_path):
    _, m, scen, ev = build(tmp_path)
    assert fixtures.verify_suite(scen, ev) == []
    assert fixtures.load_manifest(scen).body_sha256() == m.body_sha256()
    mf = scen / sid() / "manifest.json"
    mf.write_text(mf.read_text().replace('"suite_version":"1"', '"suite_version":"2"'))
    with pytest.raises(fixtures.SuiteIntegrityError):
        fixtures.load_manifest(scen)
    assert fixtures.verify_suite(scen, ev)


def test_tampered_case_file_is_rejected(tmp_path):
    _, m, scen, ev = build(tmp_path)
    p = scen / sid() / "case-03" / "values.npy"
    b = bytearray(p.read_bytes())
    b[-3] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(fixtures.SuiteIntegrityError):
        fixtures.load_scenario(scen, sid(), "case-03")
    assert any("case-03" in x for x in fixtures.verify_suite(scen, ev))


def test_sealed_key_and_reference_tamper_detected(tmp_path):
    _, m, scen, ev = build(tmp_path)
    key = ev / sid() / "sealed_key.json"
    key.write_text(key.read_text().replace('"is_fault":true', '"is_fault":false', 1))
    with pytest.raises(fixtures.SuiteIntegrityError):
        fixtures.load_sealed(ev, m)
    _, m, scen, ev = build(tmp_path / "2")
    ref = ev / sid() / "reference.npy"
    b = bytearray(ref.read_bytes())
    b[-1] ^= 0x01
    ref.write_bytes(bytes(b))
    sealed = fixtures.load_sealed(ev, m)
    with pytest.raises(fixtures.SuiteIntegrityError):
        fixtures.load_sealed_reference(ev, m, sealed)


def test_frozen_suite_cannot_be_overwritten_and_roots_must_differ(tmp_path):
    source, _, scen, ev = build(tmp_path)
    with pytest.raises(fixtures.SuiteAlreadyFrozen):
        fixtures.build_suite(source, scen, ev, salt=SALT, window_px=16, top_k=4)
    with pytest.raises(ValueError):
        fixtures.build_suite(source, tmp_path / "x", tmp_path / "x", salt=SALT, window_px=16)


# ----------------------------------------------------------------------------- firewall
FORBIDDEN_TOKENS = [
    "reference",
    "sealed",
    "salt",
    "seed",
    "fault",
    "evaluation",
    "expected_",
]


def test_no_hidden_truth_in_agent_visible_files_or_objects(tmp_path):
    _, m, scen, ev = build(tmp_path)
    sealed = fixtures.load_sealed(ev, m)
    truth_tokens = {t["fault_id"] for t in sealed.cases.values()} | {str(ev), str(ev.name)}
    visible_text = []
    for cid in [c.case_id for c in m.cases]:
        d = scen / sid() / cid
        assert sorted(p.name for p in d.iterdir()) == ["covered.npy", "meta.json", "values.npy"]
        visible_text.append((d / "meta.json").read_text())
        s = fixtures.load_scenario(scen, sid(), cid)
        visible_text.append(json.dumps(s.public_dict()))
        visible_text.append(s.context.model_dump_json())
        visible_text.append(repr(s.arrays.layers))
    blob = "\n".join(visible_text)
    for tok in FORBIDDEN_TOKENS + sorted(truth_tokens):
        assert tok.lower() not in blob.lower(), tok
    assert not re.search(r"\bPRO\b", blob)  # the expert-target layer name
    # also check the raw bytes of the numeric files for embedded strings
    for p in (scen / sid()).rglob("*.npy"):
        raw = p.read_bytes()
        for tok in sorted(truth_tokens) + ["reference", "sealed"]:
            assert tok.encode() not in raw


def test_public_manifest_has_no_labels_seed_or_sealed_paths(tmp_path):
    _, m, scen, ev = build(tmp_path)
    text = (scen / sid() / "manifest.json").read_text()
    for tok in ["fault_id", "is_fault", "expected_detectable", "suite_seed", str(ev), "salt"]:
        assert tok not in text
    for fid in faults.P3_SUITE_V1_FAULT_IDS:
        assert fid not in text
    assert "reference.npy" not in text
    assert len(m.sealed_commitment) == 64


def test_agent_context_type_has_no_private_or_truth_fields():
    private = {"reference_path", "prediction_raw_path", "source_manifest_path", "affine"}
    truth = {"fault_id", "is_fault", "seed", "label", "reference", "sealed", "eval_root"}
    fields = set(AgentRegionContext.model_fields)
    assert not (private | truth) & fields
    assert not {n for n in fields if "path" in n or "ref" in n}


def test_science_tool_arguments_cannot_name_files_or_paths():
    allowed = {
        "data", "modality", "window", "qs", "k_mad", "top_n", "edge_width_px",
        "psr_threshold", "cold_trap_k", "steep_deg",
    }  # fmt: skip
    for name, fn in TOOLS.items():
        params = set(inspect.signature(fn).parameters)
        assert params <= allowed, (name, params - allowed)
        assert not any(("path" in p or "file" in p or "dir" in p) for p in params)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def test_import_boundaries_evaluator_is_never_imported_by_agent_side_code():
    banned = "iceclaw.evaluation"
    for sub in ("science", "data", "agents", "swarm", "reports"):
        d = SRC / sub
        if not d.is_dir():
            continue
        for f in d.rglob("*.py"):
            assert not any(i.startswith(banned) for i in _imports(f)), f
    for f in (SRC / "contracts.py", SRC / "config.py"):
        assert not any(i.startswith(banned) for i in _imports(f)), f
    for f in (SRC / "evaluation").rglob("*.py"):
        assert not any(i.startswith(("iceclaw.agents", "iceclaw.swarm")) for i in _imports(f)), f


def test_baseline_and_scenarios_work_with_the_sealed_root_deleted(tmp_path):
    _, m, scen, ev = build(tmp_path)
    before = {
        c: baselines.run_baseline(fixtures.load_scenario(scen, sid(), c)).flagged
        for c in m.detection_case_ids
    }
    shutil.rmtree(ev)  # the baseline/agent side cannot depend on the evaluator root
    after = {
        c: baselines.run_baseline(fixtures.load_scenario(scen, sid(), c)).flagged
        for c in m.detection_case_ids
    }
    assert before == after


# -------------------------------------------------------------------------------- baseline
def run_all(scen, m):
    return {
        c: baselines.run_baseline(fixtures.load_scenario(scen, sid(), c))
        for c in m.detection_case_ids
    }


def test_baseline_is_deterministic_and_config_is_hashed(tmp_path):
    _, m, scen, _ = build(tmp_path)
    r1, r2 = run_all(scen, m), run_all(scen, m)
    assert {c: (r.flagged, r.tool_run_ids) for c, r in r1.items()} == {
        c: (r.flagged, r.tool_run_ids) for c, r in r2.items()
    }
    assert baselines.BaselineConfig().sha256() == baselines.BaselineConfig().sha256()
    assert (
        baselines.BaselineConfig(max_plausible=2.0).sha256() != baselines.BaselineConfig().sha256()
    )


def test_baseline_outcomes_on_synthetic_suite(tmp_path):
    _, m, scen, ev = build(tmp_path)
    results = run_all(scen, m)
    sealed = fixtures.load_sealed(ev, m)
    by_fault = {sealed.cases[c]["fault_id"]: results[c] for c in results}
    for clean in ("clean_identical", "clean_float32_roundtrip", "clean_quantization_noise"):
        assert not by_fault[clean].flagged, by_fault[clean].rules
    expect = {
        "value_unit_scale_x100": "value_range",
        "value_unit_scale_div100": "dynamic_range",
        "value_nan_block": "missing_data",
        "value_saturation_block": "value_range",
        "metadata_footprint_shift": "geometry_consistency",
        "metadata_grid_origin_shift": "geometry_consistency",
        "metadata_resolution_unit_error": "geometry_consistency",
    }
    for fid, rule in expect.items():
        assert rule in by_fault[fid].triggered_rules, (fid, by_fault[fid].rules)
    assert not by_fault["value_offset_small"].flagged  # pre-registered miss


# -------------------------------------------------------------------------------- scoring
def test_detection_metrics_known_values_and_undefined_denominators():
    labels = {"a": True, "b": True, "c": False, "d": False}
    flags = {"a": True, "b": False, "c": True, "d": False}
    m = score.detection_metrics(flags, labels)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["false_positive_rate"] == 0.5 and m["unit"] == "case" and m["n_cases"] == 4
    # no flags at all: precision undefined, NOT 1.0 or 0.0
    none = score.detection_metrics({"a": False, "b": False}, {"a": True, "b": False})
    assert none["precision"] is None and none["recall"] == 0.0 and none["f1"] == 0.0
    # no fault cases: recall/f1 undefined; no clean cases: FPR undefined
    clean_only = score.detection_metrics({"a": False}, {"a": False})
    assert clean_only["recall"] is None and clean_only["f1"] is None
    assert clean_only["false_positive_rate"] == 0.0
    fault_only = score.detection_metrics({"a": True}, {"a": True})
    assert fault_only["false_positive_rate"] is None and fault_only["f1"] == 1.0
    assert score.detection_metrics({}, {})["f1"] is None
    with pytest.raises(ValueError):
        score.detection_metrics({"a": True}, {"b": True})


def test_freeze_requires_exact_case_set_and_binds_to_manifest(tmp_path):
    _, m, scen, ev = build(tmp_path)
    flags = {c: False for c in m.detection_case_ids}
    f = score.freeze_detections(m, flags)
    assert f.suite_manifest_sha256 == m.body_sha256() and len(f.sha256) == 64
    assert score.freeze_detections(m, flags).sha256 == f.sha256
    with pytest.raises(ValueError):
        score.freeze_detections(m, {**flags, "case-99": True})
    short = dict(flags)
    short.pop("case-01")
    with pytest.raises(ValueError):
        score.freeze_detections(m, short)
    # frozen output from a DIFFERENT suite cannot be scored against this one
    _, m2, _, _ = build(tmp_path / "other", seed=99)
    other = score.freeze_detections(m2, {c: False for c in m2.detection_case_ids})
    with pytest.raises(fixtures.SuiteIntegrityError):
        score.score_detections(other, scen, ev, sid())


def test_score_detections_matches_manual_counts_and_checks_expectations(tmp_path):
    _, m, scen, ev = build(tmp_path)
    results = run_all(scen, m)
    frozen = score.freeze_detections(m, {c: r.flagged for c, r in results.items()})
    rep = score.score_detections(frozen, scen, ev, sid())
    sealed = fixtures.load_sealed(ev, m)
    manual = score.detection_metrics(frozen.as_dict(), sealed.labels)
    assert {k: rep[k] for k in manual} == manual
    assert rep["n_cases"] == len(m.detection_case_ids)  # case, not pixel, counts
    assert rep["n_clean_cases"] == 3 and rep["fp"] == 0
    assert rep["confidence_intervals"] is None
    assert all(r["outcome"] in {"TP", "FP", "FN", "TN"} for r in rep["per_case"])


def test_window_grid_and_topk_recovery_known_values():
    w = fixtures.window_grid((64, 64), 16)
    assert len(w) == 16 and w[0] == (0, 16, 0, 16) and w[5] == (16, 32, 16, 32)
    with pytest.raises(ValueError):
        fixtures.window_grid((16, 16), 16)
    res = np.array([0.1, 0.9, 0.5, 0.2, 0.8, 0.0])
    perfect = score.topk_recovery(res, [1, 4], 2)
    assert perfect["top_k_recovery"] == 1.0 and perfect["cumulative_residual_fraction"] == 1.0
    half = score.topk_recovery(res, [1, 0], 2)
    assert half["top_k_recovery"] == 0.5
    assert half["cumulative_residual_fraction"] == pytest.approx((0.9 + 0.1) / (0.9 + 0.8))
    assert half["random_expectation"] == pytest.approx(2 / 6)
    with pytest.raises(ValueError):
        score.topk_recovery(res, [1, 1], 2)
    zero = score.topk_recovery(np.zeros(4), [0, 1], 2)
    assert zero["cumulative_residual_fraction"] is None  # 0/0 is undefined, not 1


def test_window_residuals_ignore_nonfinite():
    pred = np.zeros((4, 4))
    ref = np.zeros((4, 4))
    ref[:2, :2] = 1.0
    ref[2, 2] = np.nan
    out = score.window_residuals(pred, ref, [[0, 2, 0, 2], [2, 4, 2, 4]])
    assert out[0] == 1.0 and out[1] == 0.0


def test_ranking_task_end_to_end_synthetic(tmp_path):
    _, m, scen, ev = build(tmp_path)
    sec = m.secondary_task
    scenario = fixtures.load_scenario(scen, sid(), sec["case_id"])
    ranking = baselines.rank_windows_baseline(scenario, sec["windows"], sec["top_k"])
    assert len(ranking) == 4 and len(set(ranking)) == 4
    assert ranking == baselines.rank_windows_baseline(scenario, sec["windows"], sec["top_k"])
    ref = fixtures.load_sealed_reference(ev, m, fixtures.load_sealed(ev, m))
    pred = next(iter(scenario.arrays.layers.values())).values
    res = score.window_residuals(pred, ref, sec["windows"])
    truth = sorted(range(len(res)), key=lambda i: (-res[i], i))[:4]
    assert 0 in truth  # the +0.3 disagreement window is the largest by construction
    perfect = score.score_ranking(score.freeze_ranking(m, truth), scen, ev, sid())
    assert perfect["top_k_recovery"] == 1.0 and perfect["cumulative_residual_fraction"] == 1.0
    worst = [i for i in range(16) if i not in truth][:4]
    bad = score.score_ranking(score.freeze_ranking(m, worst), scen, ev, sid())
    assert bad["hits"] == 0 and bad["top_k_recovery"] == 0.0
    with pytest.raises(ValueError):
        score.freeze_ranking(m, [0, 1, 2])
    with pytest.raises(ValueError):
        score.freeze_ranking(m, [0, 1, 2, 99])


def test_ranking_case_prediction_is_the_untouched_source(tmp_path):
    source, m, scen, _ = build(tmp_path)
    sc = fixtures.load_scenario(scen, sid(), m.secondary_task["case_id"])
    layer = next(iter(sc.arrays.layers.values()))
    assert np.array_equal(layer.values, source.data.values)


# -------------------------------------------------------------- real published sector
needs_upstream = pytest.mark.skipif(
    not (SMOKE / "manifest.json").is_file(), reason="MOONATLAS smoke export not checked out"
)


@pytest.fixture(scope="module")
def real_suite(tmp_path_factory):
    root = tmp_path_factory.mktemp("real")
    settings = Settings(
        data_root=Path("/nonexistent-iceclaw-data"),
        runs_root=Path("/nonexistent-iceclaw-runs"),
        moonatlas_roots=(SMOKE,),
    )
    source = fixtures.load_real_source_bundle(settings, "ice-s-0001-0002")
    manifest = fixtures.build_suite(source, root / "s", root / "e", salt=SALT, suite_seed=11)
    return source, manifest, root / "s", root / "e", settings


@needs_upstream
def test_real_source_is_authentic_and_untouched(real_suite):
    source, m, scen, ev, _ = real_suite
    assert source.prediction_sha256 == ma.sha256_file(
        SMOKE / "overlays/ice/ice-s-0001-0002-raw.webp"
    )
    assert source.data.values.shape == (256, 256) and not source.data.values.flags.writeable
    assert np.nanmean(source.reference) == pytest.approx(0.406365, abs=2e-5)  # upstream stat
    assert m.source["source_split"] == "test" and m.source["region_id"] == "ice-s-0001-0002"
    assert fixtures.verify_suite(scen, ev) == []
    # clean-identical and ranking copies equal the authentic prediction exactly
    sealed = fixtures.load_sealed(ev, m)
    cid = next(c for c, t in sealed.cases.items() if t["fault_id"] == "clean_identical")
    sc = fixtures.load_scenario(scen, sid(), cid)
    assert np.array_equal(next(iter(sc.arrays.layers.values())).values, source.data.values)


@needs_upstream
def test_real_suite_baseline_scores_are_reproducible_and_pinned(real_suite):
    source, m, scen, ev, _ = real_suite
    results = run_all(scen, m)
    frozen = score.freeze_detections(m, {c: r.flagged for c, r in results.items()})
    rep = score.score_detections(frozen, scen, ev, sid())
    assert (rep["tp"], rep["fp"], rep["fn"], rep["tn"]) == (7, 0, 2, 3)
    assert rep["precision"] == 1.0 and rep["recall"] == pytest.approx(7 / 9)
    assert rep["f1"] == pytest.approx(0.875) and rep["false_positive_rate"] == 0.0
    assert rep["expectation_check"]["mismatches_vs_preregistered"] == []
    missed = {r["fault_id"] for r in rep["per_case"] if r["outcome"] == "FN"}
    assert missed == {"value_offset_small", "value_spatial_roll"}
    # rebuilding from the same authentic source, seed and salt reproduces the frozen hash
    other = fixtures.build_suite(
        source, scen.parent / "s2", scen.parent / "e2", salt=SALT, suite_seed=11
    )
    assert other.body_sha256() == m.body_sha256()


@needs_upstream
def test_real_ranking_baseline_vs_random_expectation(real_suite):
    _, m, scen, ev, _ = real_suite
    sec = m.secondary_task
    sc = fixtures.load_scenario(scen, sid(), sec["case_id"])
    ranking = baselines.rank_windows_baseline(sc, sec["windows"], sec["top_k"])
    out = score.score_ranking(score.freeze_ranking(m, ranking), scen, ev, sid())
    assert out["n_windows"] == 64 and out["k"] == 8 and out["random_expectation"] == 0.125
    assert (out["hits"], out["top_k_recovery"]) == (2, 0.25)
    assert out["cumulative_residual_fraction"] == pytest.approx(0.857772, abs=1e-5)
