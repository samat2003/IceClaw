"""Research triage: independent reference, boundaries, ties, mutations, artifact verification.

Synthetic arrays only (no lunar claim). The reference implementation below is deliberately written
a different way (pure-Python loops and a hand-rolled percentile) so it does not share logic with
``science.triage``. The real-data test at the end skips if the preserved public bundle is absent.
"""

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from agent_helpers import GRID, PROJ, SHA
from bundle_helpers import scripted_service

from iceclaw.cli import main
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.reports import package as pkg
from iceclaw.reports import triage_artifact as ta
from iceclaw.reports.verify import verify_bundle
from iceclaw.science import triage as tg
from iceclaw.science.common import RasterLayer, RegionArrays

REPO = Path(__file__).resolve().parents[1]
SAMPLE_PUBLIC = REPO / "runs" / "inv-20261031T201513Z-p6-public"
UNITS = {tg.PRED: "prospectivity_index", tg.TEMP: "K", tg.SHADOW: "fraction", tg.TERRAIN: "deg"}


def region(pred, temp, shadow, slope, covered=None, units=None, footprint_shift=0.0, grid=None):
    h, w = pred.shape
    cov = np.ones((h, w), bool) if covered is None else covered
    vals = {tg.PRED: pred, tg.TEMP: temp, tg.SHADOW: shadow, tg.TERRAIN: slope}
    units = units or UNITS
    layers = {
        n: RasterLayer(n, np.array(v, float), cov.copy(), units[n], f"syn:{n}", SHA[n])
        for n, v in vals.items()
    }
    g = {**GRID, "widthPx": w, "heightPx": h} if grid is None else grid
    corners = ((0, 0), (w, 0), (w, h), (0, h))
    fp = tuple(
        (lat + footprint_shift, lon)
        for lat, lon in (ma.pixel_to_lat_lon(PROJ, GRID, c, r) for c, r in corners)
    )
    return RegionArrays("synthetic", fp, 240.0, PROJ, g, layers)


def structured(n=32):
    """Random background that can never qualify, plus two known 8x8 blocks (window 8):
    F = window id 9 (rows 16-24, cols 8-16): high index, shadowed, cold, gentle  -> follow-up
    R = window id 5 (rows 8-16,  cols 8-16): high index, shadowed, NOT cold, gentle -> review."""
    rng = np.random.default_rng(5)
    pred = rng.uniform(0.0, 1.0, (n, n))
    temp = rng.uniform(120.0, 300.0, (n, n))  # never <= 110 K
    shadow = rng.choice([0.0, 0.2, 0.4], (n, n))  # never >= 0.5
    slope = rng.uniform(16.0, 30.0, (n, n))  # never <= 15 deg
    for r0, temp_v in ((16, 90.0), (8, 200.0)):
        pred[r0 : r0 + 8, 8:16] = 1.2
        shadow[r0 : r0 + 8, 8:16] = 1.0
        temp[r0 : r0 + 8, 8:16] = temp_v
        slope[r0 : r0 + 8, 8:16] = 5.0
    return pred, temp, shadow, slope


CFG8 = tg.TriageConfig(window_px=8)


# -------------------------------------------------------------- independent reference (pure Python)
def reference(pred, temp, shadow, slope, cov, cfg):
    h, w = pred.shape

    def ok(a, r, c):
        return bool(cov[r, c]) and math.isfinite(a[r, c])

    vals = sorted(float(pred[r, c]) for r in range(h) for c in range(w) if ok(pred, r, c))
    pos = (len(vals) - 1) * cfg.index_percentile / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(vals) - 1)
    thr = vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)
    out = {}
    px = cfg.window_px
    for wr in range(h // px):
        for wc in range(w // px):
            n_all = n_high = n_joint = 0
            sup = {"shadow": 0, "cold": 0, "gentle": 0}
            tvals, sh_ok, sh_hit = [], 0, 0
            for r in range(wr * px, (wr + 1) * px):
                for c in range(wc * px, (wc + 1) * px):
                    if ok(temp, r, c):
                        tvals.append(float(temp[r, c]))
                    if ok(shadow, r, c):
                        sh_ok += 1
                        sh_hit += shadow[r, c] >= cfg.shadow_min
                    if not all(ok(a, r, c) for a in (pred, temp, shadow, slope)):
                        continue
                    n_all += 1
                    s = shadow[r, c] >= cfg.shadow_min
                    k = temp[r, c] <= cfg.tmax_max_k
                    g = slope[r, c] <= cfg.slope_max_deg
                    hgh = pred[r, c] >= thr
                    n_high += hgh
                    if hgh:
                        sup["shadow"] += s
                        sup["cold"] += k
                        sup["gentle"] += g
                    n_joint += hgh and s and k and g
            out[wr * (w // px) + wc] = {
                "n_all": n_all,
                "n_high": n_high,
                "n_joint": n_joint,
                "sup": {k: (v / n_high if n_high else None) for k, v in sup.items()},
                "shadow_fraction": sh_hit / sh_ok if sh_ok else None,
                "tmax_median": statistics.median(tvals) if tvals else None,
            }
    follow = sorted(
        (i for i, x in out.items() if x["n_joint"] >= 1), key=lambda i: (-out[i]["n_joint"], i)
    )
    review, reasons = [], {}
    for i, x in out.items():
        if (
            i in follow
            or not x["n_all"]
            or x["n_high"] / x["n_all"] < cfg.review_min_high_index_fraction
        ):
            continue
        weak = [
            k
            for k in ("shadow", "cold", "gentle")
            if x["sup"][k] is not None and x["sup"][k] < cfg.review_support_min_fraction
        ]
        if weak:
            review.append(i)
            reasons[i] = weak
    review.sort(key=lambda i: (-out[i]["n_high"] / out[i]["n_all"], i))
    return thr, out, follow, review, reasons


def assert_matches_reference(arrays, cfg):
    res = tg.screen_arrays(arrays, cfg, sensitivity=False)
    v = {n: arrays.layers[n].values for n in tg.LAYERS}
    thr, ref, follow, review, reasons = reference(
        v[tg.PRED], v[tg.TEMP], v[tg.SHADOW], v[tg.TERRAIN], arrays.layers[tg.PRED].covered, cfg
    )
    assert res["index_threshold_raw"] == pytest.approx(thr, rel=1e-8)
    for w in res["windows"]:
        r = ref[w["id"]]
        assert (w["n_valid_all_layers"], w["n_high_index"], w["n_joint"]) == (
            r["n_all"],
            r["n_high"],
            r["n_joint"],
        )
        for k in ("shadow", "cold", "gentle"):
            a, b = w["support_among_high_index"][k], r["sup"][k]
            assert (a is None) == (b is None) and (a is None or a == pytest.approx(b, rel=1e-8))
        if r["shadow_fraction"] is not None:
            assert w["shadow_fraction"] == pytest.approx(r["shadow_fraction"], rel=1e-8)
        if r["tmax_median"] is not None:
            assert w["tmax_k"]["median"] == pytest.approx(r["tmax_median"], rel=1e-8)
    assert res["selection"]["followup_ids"] == follow
    assert res["selection"]["review_ids"] == review
    assert {int(k): v for k, v in res["selection"]["review_reasons"].items()} == reasons
    return res


# ======================================================================= the screen itself
def test_hand_known_example_selects_exactly_the_planted_windows():
    res = tg.screen_arrays(region(*structured()), CFG8)
    sel, c = res["selection"], res["counts"]
    assert res["index_threshold_raw"] == pytest.approx(1.2)
    assert sel["followup_ids"] == [9] and sel["review_ids"] == [5]
    assert sel["review_reasons"] == {"5": ["cold"]}
    assert (c["n_followup_windows"], c["n_review_windows"], c["n_joint_pixels"]) == (1, 1, 64)
    w9 = tg.window_by_id(res, 9)
    assert w9["window"] == [16, 24, 8, 16] and w9["joint_fraction"] == 1.0
    assert w9["tmax_k"]["max"] == 90.0 and w9["slope_deg"]["max"] == 5.0
    assert tg.window_category(res, 9) == "followup" and tg.window_category(res, 5) == "review"
    assert tg.window_category(res, 0) == "other"
    assert c["pixels_cumulative"] == {
        "high_index": 128, "high_and_shadow": 128, "high_shadow_cold": 64, "all_four": 64,
    }  # fmt: skip
    assert res["messages"] == []


def test_matches_an_independent_reference_on_clean_masked_and_nonfinite_data():
    assert_matches_reference(region(*structured()), CFG8)
    pred, temp, shadow, slope = structured()
    rng = np.random.default_rng(9)
    cov = rng.random(pred.shape) > 0.08  # uncovered pixels
    temp[rng.random(pred.shape) < 0.05] = np.nan  # non-finite but 'covered'
    slope[rng.random(pred.shape) < 0.05] = np.inf
    pred[rng.random(pred.shape) < 0.03] = np.nan
    res = assert_matches_reference(region(pred, temp, shadow, slope, covered=cov), CFG8)
    assert res["counts"]["n_valid_all_layer_pixels"] < pred.size
    # other grids / window sizes / percentiles agree too
    for cfg in (
        tg.TriageConfig(window_px=4, index_percentile=75.0, tmax_max_k=150.0),
        tg.TriageConfig(window_px=16, slope_max_deg=25.0, shadow_min=0.2),
        tg.TriageConfig(window_px=8, index_percentile=50.0, review_support_min_fraction=0.9),
    ):
        assert_matches_reference(region(*structured()), cfg)


def test_every_threshold_is_inclusive_and_just_beyond_fails():
    """3x3 grid, one window. The candidate pixel [0,0] sits exactly ON each boundary; the median
    pixel [1,1] (index 0.4) fixes the P50 threshold and itself fails the other conditions."""
    cfg = tg.TriageConfig(window_px=3, index_percentile=50.0)
    ramp = np.arange(9, dtype=float).reshape(3, 3) / 10.0  # median (P50) = 0.4 at [1,1]

    def build(pred_v=0.4, temp_v=110.0, shadow_v=0.5, slope_v=15.0):
        pred = ramp.copy()
        pred[0, 0] = pred_v
        temp = np.full((3, 3), 250.0)
        shadow = np.zeros((3, 3))
        slope = np.full((3, 3), 40.0)
        temp[0, 0], shadow[0, 0], slope[0, 0] = temp_v, shadow_v, slope_v
        return tg.screen_arrays(region(pred, temp, shadow, slope), cfg, sensitivity=False)

    at = build()
    assert at["index_threshold_raw"] == pytest.approx(0.4)
    assert (
        at["counts"]["n_joint_pixels"] == 1
    )  # index == P50, TMAX == 110, LPSR == 0.5, slope == 15
    assert tg.window_by_id(at, 0)["joint_pixels_example"][0]["row"] == 0
    assert build(pred_v=0.3999999)["counts"]["n_joint_pixels"] == 0  # just below the percentile
    assert build(temp_v=110.0000001)["counts"]["n_joint_pixels"] == 0
    assert build(shadow_v=0.4999999)["counts"]["n_joint_pixels"] == 0
    assert build(slope_v=15.0000001)["counts"]["n_joint_pixels"] == 0
    assert (
        build(pred_v=0.5, temp_v=109.9, shadow_v=0.9, slope_v=14.9)["counts"]["n_joint_pixels"] == 1
    )


def test_ties_break_by_ascending_window_id_and_results_are_deterministic():
    pred, temp, shadow, slope = structured()
    for r0, c0 in ((0, 0), (24, 24)):  # two more windows with the SAME joint count as F (64)
        pred[r0 : r0 + 8, c0 : c0 + 8] = 1.2
        shadow[r0 : r0 + 8, c0 : c0 + 8] = 1.0
        temp[r0 : r0 + 8, c0 : c0 + 8] = 90.0
        slope[r0 : r0 + 8, c0 : c0 + 8] = 5.0
    a = tg.screen_arrays(region(pred, temp, shadow, slope), CFG8)
    ids = a["selection"]["followup_ids"]
    counts = [tg.window_by_id(a, i)["n_joint"] for i in ids]
    assert counts == [64, 64, 64] and ids == sorted(ids) == [0, 9, 15]
    b = tg.screen_arrays(region(pred, temp, shadow, slope), CFG8)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.loads(json.dumps(a)) == a  # JSON round trip is lossless (9-digit rounding)


def test_zero_qualifying_windows_is_reported_honestly_without_tuning():
    pred, temp, shadow, slope = structured()
    temp[:] = 250.0  # nothing is cold
    res = tg.screen_arrays(region(pred, temp, shadow, slope), CFG8)
    assert res["counts"]["n_followup_windows"] == 0 and res["counts"]["n_joint_pixels"] == 0
    assert res["selection"]["followup_ids"] == []
    assert any("No window contains a pixel meeting all four" in m for m in res["messages"])
    assert any("no threshold was changed" in m for m in res["messages"])
    assert res["config"]["tmax_max_k"] == 110.0  # the default was not touched
    assert len(res["sensitivity"]["variants"]) == len(tg.SENSITIVITY_VARIANTS)


def test_sensitivity_is_predeclared_separate_and_leaves_the_default_unchanged():
    res = tg.screen_arrays(region(*structured()), CFG8)
    names = [v["variant"] for v in res["sensitivity"]["variants"]]
    assert names == [n for n, _ in tg.SENSITIVITY_VARIANTS]
    without = tg.screen_arrays(region(*structured()), CFG8, sensitivity=False)
    assert "sensitivity" not in without
    for k in ("counts", "selection", "windows", "config_sha256"):
        assert res[k] == without[k]  # sensitivity never changes the default screening result
    relaxed = next(v for v in res["sensitivity"]["variants"] if v["variant"] == "relaxed_all")
    assert relaxed["n_joint_pixels"] >= res["counts"]["n_joint_pixels"]
    assert "select nothing" in res["sensitivity"]["note"]


# ======================================================================= mutation tests
def test_wrong_units_are_rejected_and_value_scale_mutations_change_the_screen():
    base = structured()
    good = tg.screen_arrays(region(*base), CFG8, sensitivity=False)
    for layer, bad in (
        (tg.TEMP, "C"),
        (tg.TERRAIN, "rad"),
        (tg.SHADOW, "percent"),
        (tg.PRED, "index"),
    ):
        with pytest.raises(tg.TriageError) as e:
            tg.screen_arrays(region(*base, units={**UNITS, layer: bad}), CFG8)
        assert e.value.code == "UNIT_MISMATCH"
    pred, temp, shadow, slope = base
    celsius = tg.screen_arrays(region(pred, temp - 273.15, shadow, slope), CFG8, sensitivity=False)
    radians = tg.screen_arrays(
        region(pred, temp, shadow, np.deg2rad(slope)), CFG8, sensitivity=False
    )
    assert celsius["counts"] != good["counts"] and radians["counts"] != good["counts"]


def test_missing_and_nonfinite_pixels_never_count_as_evidence():
    pred, temp, shadow, slope = structured()
    base = tg.screen_arrays(region(pred, temp, shadow, slope), CFG8, sensitivity=False)
    t2 = temp.copy()
    t2[16:20, 8:16] = np.nan  # half of window 9 loses its temperature
    cov = np.ones(pred.shape, bool)
    cov[20:24, 8:16] = False  # the other half is uncovered
    gone = tg.screen_arrays(region(pred, t2, shadow, slope, covered=cov), CFG8, sensitivity=False)
    assert tg.window_by_id(base, 9)["n_joint"] == 64
    assert (
        tg.window_by_id(gone, 9)["n_joint"] == 0
        and tg.window_by_id(gone, 9)["n_valid_all_layers"] == 0
    )
    assert gone["selection"]["followup_ids"] == []
    assert (
        tg.window_by_id(gone, 9)["tmax_k"]["n"] == 0 and tg.window_by_id(gone, 9)["coverage"] == 0.0
    )


def test_georeference_and_grid_mutations_are_rejected():
    base = structured()
    with pytest.raises(tg.TriageError) as e:
        tg.screen_arrays(region(*base, footprint_shift=0.4), CFG8)  # footprint moved +0.4 deg
    assert e.value.code == "GEOREFERENCE_MISMATCH"
    moved = {
        **GRID,
        "xMinM": GRID["xMinM"] + 25 * GRID["pixelSizeXM"],
        "widthPx": 32,
        "heightPx": 32,
    }
    with pytest.raises(tg.TriageError) as e2:
        tg.screen_arrays(region(*base, grid=moved), CFG8)  # origin shifted, footprint unchanged
    assert e2.value.code == "GEOREFERENCE_MISMATCH"
    a = region(*base)
    small = RasterLayer(
        tg.TEMP, np.zeros((24, 24)), np.ones((24, 24), bool), "K", "x", SHA[tg.TEMP]
    )
    with pytest.raises(tg.TriageError) as e3:
        tg.screen_arrays(replace(a, layers={**a.layers, tg.TEMP: small}), CFG8)
    assert e3.value.code == "SHAPE_MISMATCH"
    no_slope = {k: v for k, v in a.layers.items() if k != tg.TERRAIN}
    with pytest.raises(tg.TriageError) as e4:
        tg.screen_arrays(replace(a, layers=no_slope), CFG8)
    assert e4.value.code == "MISSING_LAYER"
    for bad in (tg.TriageConfig(window_px=1), tg.TriageConfig(index_percentile=100.0)):
        with pytest.raises(tg.TriageError):
            tg.screen_arrays(a, bad)


def test_window_coordinates_round_trip_through_the_original_projection():
    res = tg.screen_arrays(region(*structured()), CFG8, sensitivity=False)
    for w in res["windows"]:
        r0, r1, c0, c1 = w["window"]
        col, row = ma.lat_lon_to_pixel(PROJ, GRID, w["centre_lat"], w["centre_lon"])
        assert col == pytest.approx((c0 + c1) / 2, abs=1e-4) and row == pytest.approx(
            (r0 + r1) / 2, abs=1e-4
        )
        (nw, ne, se, sw) = w["footprint"]
        assert ma.lat_lon_to_pixel(PROJ, GRID, *nw) == pytest.approx((c0, r0), abs=1e-4)
        assert ma.lat_lon_to_pixel(PROJ, GRID, *se) == pytest.approx((c1, r1), abs=1e-4)


# ================================================================== artifact and verifier
@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("tri")
    svc = scripted_service(root / "runs", root / "src")
    res = svc.investigate("synthetic-region", run_id="inv-tri")
    return svc, res, root


def tree(d: Path):
    import hashlib

    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


def failed(rep) -> set[str]:
    return {c.id for c in rep.failures}


def test_artifact_round_trip_verifies_and_never_touches_the_bundle(bundle, tmp_path):
    _, res, _ = bundle
    before = tree(res.run_dir)
    art = ta.build_artifact(res.run_dir / "source", "inv-tri")
    out = ta.write_artifact(art, tmp_path / "out")
    assert tree(res.run_dir) == before  # bundle contract and files untouched
    assert art["format"] == "iceclaw-triage/1" and art["label"] == ta.LABEL
    loaded = ta.load_artifact(out.parent)
    rep = ta.verify_artifact(loaded, res.run_dir / "source")
    assert rep.ok, rep.render()
    assert {"recomputation", "source_package_integrity", "config_hash", "result_hash"} <= {
        c.id for c in rep.checks
    }
    assert verify_bundle(res.run_dir).ok  # the existing P6 verifier is unaffected
    rows = ta.windows_csv(art).splitlines()
    assert len(rows) == 1 + len(art["triage"]["windows"]) and "/home/" not in "\n".join(rows)


def test_tampered_artifact_or_source_fails_verification(bundle, tmp_path):
    _, res, _ = bundle
    art = ta.build_artifact(res.run_dir / "source", "inv-tri")
    ok = ta.verify_artifact(art, res.run_dir / "source")
    assert ok.ok

    forged = json.loads(json.dumps(art))
    forged["triage"]["windows"][0]["n_joint"] += 1
    assert {"recomputation", "result_hash"} <= failed(
        ta.verify_artifact(forged, res.run_dir / "source")
    )

    forged = json.loads(json.dumps(art))
    forged["triage"]["config"]["tmax_max_k"] = 130.0
    assert "config_hash" in failed(ta.verify_artifact(forged, res.run_dir / "source"))

    forged = json.loads(json.dumps(art))
    forged["source"]["files"][tg.TEMP]["sha256"] = "0" * 64
    assert "source_identity" in failed(ta.verify_artifact(forged, res.run_dir / "source"))

    forged = json.loads(json.dumps(art))
    forged["format"] = "iceclaw-triage/9"
    assert failed(ta.verify_artifact(forged, res.run_dir / "source")) == {"format"}

    src = tmp_path / "source"
    shutil.copytree(res.run_dir / "source", src)
    f = next((src / "files").glob("*TMAX.tif"))
    raw = bytearray(f.read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # altered source raster
    f.write_bytes(bytes(raw))
    assert "source_package_integrity" in failed(ta.verify_artifact(art, src))
    with pytest.raises(pkg.PackageError):
        ta.build_artifact(src, "x")


def test_artifact_labels_itself_as_post_processing_and_carries_no_secrets(bundle):
    _, res, _ = bundle
    art = ta.build_artifact(res.run_dir / "source", "inv-tri")
    text = ta.canonical(art)
    assert art["label"].startswith("POST-PROCESSING")
    assert "not an action of the agents" in art["label"]
    for tok in ("PRO", "sealed", "salt", "api_key", "sk-"):
        assert tok not in text.replace("PROCESSING", "")
    assert any("not independent confirmation of ice" in x for x in art["triage"]["limitations"])


# ======================================================================== CLI and real data
def test_cli_triage_and_verify_triage(bundle, tmp_path, monkeypatch, capsys):
    _, res, _ = bundle
    monkeypatch.setenv("ICECLAW_RUNS_ROOT", str(res.run_dir.parent))
    out = tmp_path / "t"
    assert main(["triage", "--run", "inv-tri", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "POST-PROCESSING" in text and "follow-up screening windows" in text
    assert main(["verify-triage", "--artifact", str(out), "--run", "inv-tri"]) == 0
    assert "RESULT: VERIFIED" in capsys.readouterr().out
    art = json.loads((out / "triage.json").read_text())
    art["triage"]["counts"]["n_joint_pixels"] += 1
    (out / "triage.json").write_text(json.dumps(art))
    assert main(["verify-triage", "--artifact", str(out), "--run", "inv-tri"]) == 1
    assert main(["triage", "--run", "nonexistent"]) == 2
    assert main(["verify-triage", "--artifact", str(tmp_path / "none"), "--run", "inv-tri"]) == 2


def test_cli_triage_works_in_a_clean_process_without_key_or_catalog(bundle, tmp_path):
    _, res, _ = bundle
    iso = tmp_path / "downloaded"
    shutil.copytree(res.run_dir, iso)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "ICECLAW_SKIP_DOTENV": "1",
           "ICECLAW_MOONATLAS_ROOTS": "/nonexistent", "ICECLAW_SOMBENCH_ROOT": "/nonexistent"}  # fmt: skip
    cp = subprocess.run(
        [sys.executable, "-m", "iceclaw.cli", "triage", "--run", str(iso), "--no-write"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert cp.returncode == 0, cp.stderr
    assert "follow-up screening windows" in cp.stdout


@pytest.mark.skipif(not SAMPLE_PUBLIC.is_dir(), reason="the preserved public P6 bundle is absent")
def test_real_paired_tile_screen_is_pinned_and_verifiable():
    """Golden result on the genuine tile (no LLM). Thresholds were fixed before this was run."""
    before = tree(SAMPLE_PUBLIC)
    art = ta.build_artifact(SAMPLE_PUBLIC / "source", SAMPLE_PUBLIC.name)
    t, c = art["triage"], art["triage"]["counts"]
    assert (c["n_windows"], c["n_followup_windows"], c["n_review_windows"]) == (256, 16, 8)
    assert (c["n_joint_pixels"], c["n_valid_all_layer_pixels"], c["n_high_index_pixels"]) == (
        195,
        65536,
        6554,
    )
    assert c["pixels_cumulative"] == {
        "high_index": 6554, "high_and_shadow": 4078, "high_shadow_cold": 285, "all_four": 195,
    }  # fmt: skip
    assert t["index_threshold_raw"] == pytest.approx(0.837911739)
    assert t["selection"]["followup_ids"][:3] == [16, 166, 212]  # top by joint pixels (45, 34, 27)
    top = tg.window_by_id(t, 16)
    assert top["n_joint"] == 45 and top["window"] == [16, 32, 0, 16]
    assert top["centre_lat"] == pytest.approx(-80.2323688) and top["centre_lon"] == pytest.approx(
        -37.1153635
    )
    assert {v["variant"]: v["n_followup_windows"] for v in t["sensitivity"]["variants"]}[
        "strict_all"
    ] == 9
    assert ta.verify_artifact(art, SAMPLE_PUBLIC / "source").ok
    assert tree(SAMPLE_PUBLIC) == before  # the saved public bundle is unchanged
    assert verify_bundle(SAMPLE_PUBLIC).ok
