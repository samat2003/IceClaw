"""Unit tests for deterministic science tools on SMALL SYNTHETIC arrays.

These prove array math and structured failures only; they say nothing about the Moon.
"""

import numpy as np
import pytest

from iceclaw.contracts import MeasurementFailure, ToolMeasurement
from iceclaw.science import model_audit as audit
from iceclaw.science import raster_metrics as rm
from iceclaw.science import terrain, thermal
from iceclaw.science.common import (
    RasterLayer,
    RegionArrays,
    ScalarMetadata,
    ScienceToolError,
)

SHA = "b" * 64
PROJ = {
    "kind": "polar-stereographic",
    "latitudeOfOriginDeg": -90,
    "centralMeridianDeg": 0,
    "scaleFactor": 0.994,
    "falseEastingM": 500000.0,
    "falseNorthingM": 500000.0,
    "radiusM": 1737400.0,
}
GRID = {"xMinM": 320000.0, "yMaxM": 740000.0, "pixelSizeXM": 10.0, "pixelSizeYM": 10.0}


def layer(name, values, covered=None, unit=rm.PREDICTION_UNIT, sha=SHA):
    values = np.asarray(values, dtype=np.float64)
    cov = np.ones(values.shape, bool) if covered is None else np.asarray(covered, bool)
    return RasterLayer(name, values, cov, unit, f"{name}.raw", sha)


def region(*layers, scalars=None, verified=True):
    return RegionArrays(
        region_id="synthetic-1",
        footprint=((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
        resolution_m=10.0,
        projection=PROJ,
        grid=GRID,
        layers={x.name: x for x in layers},
        scalars=scalars or {},
        source_verified=verified,
    )


def pred_region(values, covered=None, **kw):
    return region(layer(rm.PREDICTION_MODALITY, values, covered), **kw)


# -------------------------------------------------------------------------------- pure
def test_summary_and_quantiles_known_values():
    v = np.arange(1.0, 11.0).reshape(2, 5)  # 1..10
    s = rm.summary_stats(v, np.ones_like(v, bool))
    assert s == {
        "min": 1.0,
        "max": 10.0,
        "mean": 5.5,
        "std": pytest.approx(np.sqrt(8.25)),
        "median": 5.5,
    }
    q = rm.quantile_values(v, np.ones_like(v, bool), (0.0, 0.5, 0.9, 1.0))
    assert q == {"q0": 1.0, "q0.5": 5.5, "q0.9": pytest.approx(9.1), "q1": 10.0}
    with pytest.raises(ValueError):
        rm.quantile_values(v, np.ones_like(v, bool), (1.5,))


def test_nan_and_uncovered_are_excluded_not_filled():
    v = np.array([[1.0, np.nan, 3.0], [100.0, 5.0, 7.0]])
    c = np.array([[True, True, True], [False, True, True]])
    s = rm.summary_stats(v, c)
    assert (s["min"], s["max"], s["mean"]) == (1.0, 7.0, 4.0)
    m = rm.missingness_counts(v, c)
    assert m == {
        "total_pixels": 6.0,
        "valid_pixels": 4.0,
        "uncovered_pixels": 1.0,
        "nonfinite_pixels": 1.0,
        "nonfinite_covered_pixels": 1.0,
    }


def test_no_valid_pixels_raises():
    with pytest.raises(ScienceToolError) as e:
        rm.summary_stats(np.zeros((2, 2)), np.zeros((2, 2), bool))
    assert e.value.code == "NO_VALID_PIXELS"


def test_unit_interval_counts_keep_raw_values():
    v = np.array([[-0.1, 0.0, 1.0, 1.2]])
    c = rm.unit_interval_counts(v, np.ones_like(v, bool))
    assert c == {
        "below_zero_pixels": 1.0,
        "above_one_pixels": 1.0,
        "outside_unit_interval_pixels": 2.0,
    }


def test_gradient_of_linear_ramp_is_exact_and_borders_are_nan():
    # v = 0.1 * col, pixel 10 m  =>  |grad| = 0.01 per metre on the interior
    v = np.tile(0.1 * np.arange(6), (5, 1))
    g = rm.gradient_magnitude(v, np.ones_like(v, bool), 10.0)
    assert np.isnan(g[0]).all() and np.isnan(g[-1]).all() and np.isnan(g[:, 0]).all()
    assert np.allclose(g[1:-1, 1:-1], 0.01)
    assert rm.gradient_summary(g)["max"] == pytest.approx(0.01)
    # 2-D gradient: v = 0.3*col + 0.4*row => hypot = 0.5 / 10 m
    rr, cc = np.mgrid[0:5, 0:5]
    g2 = rm.gradient_magnitude(0.3 * cc + 0.4 * rr, np.ones((5, 5), bool), 10.0)
    assert np.allclose(g2[1:-1, 1:-1], 0.05)


def test_gradient_invalid_neighbour_gives_nan():
    v = np.tile(0.1 * np.arange(5), (5, 1))
    c = np.ones_like(v, bool)
    c[2, 2] = False
    g = rm.gradient_magnitude(v, c, 10.0)
    assert np.isnan(g[2, 2]) and np.isnan(g[2, 1]) and np.isnan(g[2, 3])
    assert np.isnan(g[1, 2]) and np.isnan(g[3, 2])
    assert np.isfinite(g[1, 1])


def test_discontinuity_finds_step_deterministically():
    v = np.zeros((9, 9))
    v[:, 5:] = 1.0  # vertical step between cols 4 and 5
    g = rm.gradient_magnitude(v, np.ones_like(v, bool), 10.0)
    thr, n, top = rm.discontinuity_candidates(g, k_mad=3.0, top_n=3)
    assert n == 14 and thr == 0.0  # 7 interior rows x 2 columns touching the step
    assert [(r, c) for r, c, _ in top] == [(1, 4), (1, 5), (2, 4)]  # raster-order ties
    assert all(val == pytest.approx(0.05) for _, _, val in top)
    assert rm.discontinuity_candidates(g, 3.0, 3) == (thr, n, top)
    with pytest.raises(ValueError):
        rm.discontinuity_candidates(g, 0.0, 3)


def test_boundary_stats_and_peak():
    v = np.zeros((6, 6))
    v[2:4, 2:4] = 2.0  # interior block
    v[0, 0] = 5.0  # corner
    b = audit.boundary_stats(v, np.ones_like(v, bool), 1)
    assert b["edge_max"] == 5.0 and b["interior_max"] == 2.0
    assert b["edge_mean"] == pytest.approx(5.0 / 20)
    assert b["edge_minus_interior"] == pytest.approx(5.0 / 20 - 8.0 / 16)
    assert audit.peak_position(v, np.ones_like(v, bool)) == (0, 0, 0)
    v2 = np.zeros((7, 7))
    v2[3, 2] = 1.0
    assert audit.peak_position(v2, np.ones_like(v2, bool)) == (3, 2, 2)
    with pytest.raises(ScienceToolError):
        audit.boundary_stats(np.ones((3, 3)), np.ones((3, 3), bool), 5)  # no interior
    with pytest.raises(ValueError):
        audit.boundary_stats(v, np.ones_like(v, bool), 0)


# ------------------------------------------------------------------------------- tools
def test_tool_measurement_structure_and_provenance():
    d = pred_region(np.arange(16.0).reshape(4, 4))
    m = rm.valid_pixel_summary(d)
    assert isinstance(m, ToolMeasurement)
    assert m.value["max"] == 15.0 and m.valid_pixel_count == 16
    assert m.unit == "prospectivity_index" and m.resolution_m == 10.0
    assert m.source_file_ids == ["nasa_ibm_prediction_raw.raw"]
    assert m.source_sha256 == {"nasa_ibm_prediction_raw.raw": SHA}
    assert m.parameters == {"modality": "nasa_ibm_prediction_raw", "window": None}
    assert m.tolerance > 0 and m.implementation_version.startswith("iceclaw-science/")
    assert m.unavailable_reason is None


def test_tool_run_id_is_deterministic_and_input_sensitive():
    a = rm.valid_pixel_summary(pred_region(np.eye(3)))
    b = rm.valid_pixel_summary(pred_region(np.eye(3)))
    assert a.tool_run_id == b.tool_run_id
    other_sha = region(layer(rm.PREDICTION_MODALITY, np.eye(3), sha="c" * 64))
    assert rm.valid_pixel_summary(other_sha).tool_run_id != a.tool_run_id
    assert rm.quantiles(pred_region(np.eye(3))).tool_run_id != a.tool_run_id


def test_all_prediction_tools_run_and_recompute_within_tolerance():
    rng = np.random.default_rng(0)  # fixed seed; synthetic only
    d = pred_region(rng.normal(0.4, 0.3, (20, 20)))
    ms = audit.run_prediction_audit(d)
    assert [m.tool_id for m in ms] == list(audit.PREDICTION_TOOLS)
    for m in ms:
        assert m.unavailable_reason is None and m.value is not None
        assert audit.matches(m, audit.recompute(m, d))
    # round-trip through JSON keeps the measurement valid and recomputable
    for m in ms:
        again = ToolMeasurement.model_validate_json(m.model_dump_json())
        assert audit.matches(again, audit.recompute(again, d))


def test_recompute_detects_changed_input_checksum():
    d = pred_region(np.eye(4))
    m = rm.valid_pixel_summary(d)
    tampered = region(layer(rm.PREDICTION_MODALITY, np.eye(4), sha="d" * 64))
    with pytest.raises(ScienceToolError) as e:
        audit.recompute(m, tampered)
    assert e.value.code == "UNVERIFIED_SOURCE"


def test_matches_rejects_different_values():
    a = rm.valid_pixel_summary(pred_region(np.eye(4)))
    b = rm.valid_pixel_summary(pred_region(np.eye(4) * 2))
    assert not audit.matches(a, b)


def test_window_selects_subregion_and_updates_footprint():
    v = np.zeros((6, 6))
    v[2:4, 2:4] = 9.0
    d = pred_region(v)
    m = rm.valid_pixel_summary(d, window=(2, 4, 2, 4))
    assert m.value["min"] == 9.0 and m.valid_pixel_count == 4
    assert m.parameters["window"] == [2, 4, 2, 4]
    assert m.footprint != list(d.footprint) and len(m.footprint) == 4
    assert audit.matches(m, audit.recompute(m, d))


@pytest.mark.parametrize("window", [(0, 7, 0, 3), (-1, 2, 0, 2), (3, 3, 0, 2), (0, 2, 4, 9)])
def test_window_outside_grid_is_out_of_coverage(window):
    m = rm.valid_pixel_summary(pred_region(np.zeros((6, 6))), window=window)
    assert m.unavailable_reason is MeasurementFailure.OUT_OF_COVERAGE and m.value is None


def test_unverified_source_blocks_all_array_math():
    m = rm.valid_pixel_summary(pred_region(np.eye(3), verified=False))
    assert m.unavailable_reason is MeasurementFailure.UNVERIFIED_SOURCE


def test_missing_modality_is_structured_not_guessed():
    d = pred_region(np.eye(3))
    m = rm.valid_pixel_summary(d, modality="pixelwise_temperature")
    assert m.unavailable_reason is MeasurementFailure.UNAVAILABLE_MODALITY
    assert m.value is None and m.valid_pixel_count is None
    assert "pixelwise_temperature" in m.parameters["unavailable_detail"]
    assert m.source_file_ids  # still cites what the region does have


def test_thermal_and_terrain_unavailable_even_with_scalar_metadata():
    scalars = {
        "psr_fraction": ScalarMetadata("psr_fraction", "tile_scalar", 0.0665, "fraction"),
        "anchor_maxTemperatureK": ScalarMetadata(
            "anchor_maxTemperatureK", "single_anchor_point", 84.9, "K"
        ),
        "anchor_slopeDeg": ScalarMetadata("anchor_slopeDeg", "single_anchor_point", 6.56, "deg"),
    }
    d = pred_region(np.eye(4), scalars=scalars)
    t = thermal.thermal_psr_consistency(d)
    s = terrain.terrain_slope_summary(d)
    for m in (t, s):
        assert m.unavailable_reason is MeasurementFailure.UNAVAILABLE_MODALITY
        assert m.value is None
        assert "not substituted" in m.parameters["unavailable_detail"]
    assert audit.run_tool(d, "thermal_psr_consistency").unavailable_reason is not None
    # unavailable results are reproducible too
    assert audit.matches(t, audit.recompute(t, d))


def test_thermal_computation_with_synthetic_pixelwise_inputs():
    temp = np.array([[90.0, 100.0, 200.0, 250.0]] * 2)
    shadow = np.array([[1.0, 1.0, 0.0, 0.0]] * 2)
    d = region(
        layer(rm.PREDICTION_MODALITY, np.zeros((2, 4))),
        layer(thermal.TEMPERATURE_MODALITY, temp, unit="K"),
        layer(thermal.SHADOW_MODALITY, shadow, unit="fraction"),
    )
    m = thermal.thermal_psr_consistency(d, cold_trap_k=95.0)
    assert m.value["psr_mean_k"] == 95.0 and m.value["non_psr_mean_k"] == 225.0
    assert m.value["psr_minus_non_psr_k"] == -130.0
    assert m.value["psr_below_cold_trap_fraction"] == 0.5
    assert (m.value["psr_count"], m.value["non_psr_count"]) == (4.0, 4.0)
    assert sorted(m.source_file_ids) == ["pixelwise_shadow_mask.raw", "pixelwise_temperature.raw"]
    assert audit.matches(m, audit.recompute(m, d))


def test_thermal_wrong_unit_is_unavailable():
    d = region(
        layer(rm.PREDICTION_MODALITY, np.zeros((2, 2))),
        layer(thermal.TEMPERATURE_MODALITY, np.ones((2, 2)), unit="degC"),
        layer(thermal.SHADOW_MODALITY, np.ones((2, 2)), unit="fraction"),
    )
    m = thermal.thermal_psr_consistency(d)
    assert m.unavailable_reason is MeasurementFailure.UNAVAILABLE_MODALITY
    assert "degC" in m.parameters["unavailable_detail"]


def test_terrain_computation_with_synthetic_slope():
    slope = np.array([[0.0, 10.0], [25.0, 30.0]])
    d = region(
        layer(rm.PREDICTION_MODALITY, np.zeros((2, 2))),
        layer(terrain.TERRAIN_MODALITY, slope, unit="deg"),
    )
    m = terrain.terrain_slope_summary(d, steep_deg=20.0)
    assert m.value["max_deg"] == 30.0 and m.value["mean_deg"] == 16.25
    assert m.value["steep_fraction"] == 0.5
    assert audit.matches(m, audit.recompute(m, d))


def test_unknown_tool_rejected():
    with pytest.raises(KeyError):
        audit.run_tool(pred_region(np.eye(3)), "not_a_tool")
