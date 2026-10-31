"""Deterministic raster statistics. Pure array functions first, then ``ToolMeasurement`` tools.

"Valid" pixels are those that are covered AND finite. Raw values are never clipped to [0, 1].
"""

from __future__ import annotations

import numpy as np

from iceclaw.contracts import ToolMeasurement
from iceclaw.science.common import (
    RegionArrays,
    ScienceToolError,
    Window,
    base_params,
    measurement,
    resolve_layer,
)

PREDICTION_MODALITY = "nasa_ibm_prediction_raw"
PREDICTION_UNIT = "prospectivity_index"
DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


# ------------------------------------------------------------------------ pure functions
def valid_mask(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    return covered & np.isfinite(values)


def valid_values(values: np.ndarray, covered: np.ndarray) -> np.ndarray:
    v = values[valid_mask(values, covered)]
    if v.size == 0:
        raise ScienceToolError("NO_VALID_PIXELS", "no covered, finite pixels in the selection")
    return v


def summary_stats(values: np.ndarray, covered: np.ndarray) -> dict[str, float]:
    v = valid_values(values, covered)
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),  # population std (ddof=0)
        "median": float(np.median(v)),
    }


def quantile_values(
    values: np.ndarray, covered: np.ndarray, qs: tuple[float, ...]
) -> dict[str, float]:
    if not qs or any(not (0.0 <= q <= 1.0) for q in qs):
        raise ValueError("quantiles must be a non-empty sequence within [0, 1]")
    v = valid_values(values, covered)
    # method="linear" is numpy's default and matches upstream's p90 definition.
    return {f"q{q:g}": float(np.quantile(v, q, method="linear")) for q in qs}


def missingness_counts(values: np.ndarray, covered: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(values)
    return {
        "total_pixels": float(values.size),
        "valid_pixels": float((covered & finite).sum()),
        "uncovered_pixels": float((~covered).sum()),
        "nonfinite_pixels": float((~finite).sum()),
        "nonfinite_covered_pixels": float((covered & ~finite).sum()),
    }


def unit_interval_counts(values: np.ndarray, covered: np.ndarray) -> dict[str, float]:
    v = valid_values(values, covered)
    below, above = int((v < 0).sum()), int((v > 1).sum())
    return {
        "below_zero_pixels": float(below),
        "above_one_pixels": float(above),
        "outside_unit_interval_pixels": float(below + above),
    }


def gradient_magnitude(values: np.ndarray, covered: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """|grad| per metre from central differences; NaN wherever any of the 5 stencil pixels is
    invalid or on the selection border."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    ok = valid_mask(values, covered)
    v = np.where(ok, values, np.nan)
    g = np.full(v.shape, np.nan)
    if v.shape[0] < 3 or v.shape[1] < 3:
        return g
    gx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * pixel_size_m)
    gy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * pixel_size_m)
    g[1:-1, 1:-1] = np.hypot(gx, gy)  # NaN propagates from any invalid neighbour
    g[~ok] = np.nan
    return g


def gradient_summary(grad: np.ndarray) -> dict[str, float]:
    g = grad[np.isfinite(grad)]
    if g.size == 0:
        raise ScienceToolError("NO_VALID_PIXELS", "no pixel has a complete gradient stencil")
    return {
        "mean": float(g.mean()),
        "median": float(np.median(g)),
        "p95": float(np.quantile(g, 0.95)),
        "max": float(g.max()),
    }


def discontinuity_candidates(
    grad: np.ndarray, k_mad: float, top_n: int
) -> tuple[float, int, list[tuple[int, int, float]]]:
    """Robust outliers of the gradient field: ``g > median + k_mad * 1.4826 * MAD``.

    Returns (threshold, n_candidates, top_n list of (row, col, gradient)) ordered by descending
    gradient, ties broken by raster order -- fully deterministic.
    """
    if k_mad <= 0 or top_n < 0:
        raise ValueError("k_mad must be > 0 and top_n >= 0")
    finite = np.isfinite(grad)
    g = grad[finite]
    if g.size == 0:
        raise ScienceToolError("NO_VALID_PIXELS", "no pixel has a complete gradient stencil")
    med = float(np.median(g))
    mad = float(np.median(np.abs(g - med)))
    thr = med + k_mad * 1.4826 * mad
    rows, cols = np.nonzero(finite & (grad > thr))
    vals = grad[rows, cols]
    order = np.lexsort((cols, rows, -vals))
    top = [(int(rows[i]), int(cols[i]), float(vals[i])) for i in order[:top_n]]
    return thr, int(rows.size), top


# ----------------------------------------------------------------------------- tools
def _run(data, tool_id, modality, window, extra, compute, unit, required_unit=None):
    params = base_params(modality, window, **extra)
    got = resolve_layer(data, tool_id, params, modality, window, required_unit)
    if isinstance(got, ToolMeasurement):
        return got
    layer, values, covered = got
    value, n_valid = compute(values, covered)
    return measurement(data, layer, tool_id, params, value, unit, n_valid, window, values.shape)


def valid_pixel_summary(
    data: RegionArrays, modality: str = PREDICTION_MODALITY, window: Window | None = None
) -> ToolMeasurement:
    """min/max/mean/std/median over valid pixels (layer unit)."""
    unit = data.layers[modality].unit if modality in data.layers else PREDICTION_UNIT

    def compute(v, c):
        return summary_stats(v, c), int(valid_mask(v, c).sum())

    return _run(data, "valid_pixel_summary", modality, window, {}, compute, unit)


def quantiles(
    data: RegionArrays,
    modality: str = PREDICTION_MODALITY,
    window: Window | None = None,
    qs: tuple[float, ...] = DEFAULT_QUANTILES,
) -> ToolMeasurement:
    unit = data.layers[modality].unit if modality in data.layers else PREDICTION_UNIT

    def compute(v, c):
        return quantile_values(v, c, qs), int(valid_mask(v, c).sum())

    return _run(data, "quantiles", modality, window, {"qs": list(qs)}, compute, unit)


def missingness(
    data: RegionArrays, modality: str = PREDICTION_MODALITY, window: Window | None = None
) -> ToolMeasurement:
    def compute(v, c):
        return missingness_counts(v, c), int(valid_mask(v, c).sum())

    return _run(data, "missingness", modality, window, {}, compute, "pixels")


def unit_interval_check(
    data: RegionArrays, modality: str = PREDICTION_MODALITY, window: Window | None = None
) -> ToolMeasurement:
    """Counts of raw values outside [0, 1] (informational: regression output is unconstrained)."""

    def compute(v, c):
        return unit_interval_counts(v, c), int(valid_mask(v, c).sum())

    return _run(data, "unit_interval_check", modality, window, {}, compute, "pixels")


def gradient_stats(
    data: RegionArrays, modality: str = PREDICTION_MODALITY, window: Window | None = None
) -> ToolMeasurement:
    """Spatial-gradient magnitude statistics at native resolution (layer unit per metre)."""
    unit = f"{data.layers[modality].unit}/m" if modality in data.layers else f"{PREDICTION_UNIT}/m"

    def compute(v, c):
        g = gradient_magnitude(v, c, data.resolution_m)
        return gradient_summary(g), int(np.isfinite(g).sum())

    return _run(data, "gradient_stats", modality, window, {}, compute, unit)


def discontinuity_candidates_tool(
    data: RegionArrays,
    modality: str = PREDICTION_MODALITY,
    window: Window | None = None,
    k_mad: float = 6.0,
    top_n: int = 5,
) -> ToolMeasurement:
    """Robust-outlier gradient pixels (candidate discontinuities; not a physical claim).

    Value keys: ``threshold_gradient``, ``n_candidates_count``, and for rank i:
    ``cand{i}_row``/``cand{i}_col`` (window-relative pixels) and ``cand{i}_gradient``.
    """
    unit = (
        "mixed: *_count/_row/_col in pixels; *_gradient in "
        f"{data.layers[modality].unit if modality in data.layers else PREDICTION_UNIT}/m"
    )

    def compute(v, c):
        g = gradient_magnitude(v, c, data.resolution_m)
        thr, n, top = discontinuity_candidates(g, k_mad, top_n)
        out = {"threshold_gradient": thr, "n_candidates_count": float(n)}
        for i, (r, col, val) in enumerate(top):
            out.update(
                {f"cand{i}_row": float(r), f"cand{i}_col": float(col), f"cand{i}_gradient": val}
            )
        return out, int(np.isfinite(g).sum())

    return _run(
        data,
        "discontinuity_candidates",
        modality,
        window,
        {"k_mad": k_mad, "top_n": top_n},
        compute,
        unit,
    )
