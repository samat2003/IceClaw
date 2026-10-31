"""Prediction-audit tools (boundary, peak) plus the shared tool registry and recomputation."""

from __future__ import annotations

import inspect

import numpy as np

from iceclaw.contracts import MeasurementFailure, ToolMeasurement
from iceclaw.science import raster_metrics as rm
from iceclaw.science import terrain, thermal
from iceclaw.science.common import (
    RegionArrays,
    ScienceToolError,
    Window,
    base_params,
    measurement,
    resolve_layer,
)


def border_distance(shape: tuple[int, int]) -> np.ndarray:
    """Chebyshev distance (pixels) from each pixel to the nearest grid border (0 on the border)."""
    h, w = shape
    r = np.arange(h)[:, None]
    c = np.arange(w)[None, :]
    return np.minimum(np.minimum(r, h - 1 - r), np.minimum(c, w - 1 - c))


def boundary_stats(values: np.ndarray, covered: np.ndarray, edge_width_px: int) -> dict[str, float]:
    """Compare valid values in the outer ``edge_width_px`` band with the interior."""
    if edge_width_px < 1:
        raise ValueError("edge_width_px must be >= 1")
    ok = rm.valid_mask(values, covered)
    edge = border_distance(values.shape) < edge_width_px
    ev, iv = values[ok & edge], values[ok & ~edge]
    if ev.size == 0 or iv.size == 0:
        raise ScienceToolError(
            "NO_VALID_PIXELS", "edge band or interior has no valid pixels for this width"
        )
    return {
        "edge_mean": float(ev.mean()),
        "interior_mean": float(iv.mean()),
        "edge_minus_interior": float(ev.mean() - iv.mean()),
        "edge_max": float(ev.max()),
        "interior_max": float(iv.max()),
    }


def peak_position(values: np.ndarray, covered: np.ndarray) -> tuple[int, int, int]:
    """(row, col, edge_distance_px) of the first-in-raster-order maximum of valid pixels."""
    ok = rm.valid_mask(values, covered)
    if not ok.any():
        raise ScienceToolError("NO_VALID_PIXELS", "no valid pixels")
    masked = np.where(ok, values, -np.inf)
    r, c = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(r), int(c), int(border_distance(values.shape)[r, c])


def boundary_check(
    data: RegionArrays,
    modality: str = rm.PREDICTION_MODALITY,
    window: Window | None = None,
    edge_width_px: int = 8,
) -> ToolMeasurement:
    """Edge-band vs interior prediction means (layer unit). Large offsets can flag tile-edge
    artifacts; this is a screening statistic, not proof of one."""
    params = base_params(modality, window, edge_width_px=edge_width_px)
    got = resolve_layer(data, "boundary_check", params, modality, window)
    if isinstance(got, ToolMeasurement):
        return got
    layer, v, c = got
    return measurement(
        data,
        layer,
        "boundary_check",
        params,
        boundary_stats(v, c, edge_width_px),
        layer.unit,
        int(rm.valid_mask(v, c).sum()),
        window,
        v.shape,
    )


def peak_location(
    data: RegionArrays, modality: str = rm.PREDICTION_MODALITY, window: Window | None = None
) -> ToolMeasurement:
    """Row/column of the raw maximum and its distance to the selection border (pixels)."""
    params = base_params(modality, window)
    got = resolve_layer(data, "peak_location", params, modality, window)
    if isinstance(got, ToolMeasurement):
        return got
    layer, v, c = got
    r, col, d = peak_position(v, c)
    return measurement(
        data,
        layer,
        "peak_location",
        params,
        {"row": float(r), "col": float(col), "edge_distance_px": float(d)},
        "pixels",
        int(rm.valid_mask(v, c).sum()),
        window,
        v.shape,
    )


# ------------------------------------------------------------------------------ registry
TOOLS = {
    "valid_pixel_summary": rm.valid_pixel_summary,
    "quantiles": rm.quantiles,
    "missingness": rm.missingness,
    "unit_interval_check": rm.unit_interval_check,
    "gradient_stats": rm.gradient_stats,
    "discontinuity_candidates": rm.discontinuity_candidates_tool,
    "boundary_check": boundary_check,
    "peak_location": peak_location,
    "thermal_psr_consistency": thermal.thermal_psr_consistency,
    "terrain_slope_summary": terrain.terrain_slope_summary,
}

PREDICTION_TOOLS = (
    "valid_pixel_summary",
    "quantiles",
    "missingness",
    "unit_interval_check",
    "gradient_stats",
    "discontinuity_candidates",
    "boundary_check",
    "peak_location",
)


def run_tool(data: RegionArrays, tool_id: str, **params) -> ToolMeasurement:
    if tool_id not in TOOLS:
        raise KeyError(f"unknown tool {tool_id!r}")
    return TOOLS[tool_id](data, **params)


def run_prediction_audit(data: RegionArrays) -> list[ToolMeasurement]:
    """All default-parameter prediction tools, in a fixed order."""
    return [run_tool(data, t) for t in PREDICTION_TOOLS]


def _tool_kwargs(m: ToolMeasurement) -> dict:
    # Pass only what the tool accepts (e.g. thermal/terrain record their fixed layer names as
    # parameters but do not take them as arguments); ``matches`` still compares full parameters.
    accepted = set(inspect.signature(TOOLS[m.tool_id]).parameters)
    p = {k: v for k, v in m.parameters.items() if k in accepted}
    if p.get("window") is not None:
        p["window"] = tuple(p["window"])
    if "qs" in p:
        p["qs"] = tuple(p["qs"])
    return p


def recompute(m: ToolMeasurement, data: RegionArrays) -> ToolMeasurement:
    """Re-run ``m.tool_id`` with ``m.parameters`` on ``data``. Raises if the source differs."""
    for fid, sha in m.source_sha256.items():
        layer = next((x for x in data.layers.values() if x.file_id == fid), None)
        if layer is None or layer.sha256 != sha:
            raise ScienceToolError(
                MeasurementFailure.UNVERIFIED_SOURCE.value,
                f"input {fid!r} does not match the checksum recorded in the measurement",
            )
    return run_tool(data, m.tool_id, **_tool_kwargs(m))


def matches(a: ToolMeasurement, b: ToolMeasurement) -> bool:
    """True if ``b`` reproduces ``a`` within ``a.tolerance`` (same ids, availability and values)."""
    if (a.tool_id, a.parameters, a.source_sha256, a.unavailable_reason, a.valid_pixel_count) != (
        b.tool_id,
        b.parameters,
        b.source_sha256,
        b.unavailable_reason,
        b.valid_pixel_count,
    ):
        return False
    if a.value is None or b.value is None:
        return a.value is None and b.value is None
    if isinstance(a.value, dict):
        if not isinstance(b.value, dict) or a.value.keys() != b.value.keys():
            return False
        pairs = [(a.value[k], b.value[k]) for k in a.value]
    elif isinstance(a.value, list):
        pairs = list(zip(a.value, b.value, strict=True))
    else:
        pairs = [(a.value, b.value)]
    return all(abs(x - y) <= a.tolerance * max(1.0, abs(x)) for x, y in pairs)
