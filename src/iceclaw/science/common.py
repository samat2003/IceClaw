"""Shared types and ToolMeasurement construction for deterministic science tools.

No LLM, Streamlit, evaluator or reference data is reachable from here. Three kinds of source
information are kept strictly apart:

* ``RegionArrays.layers``  -- pixelwise scientific rasters (the only input to array math);
* ``RegionArrays.scalars`` -- tile-level scalars and single-anchor metadata (context only,
  never substituted for arrays by any tool).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iceclaw.contracts import MeasurementFailure, ToolMeasurement
from iceclaw.data import moonatlas_adapter as ma

IMPLEMENTATION_VERSION = "iceclaw-science/0.1.0"
# Re-running a tool on identical decoded input is bit-for-bit deterministic in float64; this is the
# declared absolute/relative tolerance for independent recomputation.
REPRODUCIBILITY_TOLERANCE = 1e-9

Window = tuple[int, int, int, int]  # (row0, row1, col0, col1), half-open, in native pixels


class ScienceToolError(RuntimeError):
    """Deterministic tool could not produce a number (e.g. NO_VALID_PIXELS)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class RasterLayer:
    """One pixelwise raster on the region's native grid."""

    name: str
    values: np.ndarray  # (H, W) float64; raw values, never clipped
    covered: np.ndarray  # (H, W) bool
    unit: str
    file_id: str
    sha256: str

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape != self.covered.shape:
            raise ValueError("values/covered must be 2-D and the same shape")


@dataclass(frozen=True)
class ScalarMetadata:
    """Tile-level scalar or single-anchor metadata. NOT an array and never used as one."""

    name: str
    kind: str  # "tile_scalar" | "single_anchor_point"
    value: Any
    unit: str | None


@dataclass(frozen=True)
class RegionArrays:
    region_id: str
    footprint: tuple[tuple[float, float], ...]  # (lat, lon) corners NW, NE, SE, SW
    resolution_m: float
    projection: dict[str, Any]
    grid: dict[str, Any]
    layers: dict[str, RasterLayer]
    scalars: dict[str, ScalarMetadata] = field(default_factory=dict)
    source_verified: bool = True


# -------------------------------------------------------------------------- measurements
def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def run_id(tool_id: str, params: dict[str, Any], sha: dict[str, str], region_id: str) -> str:
    """Deterministic id: identical tool, parameters, inputs and implementation => identical id."""
    digest = hashlib.sha256(
        _canonical([tool_id, params, sha, region_id, IMPLEMENTATION_VERSION]).encode()
    ).hexdigest()
    return f"{tool_id}-{digest[:20]}"


def _all_sources(data: RegionArrays) -> tuple[list[str], dict[str, str]]:
    if not data.layers:
        raise ValueError("region has no pixelwise layers to cite as source")
    ids = sorted(lyr.file_id for lyr in data.layers.values())
    return ids, {lyr.file_id: lyr.sha256 for lyr in data.layers.values()}


def window_footprint(data: RegionArrays, window: Window | None, shape: tuple[int, int]):
    if window is None:
        return list(data.footprint)
    r0, r1, c0, c1 = window
    corners = ((c0, r0), (c1, r0), (c1, r1), (c0, r1))  # outer pixel corners NW, NE, SE, SW
    return [ma.pixel_to_lat_lon(data.projection, data.grid, c, r) for c, r in corners]


def unavailable(
    data: RegionArrays,
    tool_id: str,
    params: dict[str, Any],
    reason: MeasurementFailure,
    detail: str,
) -> ToolMeasurement:
    ids, sha = _all_sources(data)
    p = {**params, "unavailable_detail": detail}
    return ToolMeasurement(
        tool_run_id=run_id(tool_id, p, sha, data.region_id),
        tool_id=tool_id,
        parameters=p,
        region_id=data.region_id,
        footprint=list(data.footprint),
        source_file_ids=ids,
        source_sha256=sha,
        value=None,
        unavailable_reason=reason,
        unit=None,
        valid_pixel_count=None,
        resolution_m=data.resolution_m,
        tolerance=REPRODUCIBILITY_TOLERANCE,
        implementation_version=IMPLEMENTATION_VERSION,
    )


def measurement(
    data: RegionArrays,
    layer: RasterLayer,
    tool_id: str,
    params: dict[str, Any],
    value: float | dict[str, float],
    unit: str,
    valid_pixel_count: int,
    window: Window | None,
    shape: tuple[int, int],
    extra_layers: tuple[RasterLayer, ...] = (),
) -> ToolMeasurement:
    used = [layer, *extra_layers]
    sha = {lyr.file_id: lyr.sha256 for lyr in used}
    return ToolMeasurement(
        tool_run_id=run_id(tool_id, params, sha, data.region_id),
        tool_id=tool_id,
        parameters=params,
        region_id=data.region_id,
        footprint=window_footprint(data, window, shape),
        source_file_ids=sorted(sha),
        source_sha256=sha,
        value=value,
        unavailable_reason=None,
        unit=unit,
        valid_pixel_count=int(valid_pixel_count),
        resolution_m=data.resolution_m,
        tolerance=REPRODUCIBILITY_TOLERANCE,
        implementation_version=IMPLEMENTATION_VERSION,
    )


def resolve_layer(
    data: RegionArrays,
    tool_id: str,
    params: dict[str, Any],
    modality: str,
    window: Window | None,
    required_unit: str | None = None,
) -> tuple[RasterLayer, np.ndarray, np.ndarray] | ToolMeasurement:
    """Return (layer, values, covered) cropped to ``window``, or a structured failure measurement.

    Failures: UNVERIFIED_SOURCE (source not validated), UNAVAILABLE_MODALITY (no such pixelwise
    layer, or wrong unit), OUT_OF_COVERAGE (window outside the grid).
    """
    if not data.source_verified:
        return unavailable(
            data, tool_id, params, MeasurementFailure.UNVERIFIED_SOURCE, "source failed validation"
        )
    layer = data.layers.get(modality)
    if layer is None:
        return unavailable(
            data,
            tool_id,
            params,
            MeasurementFailure.UNAVAILABLE_MODALITY,
            f"pixelwise layer {modality!r} is not available for region {data.region_id}; "
            "tile scalars and single-anchor metadata are not substituted for arrays",
        )
    if required_unit is not None and layer.unit != required_unit:
        return unavailable(
            data,
            tool_id,
            params,
            MeasurementFailure.UNAVAILABLE_MODALITY,
            f"layer {modality!r} has unit {layer.unit!r}, tool requires {required_unit!r}",
        )
    h, w = layer.values.shape
    if window is None:
        return layer, layer.values, layer.covered
    r0, r1, c0, c1 = window
    if not (0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w):
        return unavailable(
            data,
            tool_id,
            params,
            MeasurementFailure.OUT_OF_COVERAGE,
            f"window {list(window)} is outside the {h}x{w} grid",
        )
    return layer, layer.values[r0:r1, c0:c1], layer.covered[r0:r1, c0:c1]


def base_params(modality: str, window: Window | None, **extra: Any) -> dict[str, Any]:
    return {"modality": modality, "window": list(window) if window else None, **extra}
