"""Terrain slope summary. Needs a PIXELWISE slope array (layer ``pixelwise_terrain``, unit "deg").

If the layer is absent the tool returns ``UNAVAILABLE_MODALITY``; the single-anchor slope in the
published example is never used as an array.
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

TERRAIN_MODALITY = "pixelwise_terrain"


def slope_stats(
    slope: np.ndarray, cov: np.ndarray, steep_deg: float
) -> tuple[dict[str, float], int]:
    ok = cov & np.isfinite(slope)
    if not ok.any():
        raise ScienceToolError("NO_VALID_PIXELS", "no valid slope pixels")
    v = slope[ok]
    return (
        {
            "mean_deg": float(v.mean()),
            "median_deg": float(np.median(v)),
            "p95_deg": float(np.quantile(v, 0.95)),
            "max_deg": float(v.max()),
            "steep_fraction": float((v > steep_deg).mean()),
        },
        int(ok.sum()),
    )


def terrain_slope_summary(
    data: RegionArrays, window: Window | None = None, steep_deg: float = 20.0
) -> ToolMeasurement:
    """Slope statistics and the fraction of pixels steeper than ``steep_deg`` (caller parameter).

    Unit is mixed: ``*_deg`` in degrees, ``steep_fraction`` dimensionless.
    """
    tool_id = "terrain_slope_summary"
    params = base_params(TERRAIN_MODALITY, window, steep_deg=steep_deg)
    got = resolve_layer(data, tool_id, params, TERRAIN_MODALITY, window, required_unit="deg")
    if isinstance(got, ToolMeasurement):
        return got
    layer, v, c = got
    value, n = slope_stats(v, c, steep_deg)
    return measurement(
        data,
        layer,
        tool_id,
        params,
        value,
        "mixed: *_deg in degrees; steep_fraction dimensionless",
        n,
        window,
        v.shape,
    )
