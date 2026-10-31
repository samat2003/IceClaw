"""Thermal / permanently-shadowed-region (PSR) consistency. Requires PIXELWISE arrays.

Inputs: a temperature layer (unit "K") and a shadow layer (unit "fraction", 0..1). If either is
absent the tool returns ``UNAVAILABLE_MODALITY``. The tile-level ``psr_fraction`` and the
single-anchor temperature are never substituted for arrays.
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

TEMPERATURE_MODALITY = "pixelwise_temperature"
SHADOW_MODALITY = "pixelwise_shadow_mask"


def psr_temperature_stats(
    temp: np.ndarray,
    tcov: np.ndarray,
    shadow: np.ndarray,
    scov: np.ndarray,
    psr_threshold: float,
    cold_trap_k: float,
) -> tuple[dict[str, float], int]:
    ok = tcov & scov & np.isfinite(temp) & np.isfinite(shadow)
    psr = ok & (shadow >= psr_threshold)
    sunlit = ok & ~psr
    if not psr.any() or not sunlit.any():
        raise ScienceToolError(
            "NO_VALID_PIXELS", "need both PSR and non-PSR valid pixels to compare temperatures"
        )
    t_psr, t_sun = temp[psr], temp[sunlit]
    return (
        {
            "psr_mean_k": float(t_psr.mean()),
            "non_psr_mean_k": float(t_sun.mean()),
            "psr_minus_non_psr_k": float(t_psr.mean() - t_sun.mean()),
            "psr_below_cold_trap_fraction": float((t_psr < cold_trap_k).mean()),
            "psr_count": float(psr.sum()),
            "non_psr_count": float(sunlit.sum()),
        },
        int(ok.sum()),
    )


def thermal_psr_consistency(
    data: RegionArrays,
    window: Window | None = None,
    psr_threshold: float = 0.5,
    cold_trap_k: float = 110.0,
) -> ToolMeasurement:
    """Mean temperature inside vs outside PSR pixels and the fraction of PSR pixels colder than
    ``cold_trap_k`` (a caller-chosen parameter, not a validated physical threshold).

    Unit is mixed: ``*_k`` in K, ``*_count`` in pixels, ``*_fraction`` dimensionless.
    """
    tool_id = "thermal_psr_consistency"
    params = base_params(
        TEMPERATURE_MODALITY,
        window,
        shadow_modality=SHADOW_MODALITY,
        psr_threshold=psr_threshold,
        cold_trap_k=cold_trap_k,
    )
    t = resolve_layer(data, tool_id, params, TEMPERATURE_MODALITY, window, required_unit="K")
    if isinstance(t, ToolMeasurement):
        return t
    s = resolve_layer(data, tool_id, params, SHADOW_MODALITY, window, required_unit="fraction")
    if isinstance(s, ToolMeasurement):
        return s
    (tl, tv, tc), (sl, sv, sc) = t, s
    if tv.shape != sv.shape:
        raise ScienceToolError("GRID_MISMATCH", "temperature and shadow layers differ in shape")
    value, n = psr_temperature_stats(tv, tc, sv, sc, psr_threshold, cold_trap_k)
    return measurement(
        data,
        tl,
        tool_id,
        params,
        value,
        "mixed: *_k in K; *_count in pixels; *_fraction dimensionless",
        n,
        window,
        tv.shape,
        extra_layers=(sl,),
    )
