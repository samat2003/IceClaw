"""Seeded, deterministic fault injectors. They only ever operate on COPIES.

Every fault is a *synthetic* corruption of a genuine prediction raster or its metadata. A seeded
fault is never a real NASA/IBM anomaly. ``expected_detectable`` is the pre-registered claim about
whether the reference-free P3 baseline (see ``baselines.py``) should catch it; it is written down
BEFORE the baseline is run and is compared with the outcome in the evaluation report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaseData:
    """Everything a prediction-only scenario exposes: raster plus the metadata that describes it."""

    values: np.ndarray  # (H, W) float64 raw model output (never clipped)
    covered: np.ndarray  # (H, W) bool
    footprint: tuple[tuple[float, float], ...]  # (lat, lon) NW, NE, SE, SW
    grid: dict[str, Any]
    projection: dict[str, Any]
    resolution_m: float

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape != self.covered.shape:
            raise ValueError("values/covered must be 2-D and the same shape")

    def copy(self) -> CaseData:
        return CaseData(
            self.values.copy(),
            self.covered.copy(),
            tuple(tuple(c) for c in self.footprint),
            dict(self.grid),
            dict(self.projection),
            float(self.resolution_m),
        )


@dataclass(frozen=True)
class FaultSpec:
    fault_id: str
    category: str  # "clean" | "value" | "geometry" | "metadata"
    is_fault: bool
    intent: str
    severity: str
    expected_detectable: bool | None  # None for clean controls
    detectable_rationale: str


Builder = Callable[[CaseData, np.random.Generator], tuple[CaseData, dict[str, Any]]]


def _block(shape: tuple[int, int], size: int, rng: np.random.Generator) -> tuple[int, int, int]:
    size = max(2, min(size, shape[0] // 2, shape[1] // 2))
    r0 = int(rng.integers(0, shape[0] - size + 1))
    c0 = int(rng.integers(0, shape[1] - size + 1))
    return r0, c0, size


# --------------------------------------------------------------------------- clean controls
def clean_identical(src: CaseData, rng: np.random.Generator):
    return src.copy(), {}


def clean_float32_roundtrip(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    return replace(out, values=out.values.astype(np.float32).astype(np.float64)), {"dtype": "f4"}


def clean_quantization_noise(src: CaseData, rng: np.random.Generator):
    sigma = 1e-6  # below the raw encoding's own quantization step (~1.6e-5)
    out = src.copy()
    return replace(out, values=out.values + rng.normal(0.0, sigma, out.values.shape)), {
        "sigma": sigma
    }


# ------------------------------------------------------------------------------ value faults
def value_unit_scale_x100(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    return replace(out, values=out.values * 100.0), {"factor": 100.0}


def value_unit_scale_div100(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    return replace(out, values=out.values / 100.0), {"factor": 0.01}


def value_nan_block(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    r0, c0, n = _block(out.values.shape, 24, rng)
    out.values[r0 : r0 + n, c0 : c0 + n] = np.nan  # covered mask deliberately left True
    return out, {"row": r0, "col": c0, "size": n}


def value_saturation_block(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    r0, c0, n = _block(out.values.shape, 32, rng)
    out.values[r0 : r0 + n, c0 : c0 + n] = 5.0
    return out, {"row": r0, "col": c0, "size": n, "value": 5.0}


def value_offset_small(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    return replace(out, values=out.values + 0.05), {"offset": 0.05}


def value_spatial_roll(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    w = out.values.shape[1]
    shift = int(rng.integers(w // 8, w // 3 + 1))
    return replace(out, values=np.roll(out.values, shift, axis=1)), {"axis": 1, "shift_px": shift}


# --------------------------------------------------------------- metadata / geometry faults
def metadata_footprint_shift(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    delta = 0.4
    fp = tuple((lat + delta, lon) for lat, lon in out.footprint)
    return replace(out, footprint=fp), {"delta_lat_deg": delta}


def metadata_grid_origin_shift(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    px = 25
    grid = dict(out.grid)
    grid["xMinM"] = grid["xMinM"] + px * grid["pixelSizeXM"]
    return replace(out, grid=grid), {"shift_px": px}


def metadata_resolution_unit_error(src: CaseData, rng: np.random.Generator):
    out = src.copy()
    return replace(out, resolution_m=out.resolution_m * 0.1), {"factor": 0.1}


FAULTS: dict[str, tuple[FaultSpec, Builder]] = {
    s.fault_id: (s, b)
    for s, b in [
        (
            FaultSpec("clean_identical", "clean", False, "unmodified copy", "none", None, ""),
            clean_identical,
        ),
        (
            FaultSpec(
                "clean_float32_roundtrip",
                "clean",
                False,
                "benign float32 round-trip (<1e-7 change)",
                "none",
                None,
                "",
            ),
            clean_float32_roundtrip,
        ),
        (
            FaultSpec(
                "clean_quantization_noise",
                "clean",
                False,
                "benign noise below the raw encoding quantization step",
                "none",
                None,
                "",
            ),
            clean_quantization_noise,
        ),
        (
            FaultSpec(
                "value_unit_scale_x100",
                "value",
                True,
                "percent-vs-fraction conversion error (values x100)",
                "high",
                True,
                "max far above any plausible regression range (rule value_range)",
            ),
            value_unit_scale_x100,
        ),
        (
            FaultSpec(
                "value_unit_scale_div100",
                "value",
                True,
                "inverse conversion error (values /100): dynamic range collapses",
                "high",
                True,
                "max-min collapses below the dynamic-range floor (rule dynamic_range)",
            ),
            value_unit_scale_div100,
        ),
        (
            FaultSpec(
                "value_nan_block",
                "value",
                True,
                "block of covered pixels turned into NaN",
                "medium",
                True,
                "non-finite covered pixels (rule missing_data)",
            ),
            value_nan_block,
        ),
        (
            FaultSpec(
                "value_saturation_block",
                "value",
                True,
                "block overwritten with the constant 5.0",
                "medium",
                True,
                "max far above plausible range (rule value_range)",
            ),
            value_saturation_block,
        ),
        (
            FaultSpec(
                "value_offset_small",
                "value",
                True,
                "constant +0.05 bias on every pixel",
                "low",
                False,
                "stays inside the plausible range and shifts few pixels outside [0,1]; "
                "a reference-free range/missingness baseline is not expected to see it",
            ),
            value_offset_small,
        ),
        (
            FaultSpec(
                "value_spatial_roll",
                "value",
                True,
                "prediction rolled along columns (spatial misalignment with wrap seam)",
                "medium",
                False,
                "seam jump is comparable to natural gradients, so a robust-outlier "
                "line rule is not expected to isolate it",
            ),
            value_spatial_roll,
        ),
        (
            FaultSpec(
                "metadata_footprint_shift",
                "metadata",
                True,
                "declared footprint shifted +0.4 deg latitude",
                "high",
                True,
                "footprint no longer matches grid+projection (rule geometry_consistency)",
            ),
            metadata_footprint_shift,
        ),
        (
            FaultSpec(
                "metadata_grid_origin_shift",
                "geometry",
                True,
                "grid origin moved 25 px east without updating the footprint",
                "high",
                True,
                "grid+projection no longer reproduce the footprint (rule geometry_consistency)",
            ),
            metadata_grid_origin_shift,
        ),
        (
            FaultSpec(
                "metadata_resolution_unit_error",
                "metadata",
                True,
                "declared resolution is 10x smaller than the grid pixel size",
                "high",
                True,
                "resolution disagrees with grid pixel size (rule geometry_consistency)",
            ),
            metadata_resolution_unit_error,
        ),
    ]
}

P3_SUITE_V1_FAULT_IDS = tuple(FAULTS)  # canonical declaration order


def apply_fault(fault_id: str, source: CaseData, seed: int) -> tuple[CaseData, dict[str, Any]]:
    """Apply ``fault_id`` to a COPY of ``source`` with a generator seeded from ``seed``."""
    if fault_id not in FAULTS:
        raise KeyError(f"unknown fault {fault_id!r}")
    rng = np.random.default_rng(seed)
    return FAULTS[fault_id][1](source, rng)
