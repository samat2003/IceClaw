"""Deterministic lunar research triage over the verified, paired prediction + TMAX + LPSR + SLOPE.

Pure NumPy over the existing native-grid arrays: no resampling, no LLM, no evaluator or reference
data. A fixed grid of non-overlapping windows (default 16 x 16 native pixels) is screened with a
transparent, conjunctive, per-pixel rule whose thresholds are ANALYST-CHOSEN screening choices
(declared in ``TriageConfig`` before any candidate list was inspected):

    LPSR >= 0.5   AND   TMAX <= 110 K   AND   SLOPE <= 15 deg
                  AND   prediction index >= the tile's 90th percentile of valid raw values

Two clearly separated outputs:

* **follow-up screening windows** -- windows containing at least one pixel meeting all four
  conditions;
* **model-observation review windows** -- windows (not already follow-up windows) where at least
  half of the valid pixels are high-index but at least one observation condition supports fewer
  than half of those high-index pixels. These are evidence discrepancies for review, NOT confirmed
  model faults.

None of this measures ice, ice abundance, a calibrated likelihood, ice stability or landing
safety. The NASA-IBM model was trained on related scientific modalities, so agreement between the
prediction and the observation layers is not independent confirmation of ice. If no window meets
the joint rule the result says so; thresholds are never tuned to produce candidates. Sensitivity
to predeclared nearby choices is reported separately from the default configuration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from iceclaw.data import moonatlas_adapter as ma
from iceclaw.science.common import RegionArrays

TRIAGE_VERSION = "iceclaw-triage/1"
PRED = "nasa_ibm_prediction_raw"
TEMP = "pixelwise_temperature"
SHADOW = "pixelwise_shadow_mask"
TERRAIN = "pixelwise_terrain"
LAYERS = (PRED, TEMP, SHADOW, TERRAIN)
EXPECTED_UNITS = {PRED: "prospectivity_index", TEMP: "K", SHADOW: "fraction", TERRAIN: "deg"}
FOOTPRINT_TOLERANCE_DEG = 1e-3
MAX_EXAMPLE_PIXELS = 5
SENSITIVITY_KEYS = ("tmax_max_k", "slope_max_deg", "index_percentile")


class TriageError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class TriageConfig:
    """Versioned screening configuration. Its SHA256 is recorded in every artifact."""

    version: str = TRIAGE_VERSION
    window_px: int = 16
    shadow_min: float = 0.5  # pixel-level LPSR >= shadow_min counts as permanently shadowed
    tmax_max_k: float = 110.0  # analyst-chosen screening threshold, not a stability limit
    slope_max_deg: float = 15.0  # analyst-chosen screening threshold
    index_percentile: float = 90.0  # of the tile's valid raw prediction values (no clipping)
    review_min_high_index_fraction: float = 0.5
    review_support_min_fraction: float = 0.5

    def sha256(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


# Predeclared nearby choices (reported separately from the default screening configuration).
SENSITIVITY_VARIANTS: tuple[tuple[str, dict[str, float]], ...] = (
    ("tmax_100K", {"tmax_max_k": 100.0}),
    ("tmax_120K", {"tmax_max_k": 120.0}),
    ("slope_10deg", {"slope_max_deg": 10.0}),
    ("slope_20deg", {"slope_max_deg": 20.0}),
    ("index_p85", {"index_percentile": 85.0}),
    ("index_p95", {"index_percentile": 95.0}),
    ("strict_all", {"tmax_max_k": 100.0, "slope_max_deg": 10.0, "index_percentile": 95.0}),
    ("relaxed_all", {"tmax_max_k": 120.0, "slope_max_deg": 20.0, "index_percentile": 85.0}),
)

LIMITATIONS = (
    "Thresholds (LPSR >= 0.5, TMAX <= 110 K, SLOPE <= 15 deg, index >= tile P90) are "
    "analyst-chosen screening choices, not proven physical ice-stability limits and not "
    "calibrated probabilities.",
    "The NASA-IBM model was trained on related scientific modalities: agreement between the "
    "prediction and these observation layers is not independent confirmation of ice.",
    "Windows are spatially correlated pixels of ONE tile; counts are not independent samples.",
    "Review windows are evidence discrepancies for human review, not confirmed model faults.",
    "No ice presence, ice abundance, calibrated likelihood, landing safety or mission "
    "prioritization is inferred.",
)


def _r(x: Any) -> Any:
    """Round to 9 significant digits so JSON round-trips reproduce identical numbers."""
    if x is None:
        return None
    return float(f"{float(x):.9g}")


def _validate(arrays: RegionArrays, cfg: TriageConfig) -> tuple[int, int]:
    missing = [n for n in LAYERS if n not in arrays.layers]
    if missing:
        raise TriageError("MISSING_LAYER", f"need all four layers; missing {missing}")
    if not (2 <= cfg.window_px) or not 0.0 < cfg.index_percentile < 100.0:
        raise TriageError("BAD_CONFIG", "window_px >= 2 and 0 < index_percentile < 100 required")
    for n in LAYERS:
        if arrays.layers[n].unit != EXPECTED_UNITS[n]:
            raise TriageError(
                "UNIT_MISMATCH", f"{n}: unit {arrays.layers[n].unit!r} != {EXPECTED_UNITS[n]!r}"
            )
    shapes = {arrays.layers[n].values.shape for n in LAYERS}
    shapes |= {arrays.layers[n].covered.shape for n in LAYERS}
    if len(shapes) != 1:
        raise TriageError("SHAPE_MISMATCH", f"layers are not on one grid: {sorted(shapes)}")
    (h, w) = next(iter(shapes))
    if h < cfg.window_px or w < cfg.window_px:
        raise TriageError("BAD_CONFIG", "grid smaller than one window")
    corners = ((0, 0), (w, 0), (w, h), (0, h))  # outer pixel corners NW, NE, SE, SW (col, row)
    for (c, r), (lat, lon) in zip(corners, arrays.footprint, strict=False):
        la, lo = ma.pixel_to_lat_lon(arrays.projection, arrays.grid, c, r)
        if (
            abs(la - lat) > FOOTPRINT_TOLERANCE_DEG
            or ma.lon_diff(lo, lon) > FOOTPRINT_TOLERANCE_DEG
        ):
            raise TriageError(
                "GEOREFERENCE_MISMATCH", "footprint does not match the projection and grid"
            )
    return h, w


def _valid(arrays: RegionArrays, name: str) -> np.ndarray:
    lay = arrays.layers[name]
    return lay.covered & np.isfinite(lay.values)


def _summ(x: np.ndarray) -> dict[str, Any]:
    if x.size == 0:
        return {"n": 0, "min": None, "median": None, "max": None, "mean": None}
    return {
        "n": int(x.size),
        "min": _r(x.min()),
        "median": _r(np.median(x)),
        "max": _r(x.max()),
        "mean": _r(x.mean()),
    }


def _frac(num: int, den: int) -> float | None:
    return None if den == 0 else _r(num / den)


def screen_arrays(
    arrays: RegionArrays, cfg: TriageConfig | None = None, *, sensitivity: bool = True
) -> dict[str, Any]:
    """Screen every native-grid window. Pure and deterministic; raises ``TriageError``."""
    cfg = cfg or TriageConfig()
    h, w = _validate(arrays, cfg)
    v = {n: arrays.layers[n].values for n in LAYERS}
    ok = {n: _valid(arrays, n) for n in LAYERS}
    ok_all = ok[PRED] & ok[TEMP] & ok[SHADOW] & ok[TERRAIN]
    if not ok[PRED].any():
        raise TriageError("NO_VALID_PREDICTION", "no covered, finite prediction pixels")
    thr = float(np.quantile(v[PRED][ok[PRED]], cfg.index_percentile / 100.0, method="linear"))
    with np.errstate(invalid="ignore"):
        shadow = ok_all & (v[SHADOW] >= cfg.shadow_min)
        cold = ok_all & (v[TEMP] <= cfg.tmax_max_k)
        gentle = ok_all & (v[TERRAIN] <= cfg.slope_max_deg)
        high = ok_all & (v[PRED] >= thr)
    joint = high & shadow & cold & gentle
    px = cfg.window_px
    nr, nc = h // px, w // px
    proj, grid = arrays.projection, arrays.grid
    windows: list[dict[str, Any]] = []
    for r in range(nr):
        for c in range(nc):
            r0, r1, c0, c1 = r * px, (r + 1) * px, c * px, (c + 1) * px
            sl = (slice(r0, r1), slice(c0, c1))
            n_all = int(ok_all[sl].sum())
            n_high = int(high[sl].sum())
            n_joint = int(joint[sl].sum())
            n_sh_valid = int(ok[SHADOW][sl].sum())
            pred_ok, t_ok, s_ok = ok[PRED][sl], ok[TEMP][sl], ok[TERRAIN][sl]
            support = {
                "shadow": _frac(int((high & shadow)[sl].sum()), n_high),
                "cold": _frac(int((high & cold)[sl].sum()), n_high),
                "gentle": _frac(int((high & gentle)[sl].sum()), n_high),
            }
            examples = []
            for rr, cc in np.argwhere(joint[sl])[:MAX_EXAMPLE_PIXELS]:
                lat, lon = ma.pixel_to_lat_lon(proj, grid, c0 + int(cc) + 0.5, r0 + int(rr) + 0.5)
                examples.append(
                    {"row": r0 + int(rr), "col": c0 + int(cc), "lat": _r(lat), "lon": _r(lon)}
                )
            corners = ((c0, r0), (c1, r0), (c1, r1), (c0, r1))
            clat, clon = ma.pixel_to_lat_lon(proj, grid, (c0 + c1) / 2, (r0 + r1) / 2)
            windows.append(
                {
                    "id": r * nc + c,
                    "label": f"w{r * nc + c:03d}",
                    "window": [r0, r1, c0, c1],
                    "centre_lat": _r(clat),
                    "centre_lon": _r(clon),
                    "footprint": [
                        [_r(a), _r(b)]
                        for a, b in (ma.pixel_to_lat_lon(proj, grid, x, y) for x, y in corners)
                    ],
                    "n_pixels": px * px,
                    "n_valid_all_layers": n_all,
                    "coverage": _r(n_all / (px * px)),
                    "prediction_index": _summ(v[PRED][sl][pred_ok]),
                    "shadow_fraction": _frac(
                        int((ok[SHADOW][sl] & (v[SHADOW][sl] >= cfg.shadow_min)).sum()), n_sh_valid
                    ),
                    "tmax_k": _summ(v[TEMP][sl][t_ok]),
                    "slope_deg": _summ(v[TERRAIN][sl][s_ok]),
                    "n_high_index": n_high,
                    "frac_high_index": _frac(n_high, n_all),
                    "support_among_high_index": support,
                    "n_joint": n_joint,
                    "joint_fraction": _frac(n_joint, n_all),
                    "joint_pixels_example": examples,
                }
            )
    followup = sorted(
        (x for x in windows if x["n_joint"] >= 1), key=lambda x: (-x["n_joint"], x["id"])
    )
    follow_ids = {x["id"] for x in followup}
    review, reasons = [], {}
    for x in windows:
        if x["id"] in follow_ids or x["frac_high_index"] is None:
            continue
        if x["frac_high_index"] < cfg.review_min_high_index_fraction:
            continue
        weak = [
            k
            for k in ("shadow", "cold", "gentle")
            if x["support_among_high_index"][k] is not None
            and x["support_among_high_index"][k] < cfg.review_support_min_fraction
        ]
        if weak:
            review.append(x)
            reasons[str(x["id"])] = weak
    review.sort(key=lambda x: (-x["frac_high_index"], x["id"]))
    msgs = []
    if not followup:
        msgs.append(
            "No window contains a pixel meeting all four screening conditions under the default "
            "configuration; no follow-up windows are reported and no threshold was changed."
        )
    if not review:
        msgs.append("No window met the model-observation review rule.")
    counts = {
        "n_windows": len(windows),
        "n_followup_windows": len(followup),
        "n_review_windows": len(review),
        "n_valid_all_layer_pixels": int(ok_all.sum()),
        "n_high_index_pixels": int(high.sum()),
        "n_joint_pixels": int(joint.sum()),
        "pixels_cumulative": {
            "high_index": int(high.sum()),
            "high_and_shadow": int((high & shadow).sum()),
            "high_shadow_cold": int((high & shadow & cold).sum()),
            "all_four": int(joint.sum()),
        },
        "pixels_by_condition": {
            "shadow": int(shadow.sum()),
            "cold": int(cold.sum()),
            "gentle": int(gentle.sum()),
            "shadow_and_cold": int((shadow & cold).sum()),
        },
        "dropped_edge_rows": h - nr * px,
        "dropped_edge_cols": w - nc * px,
    }
    out: dict[str, Any] = {
        "config": asdict(cfg),
        "config_sha256": cfg.sha256(),
        "grid": {"height": h, "width": w, "window_px": px, "n_rows": nr, "n_cols": nc},
        "index_threshold_raw": _r(thr),
        "counts": counts,
        "windows": windows,
        "selection": {
            "followup_ids": [x["id"] for x in followup],
            "review_ids": [x["id"] for x in review],
            "review_reasons": reasons,
            "followup_order": "descending joint-pixel count, then ascending window id",
            "review_order": "descending high-index fraction, then ascending window id",
            "rule_followup": "n_joint >= 1 (LPSR >= shadow_min AND TMAX <= tmax_max_k AND SLOPE <= "
            "slope_max_deg AND index >= index_threshold_raw, all on the same valid pixel)",
            "rule_review": "not a follow-up window AND high-index fraction >= "
            "review_min_high_index_fraction AND at least one of shadow/cold/gentle supports fewer "
            "than review_support_min_fraction of its high-index pixels",
        },
        "messages": msgs,
        "limitations": list(LIMITATIONS),
    }
    if sensitivity:
        rows = []
        for name, over in SENSITIVITY_VARIANTS:
            sub = screen_arrays(arrays, replace(cfg, **over), sensitivity=False)
            sc = sub["counts"]
            rows.append(
                {
                    "variant": name,
                    "overrides": over,
                    "index_threshold_raw": sub["index_threshold_raw"],
                    "n_followup_windows": sc["n_followup_windows"],
                    "n_joint_pixels": sc["n_joint_pixels"],
                    "n_review_windows": sc["n_review_windows"],
                }
            )
        out["sensitivity"] = {
            "note": "predeclared nearby choices; the default configuration above is the only "
            "screening result; these rows show how counts change and select nothing",
            "variants": rows,
        }
    return out


def window_by_id(result: dict[str, Any], window_id: int) -> dict[str, Any]:
    for x in result["windows"]:
        if x["id"] == window_id:
            return x
    raise TriageError("UNKNOWN_WINDOW", f"no window with id {window_id}")


def window_category(result: dict[str, Any], window_id: int) -> str:
    sel = result["selection"]
    if window_id in sel["followup_ids"]:
        return "followup"
    if window_id in sel["review_ids"]:
        return "review"
    return "other"
