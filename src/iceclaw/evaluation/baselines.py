"""Deterministic reference-free QC baseline built on the existing P2 science tools.

Inputs are ``AgentScenario`` objects only: the baseline cannot see fault labels, seeds, the expert
reference or evaluator paths (tested). Thresholds are pre-registered in ``BaselineConfig`` and
its SHA256 is recorded with every result.

Rules (a case is flagged if ANY rule triggers):

* ``missing_data``          non-finite or uncovered pixels above tolerance (P2 ``missingness``)
* ``value_range``           raw max/min outside a plausible range (``valid_pixel_summary``)
* ``outside_unit_fraction`` too many raw values outside [0,1] (``unit_interval_check``)
* ``dynamic_range``         max-min collapsed (``valid_pixel_summary``)
* ``geometry_consistency``  declared footprint/resolution disagree with grid+projection
                            (P1 ``pixel_to_lat_lon``)
* ``seam_line``             a full row/column of robust-outlier gradient pixels
                            (P2 gradient helpers; ``discontinuity_candidates`` tool is recorded)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from iceclaw.data import moonatlas_adapter as ma
from iceclaw.evaluation.fixtures import AgentScenario, canonical_json, sha256_bytes
from iceclaw.science import raster_metrics as rm
from iceclaw.science.common import ScienceToolError
from iceclaw.science.model_audit import run_tool


@dataclass(frozen=True)
class BaselineConfig:
    min_valid_fraction: float = 0.999
    max_plausible: float = 1.5
    min_plausible: float = -0.5
    min_dynamic_range: float = 0.2
    outside_unit_fraction_max: float = 0.10
    footprint_tol_deg: float = 1e-4
    resolution_rel_tol: float = 1e-6
    seam_k_mad: float = 6.0
    seam_line_fraction: float = 0.5

    def sha256(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode())


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    triggered: bool
    detail: str


@dataclass(frozen=True)
class BaselineCaseResult:
    case_id: str
    flagged: bool
    rules: tuple[RuleResult, ...]
    tool_run_ids: tuple[str, ...]

    @property
    def triggered_rules(self) -> list[str]:
        return [r.rule_id for r in self.rules if r.triggered]


def _geometry_rule(s: AgentScenario, cfg: BaselineConfig) -> RuleResult:
    a = s.arrays
    layer = next(iter(a.layers.values()))
    h, w = layer.values.shape
    problems = []
    corners = ((0, 0), (w, 0), (w, h), (0, h))
    worst = 0.0
    for (clat, clon), (c, r) in zip(a.footprint, corners, strict=True):
        lat, lon = ma.pixel_to_lat_lon(a.projection, a.grid, c, r)
        worst = max(worst, abs(lat - clat), ma.lon_diff(lon, clon))
    if worst > cfg.footprint_tol_deg:
        problems.append(f"footprint off by {worst:.4g} deg")
    px = a.grid["pixelSizeXM"]
    if abs(a.resolution_m - px) / px > cfg.resolution_rel_tol:
        problems.append(f"resolution {a.resolution_m:g} m != grid pixel {px:g} m")
    return RuleResult("geometry_consistency", bool(problems), "; ".join(problems))


def _seam_rule(s: AgentScenario, cfg: BaselineConfig) -> RuleResult:
    layer = next(iter(s.arrays.layers.values()))
    grad = rm.gradient_magnitude(layer.values, layer.covered, s.arrays.resolution_m)
    try:
        thr, _, _ = rm.discontinuity_candidates(grad, cfg.seam_k_mad, 0)
    except ScienceToolError as e:
        return RuleResult("seam_line", False, e.code)
    finite = np.isfinite(grad)
    cand = finite & (grad > thr)
    best = 0.0
    for axis in (0, 1):
        n = finite.sum(axis=axis)
        frac = np.divide(cand.sum(axis=axis), n, out=np.zeros(n.shape), where=n > 0)
        best = max(best, float(frac.max()))
    return RuleResult(
        "seam_line", best >= cfg.seam_line_fraction, f"max line outlier fraction {best:.3f}"
    )


def run_baseline(scenario: AgentScenario, cfg: BaselineConfig | None = None) -> BaselineCaseResult:
    cfg = cfg or BaselineConfig()
    arrays = scenario.arrays
    ids: list[str] = []
    rules: list[RuleResult] = []

    miss = run_tool(arrays, "missingness")
    ids.append(miss.tool_run_id)
    total = miss.value["total_pixels"]
    valid_frac = miss.value["valid_pixels"] / total
    rules.append(
        RuleResult(
            "missing_data",
            valid_frac < cfg.min_valid_fraction,
            f"valid fraction {valid_frac:.4f}",
        )
    )

    try:
        summ = run_tool(arrays, "valid_pixel_summary")
        unit = run_tool(arrays, "unit_interval_check")
        ids += [summ.tool_run_id, unit.tool_run_id]
        v = summ.value
        rules.append(
            RuleResult(
                "value_range",
                v["max"] > cfg.max_plausible or v["min"] < cfg.min_plausible,
                f"min {v['min']:.4g}, max {v['max']:.4g}",
            )
        )
        frac_out = unit.value["outside_unit_interval_pixels"] / summ.valid_pixel_count
        rules.append(
            RuleResult(
                "outside_unit_fraction",
                frac_out > cfg.outside_unit_fraction_max,
                f"{frac_out:.4f} of valid pixels outside [0,1]",
            )
        )
        rng = v["max"] - v["min"]
        rules.append(RuleResult("dynamic_range", rng < cfg.min_dynamic_range, f"max-min {rng:.4g}"))
    except ScienceToolError as e:
        for rid in ("value_range", "outside_unit_fraction", "dynamic_range"):
            rules.append(RuleResult(rid, False, e.code))

    rules.append(_geometry_rule(scenario, cfg))
    rules.append(_seam_rule(scenario, cfg))
    try:
        ids.append(run_tool(arrays, "discontinuity_candidates", k_mad=cfg.seam_k_mad).tool_run_id)
    except ScienceToolError:
        pass
    return BaselineCaseResult(
        scenario.case_id, any(r.triggered for r in rules), tuple(rules), tuple(ids)
    )


def rank_windows_baseline(
    scenario: AgentScenario, windows: list[list[int]], top_k: int
) -> list[int]:
    """Reference-free ranking of candidate windows by maximum local prediction gradient.

    Returns the ids (indices into ``windows``) of the top-K windows, ties broken by id.
    """
    layer = next(iter(scenario.arrays.layers.values()))
    grad = rm.gradient_magnitude(layer.values, layer.covered, scenario.arrays.resolution_m)
    scores = []
    for r0, r1, c0, c1 in windows:
        g = grad[r0:r1, c0:c1]
        g = g[np.isfinite(g)]
        scores.append(float(g.max()) if g.size else float("-inf"))
    order = sorted(range(len(windows)), key=lambda i: (-scores[i], i))
    return order[:top_k]
