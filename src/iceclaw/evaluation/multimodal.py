"""Sealed MULTIMODAL fault suite (``p4-multimodal-v1``) built from genuine SomBench observations.

Separate from, and never altering, the frozen P3 prediction-only suite. Same firewall design:
agent-visible case files under the scenarios root, fault key + salt under the evaluator root,
public frozen manifest with a salted commitment. Cases use only layers with an ESTABLISHED unit
(``pixelwise_temperature`` K, ``pixelwise_shadow_mask`` fraction, ``pixelwise_terrain`` deg).

Design: 12 genuine south-polar test tiles (12 independent source tiles), one tile per case,
8 synthetic faults + 4 clean controls. Each fault therefore sits on a DIFFERENT tile, so fault
type and tile effects are confounded (the P3 suite is the controlled within-tile design).
Faults are synthetic corruptions of genuine arrays, never real NASA anomalies. The expert
target ``PRO`` is not used here and never enters either root.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from iceclaw.contracts import AgentRegionContext, RunBudget, SourceMode
from iceclaw.data import sombench_adapter as sa
from iceclaw.data.catalog import (
    RegionCatalog,
    observation_arrays,
    proj_string,
)
from iceclaw.evaluation.faults import FaultSpec
from iceclaw.evaluation.fixtures import (
    GENERATOR_VERSION,
    PUBLIC_BUDGET,
    AgentScenario,
    CaseEntry,
    SuiteIntegrityError,
    SuiteManifest,
    _npy_bytes,
    _suite_dir,
    canonical_json,
    load_manifest,
    load_sealed,
    sha256_bytes,
)
from iceclaw.science.common import RasterLayer, RegionArrays, ScienceToolError
from iceclaw.science.model_audit import run_tool

SUITE_ID = "p4-multimodal-v1"
DEFAULT_SUITE_SEED = 20261101
TEMP, SHADOW, TERRAIN = "pixelwise_temperature", "pixelwise_shadow_mask", "pixelwise_terrain"
LAYER_ORDER = (TEMP, SHADOW, TERRAIN)
OBSERVATION_TOOLS = (
    "discontinuity_candidates",
    "gradient_stats",
    "missingness",
    "quantiles",
    "terrain_slope_summary",
    "thermal_psr_consistency",
    "valid_pixel_summary",
)


# ------------------------------------------------------------------------------ case data
@dataclass(frozen=True)
class ObsCase:
    layers: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (values, covered)
    units: dict[str, str]
    footprint: tuple[tuple[float, float], ...]
    grid: dict[str, Any]
    projection: dict[str, Any]
    resolution_m: float
    pole: str

    def copy(self) -> ObsCase:
        return replace(self, layers={k: (v.copy(), c.copy()) for k, (v, c) in self.layers.items()})


def obs_case_from_arrays(arr: RegionArrays, pole: str) -> ObsCase:
    layers = {n: (arr.layers[n].values.copy(), arr.layers[n].covered.copy()) for n in LAYER_ORDER}
    return ObsCase(
        layers,
        {n: arr.layers[n].unit for n in LAYER_ORDER},
        tuple(arr.footprint),
        dict(arr.grid),
        dict(arr.projection),
        arr.resolution_m,
        pole,
    )


@dataclass(frozen=True)
class ObsTile:
    """One genuine source tile (immutable) with provenance."""

    tile_id: str  # SomBench patch id (sealed: not shown to agents)
    split: str
    case: ObsCase
    file_sha256: dict[str, str]  # SomBench layer -> sha256 of the original GeoTIFF

    def __post_init__(self) -> None:
        for v, c in self.case.layers.values():
            v.setflags(write=False)
            c.setflags(write=False)


def load_observation_tiles(catalog: RegionCatalog) -> list[ObsTile]:
    """All verified observation patches known to the catalog (paired or observation-only)."""
    patches = [r.observation for r in catalog.records.values() if r.observation is not None]
    patches += [o.patch for o in catalog.observation_only.values()]
    tiles = []
    for p in sorted(patches, key=lambda x: x.patch_id):
        arr = observation_arrays(p)
        tiles.append(
            ObsTile(
                p.patch_id,
                p.split,
                obs_case_from_arrays(arr, p.pole),
                {k: v.sha256 for k, v in sorted(p.layers.items())},
            )
        )
    return tiles


# ------------------------------------------------------------------------------- faults
Builder = Callable[[ObsCase, np.random.Generator], tuple[ObsCase, dict[str, Any]]]


def _set(c: ObsCase, name: str, values: np.ndarray, covered: np.ndarray | None = None) -> ObsCase:
    out = c.copy()
    out.layers[name] = (values, out.layers[name][1] if covered is None else covered)
    return out


def _clean_identical(c, rng):
    return c.copy(), {}


def _clean_float32(c, rng):
    out = c.copy()
    for n, (v, cov) in list(out.layers.items()):
        out.layers[n] = (v.astype(np.float32).astype(np.float64), cov)
    return out, {"dtype": "f4"}


def _clean_noise(c, rng):
    out = c.copy()
    for n, (v, cov) in list(out.layers.items()):
        out.layers[n] = (v + rng.normal(0.0, 1e-6, v.shape), cov)  # far below any threshold
    return out, {"sigma": 1e-6}


def _temp_celsius(c, rng):
    return _set(c, TEMP, c.layers[TEMP][0] - 273.15), {"offset": -273.15}


def _temp_offset_small(c, rng):
    return _set(c, TEMP, c.layers[TEMP][0] + 3.0), {"offset_k": 3.0}


def _slope_radians(c, rng):
    return _set(c, TERRAIN, np.deg2rad(c.layers[TERRAIN][0])), {"factor": float(np.pi / 180)}


def _psr_inverted(c, rng):
    return _set(c, SHADOW, 1.0 - c.layers[SHADOW][0]), {"op": "1-x"}


def _temp_roll(c, rng):
    w = c.layers[TEMP][0].shape[1]
    shift = int(rng.integers(w // 8, w // 4 + 1))
    out = c.copy()
    v, cov = out.layers[TEMP]
    out.layers[TEMP] = (np.roll(v, shift, axis=1), np.roll(cov, shift, axis=1))
    return out, {"axis": 1, "shift_px": shift}


def _temp_nan_block(c, rng):
    v = c.layers[TEMP][0].copy()
    h, w = v.shape
    n = max(2, min(32, h // 2, w // 2))
    r0, c0 = int(rng.integers(0, h - n + 1)), int(rng.integers(0, w - n + 1))
    v[r0 : r0 + n, c0 : c0 + n] = np.nan  # coverage mask deliberately NOT updated
    return _set(c, TEMP, v), {"row": r0, "col": c0, "size": n}


def _slope_all_nodata(c, rng):
    v, cov = c.layers[TERRAIN]
    return _set(c, TERRAIN, np.full_like(v, np.nan), np.zeros_like(cov)), {"layer": "terrain"}


def _swap_temp_slope(c, rng):
    out = c.copy()
    out.layers[TEMP], out.layers[TERRAIN] = c.layers[TERRAIN], c.layers[TEMP]
    return out, {"swapped": [TEMP, TERRAIN]}


def _spec(fid, cat, is_fault, intent, sev, exp, why=""):
    return FaultSpec(fid, cat, is_fault, intent, sev, exp, why)


OBS_FAULTS: dict[str, tuple[FaultSpec, Builder]] = {
    s.fault_id: (s, b)
    for s, b in [
        (
            _spec("clean_identical", "clean", False, "unmodified copy", "none", None),
            _clean_identical,
        ),
        (
            _spec(
                "clean_identical_second",
                "clean",
                False,
                "unmodified copy (second tile)",
                "none",
                None,
            ),
            _clean_identical,
        ),
        (
            _spec(
                "clean_float32_roundtrip", "clean", False, "benign float32 round-trip", "none", None
            ),
            _clean_float32,
        ),
        (
            _spec("clean_quantization_noise", "clean", False, "benign 1e-6 noise", "none", None),
            _clean_noise,
        ),
        (
            _spec(
                "temperature_unit_celsius",
                "value",
                True,
                "K reported as degrees Celsius",
                "high",
                True,
                "values go far below any physical Kelvin range (rule temperature_range)",
            ),
            _temp_celsius,
        ),
        (
            _spec(
                "temperature_offset_small",
                "value",
                True,
                "constant +3 K bias",
                "low",
                False,
                "stays inside the plausible Kelvin range; no reference-free rule can see it",
            ),
            _temp_offset_small,
        ),
        (
            _spec(
                "slope_unit_radians",
                "value",
                True,
                "degrees reported as radians",
                "high",
                True,
                "slope maximum collapses below 1 degree (rule slope_range)",
            ),
            _slope_radians,
        ),
        (
            _spec(
                "psr_mask_inverted",
                "value",
                True,
                "shadow mask replaced by 1-mask",
                "high",
                True,
                "PSR pixels become warmer than non-PSR (rule psr_temperature_sign)",
            ),
            _psr_inverted,
        ),
        (
            _spec(
                "temperature_spatial_roll",
                "geometry",
                True,
                "temperature layer shifted relative to the other layers",
                "medium",
                False,
                "PSR-temperature contrast may stay below the lenient -20 K sign rule",
            ),
            _temp_roll,
        ),
        (
            _spec(
                "temperature_nan_block",
                "value",
                True,
                "block of covered temperature pixels turned NaN, mask not updated",
                "medium",
                True,
                "non-finite covered pixels (rule missing_data)",
            ),
            _temp_nan_block,
        ),
        (
            _spec(
                "slope_layer_empty",
                "value",
                True,
                "slope layer entirely nodata",
                "high",
                True,
                "no valid pixels (rule layer_empty)",
            ),
            _slope_all_nodata,
        ),
        (
            _spec(
                "layers_swapped_temp_slope",
                "value",
                True,
                "temperature and slope arrays exchanged (units mislabeled)",
                "high",
                True,
                "temperature minimum falls to slope values (rule temperature_range)",
            ),
            _swap_temp_slope,
        ),
    ]
}
MULTIMODAL_V1_FAULT_IDS = tuple(OBS_FAULTS)


def apply_obs_fault(fault_id: str, tile: ObsCase, seed: int) -> tuple[ObsCase, dict[str, Any]]:
    if fault_id not in OBS_FAULTS:
        raise KeyError(fault_id)
    return OBS_FAULTS[fault_id][1](tile, np.random.default_rng(seed))


# --------------------------------------------------------------------------- build / load
def _write_obs_case(
    scen_dir: Path, case_id: str, case: ObsCase, source_mode: str
) -> dict[str, str]:
    d = scen_dir / case_id
    d.mkdir(parents=True)
    meta = {
        "case_id": case_id,
        "layers": list(LAYER_ORDER),
        "units": case.units,
        "shape": list(case.layers[TEMP][0].shape),
        "pole": case.pole,
        "source_mode": source_mode,
        "footprint": [list(c) for c in case.footprint],
        "grid": case.grid,
        "projection": case.projection,
        "resolution_m": case.resolution_m,
    }
    blobs = {"meta.json": canonical_json(meta).encode()}
    for n in LAYER_ORDER:
        v, c = case.layers[n]
        blobs[f"{n}.values.npy"] = _npy_bytes(v.astype(np.float64))
        blobs[f"{n}.covered.npy"] = _npy_bytes(c.astype(np.bool_))
    for name, b in blobs.items():
        (d / name).write_bytes(b)
    return {name: sha256_bytes(b) for name, b in blobs.items()}


def build_multimodal_suite(
    tiles: list[ObsTile],
    scenarios_root: Path,
    eval_root: Path,
    *,
    suite_seed: int = DEFAULT_SUITE_SEED,
    salt: bytes | None = None,
    suite_id: str = SUITE_ID,
    fault_ids: tuple[str, ...] = MULTIMODAL_V1_FAULT_IDS,
    dataset_provenance: dict[str, str] | None = None,
) -> SuiteManifest:
    """Create and freeze a multimodal suite (one distinct tile per case)."""
    import secrets

    from iceclaw.evaluation.fixtures import SuiteAlreadyFrozen

    if len(tiles) < len(fault_ids):
        raise ValueError(f"need >= {len(fault_ids)} distinct tiles, got {len(tiles)}")
    if len({t.tile_id for t in tiles}) != len(tiles):
        raise ValueError("tiles must be distinct")
    scen_dir, seal_dir = _suite_dir(scenarios_root, suite_id), _suite_dir(eval_root, suite_id)
    if scen_dir.exists() or seal_dir.exists():
        raise SuiteAlreadyFrozen(f"suite {suite_id!r} already exists; create a new versioned suite")
    if Path(scenarios_root).resolve() == Path(eval_root).resolve():
        raise ValueError("scenario and evaluator roots must be different directories")
    salt = salt if salt is not None else secrets.token_bytes(16)

    ordered = sorted(tiles, key=lambda t: t.tile_id)
    perm = np.random.default_rng(np.random.SeedSequence([suite_seed, 2])).permutation(len(ordered))
    chosen = [ordered[i] for i in perm[: len(fault_ids)]]  # canonical fault i -> chosen[i]
    order = list(range(len(fault_ids)))
    np.random.default_rng(np.random.SeedSequence([suite_seed, 0])).shuffle(order)

    scen_dir.mkdir(parents=True)
    seal_dir.mkdir(parents=True)
    source_mode = SourceMode.OBSERVATION_ONLY.value
    entries, truth = [], {}
    for pos, ci in enumerate(order, start=1):
        fid, tile = fault_ids[ci], chosen[ci]
        case_id = f"case-{pos:02d}"
        seed = int(np.random.SeedSequence([suite_seed, 1, ci]).generate_state(1)[0])
        case, params = apply_obs_fault(fid, tile.case, seed)
        files = _write_obs_case(scen_dir, case_id, case, source_mode)
        spec = OBS_FAULTS[fid][0]
        entries.append(
            CaseEntry(
                case_id=case_id,
                task="fault_detection",
                modalities=list(LAYER_ORDER),
                budget=PUBLIC_BUDGET,
                repeat_index=0,
                source_group=f"g{pos:02d}",
                files=files,
            )
        )
        truth[case_id] = {
            "fault_id": fid,
            "category": spec.category,
            "is_fault": spec.is_fault,
            "severity": spec.severity,
            "intent": spec.intent,
            "expected_detectable": spec.expected_detectable,
            "detectable_rationale": spec.detectable_rationale,
            "params": params,
            "seed": seed,
            "source_tile": tile.tile_id,
            "source_file_sha256": tile.file_sha256,
        }
    sealed = {"suite_id": suite_id, "suite_seed": suite_seed, "cases": truth, "secondary": {}}
    sealed_text = canonical_json(sealed)
    (seal_dir / "sealed_key.json").write_text(sealed_text)
    (seal_dir / "salt.hex").write_text(salt.hex())

    manifest = SuiteManifest(
        suite_id=suite_id,
        suite_version="1",
        generator_version=GENERATOR_VERSION,
        source={
            "dataset": sa.REPO,
            "dataset_revision": sa.PINNED_REVISION,
            "split": "test",
            "scope": "observation-only (TMAX K, LPSR fraction, SLOPE deg); no prediction",
            **(dataset_provenance or {}),
        },
        independence={
            "independent_source_tiles": len(fault_ids),
            "statistical_unit": "case",
            "warning": (
                "each case is a different tile, so fault type and tile are confounded; "
                "n=12 cases is far too small for confidence intervals"
            ),
        },
        cases=entries,
        secondary_task={},
        sealed_commitment=sha256_bytes(salt + sealed_text.encode()),
        metrics=["precision", "recall", "f1", "false_positive_rate"],
    )
    (scen_dir / "manifest.json").write_text(canonical_json(manifest.model_dump(mode="json")))
    (scen_dir / "manifest.sha256").write_text(manifest.body_sha256())
    return manifest


def load_multimodal_scenario(scenarios_root: Path, suite_id: str, case_id: str) -> AgentScenario:
    """Load one agent-visible multimodal case, verifying files against the frozen manifest."""
    manifest = load_manifest(scenarios_root, suite_id)
    entry = manifest.case(case_id)
    d = _suite_dir(scenarios_root, suite_id) / case_id
    for name, expected in entry.files.items():
        p = d / name
        if not p.is_file():
            raise SuiteIntegrityError(f"{case_id}/{name} is missing")
        if sha256_bytes(p.read_bytes()) != expected:
            raise SuiteIntegrityError(f"{case_id}/{name} does not match the frozen manifest")
    meta = json.loads((d / "meta.json").read_text())
    layers = {}
    for n in meta["layers"]:
        layers[n] = RasterLayer(
            n,
            np.load(d / f"{n}.values.npy", allow_pickle=False),
            np.load(d / f"{n}.covered.npy", allow_pickle=False),
            meta["units"][n],
            f"{case_id}/{n}.values.npy",
            entry.files[f"{n}.values.npy"],
        )
    footprint = tuple(tuple(c) for c in meta["footprint"])
    arrays = RegionArrays(
        region_id=case_id,
        footprint=footprint,
        resolution_m=meta["resolution_m"],
        projection=meta["projection"],
        grid=meta["grid"],
        layers=layers,
    )
    h, w = meta["shape"]
    context = AgentRegionContext(
        region_id=case_id,
        pole=meta["pole"],
        footprint=list(footprint),
        width=w,
        height=h,
        spatial_resolution_m=meta["resolution_m"],
        lunar_crs=proj_string(meta["projection"]),
        source_ids=[f"{case_id}/{n}.values.npy" for n in meta["layers"]],
        source_mode=SourceMode(meta["source_mode"]),
        available_modalities=list(meta["layers"]),
        available_tools=list(OBSERVATION_TOOLS),
        budget=entry.budget if isinstance(entry.budget, RunBudget) else RunBudget(**entry.budget),
    )
    return AgentScenario(case_id, context, arrays)


def verify_multimodal_suite(
    scenarios_root: Path, eval_root: Path | None = None, suite_id: str = SUITE_ID
) -> list[str]:
    problems: list[str] = []
    try:
        manifest = load_manifest(scenarios_root, suite_id)
    except (SuiteIntegrityError, OSError, ValueError) as e:
        return [f"manifest: {e}"]
    for entry in manifest.cases:
        try:
            load_multimodal_scenario(scenarios_root, suite_id, entry.case_id)
        except (SuiteIntegrityError, OSError, ValueError) as e:
            problems.append(f"{entry.case_id}: {e}")
    if eval_root is not None:
        try:
            load_sealed(eval_root, manifest)
        except (SuiteIntegrityError, OSError, ValueError) as e:
            problems.append(f"sealed: {e}")
    return problems


# ---------------------------------------------------------------------------- baseline
@dataclass(frozen=True)
class ObsBaselineConfig:
    """Pre-registered thresholds. Physical bounds are screening limits, not laws."""

    temp_min_k: float = 20.0
    temp_max_k: float = 450.0
    slope_max_floor_deg: float = 1.0
    slope_max_ceiling_deg: float = 90.0
    shadow_tol: float = 1e-6
    psr_sign_min_difference_k: float = -20.0  # PSR must be at least 20 K colder than non-PSR

    def sha256(self) -> str:
        from dataclasses import asdict

        return sha256_bytes(canonical_json(asdict(self)).encode())


@dataclass(frozen=True)
class ObsBaselineConfigV2(ObsBaselineConfig):
    """Baseline config **v2** (new version; v1 above and its frozen results are unchanged).

    Single change: ``shadow_tol`` 1e-6 -> 1e-4. v1's tolerance equalled the noise level (1e-6
    sigma) of the suite's own ``clean_quantization_noise`` control, so v1 flagged that clean
    control (the documented P4 false positive). 1e-4 sits ~100x above that noise and float32
    round-off while remaining far below any semantic corruption of a [0,1] mask.

    v2 was NOT tuned or scored on the sealed ``p4-multimodal-v1`` labels. Any comparative
    result using v2 must come from a NEW suite frozen after this config, never from v1's
    suite (see docs/AGENT_RUNTIME.md).
    """

    version: str = "2"
    shadow_tol: float = 1e-4


OBS_BASELINE_CONFIGS: dict[str, ObsBaselineConfig] = {
    "v1": ObsBaselineConfig(),
    "v2": ObsBaselineConfigV2(),
}


@dataclass(frozen=True)
class ObsRuleResult:
    rule_id: str
    triggered: bool
    detail: str


@dataclass(frozen=True)
class ObsBaselineResult:
    case_id: str
    flagged: bool
    rules: tuple[ObsRuleResult, ...]
    tool_run_ids: tuple[str, ...]

    @property
    def triggered_rules(self) -> list[str]:
        return [r.rule_id for r in self.rules if r.triggered]


def run_multimodal_baseline(
    scenario: AgentScenario, cfg: ObsBaselineConfig | None = None
) -> ObsBaselineResult:
    """Reference-free QC over the P2 tools.

    Natural nodata (NaN with covered=False) is NOT a fault.
    """
    cfg = cfg or ObsBaselineConfig()
    a = scenario.arrays
    ids: list[str] = []
    rules: list[ObsRuleResult] = []

    nonfinite, empty = [], []
    for n in LAYER_ORDER:
        m = run_tool(a, "missingness", modality=n)
        ids.append(m.tool_run_id)
        if m.value["nonfinite_covered_pixels"] > 0:
            nonfinite.append(n)
        if m.value["valid_pixels"] == 0:
            empty.append(n)
    rules.append(ObsRuleResult("missing_data", bool(nonfinite), f"non-finite covered: {nonfinite}"))
    rules.append(ObsRuleResult("layer_empty", bool(empty), f"no valid pixels: {empty}"))

    def summ(name):
        try:
            m = run_tool(a, "valid_pixel_summary", modality=name)
            ids.append(m.tool_run_id)
            return m.value
        except ScienceToolError:
            return None

    t, s, sh = summ(TEMP), summ(TERRAIN), summ(SHADOW)
    rules.append(
        ObsRuleResult(
            "temperature_range",
            t is not None and (t["min"] < cfg.temp_min_k or t["max"] > cfg.temp_max_k),
            "n/a" if t is None else f"min {t['min']:.4g} max {t['max']:.4g} K",
        )
    )
    rules.append(
        ObsRuleResult(
            "slope_range",
            s is not None
            and (s["max"] < cfg.slope_max_floor_deg or s["max"] > cfg.slope_max_ceiling_deg),
            "n/a" if s is None else f"max {s['max']:.4g} deg",
        )
    )
    rules.append(
        ObsRuleResult(
            "shadow_range",
            sh is not None and (sh["min"] < -cfg.shadow_tol or sh["max"] > 1 + cfg.shadow_tol),
            "n/a" if sh is None else f"min {sh['min']:.4g} max {sh['max']:.4g}",
        )
    )
    try:
        th = run_tool(a, "thermal_psr_consistency")
        ids.append(th.tool_run_id)
        diff = None if th.value is None else th.value["psr_minus_non_psr_k"]
    except ScienceToolError:
        diff = None  # no PSR (or no sunlit) pixels: contrast undefined, so the rule cannot fire
    rules.append(
        ObsRuleResult(
            "psr_temperature_sign",
            diff is not None and diff > cfg.psr_sign_min_difference_k,
            "undefined" if diff is None else f"PSR minus non-PSR {diff:.1f} K",
        )
    )
    # informational structural screen, never a rule
    try:
        g = run_tool(a, "gradient_stats", modality=TEMP)
        ids.append(g.tool_run_id)
    except ScienceToolError:
        pass
    return ObsBaselineResult(
        scenario.case_id, any(r.triggered for r in rules), tuple(rules), tuple(ids)
    )
