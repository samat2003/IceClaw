"""Sealed PAIRED-TILE fault suite (``p7-paired-v1``) for the P7 four-arm comparison.

Built from copies of the ONE verified predicted+observed tile (``ice-s-0001-0002``): the published
NASA-IBM prediction plus the genuine SomBench TMAX (K), LPSR (fraction) and SLOPE (deg) layers.
The P3 (``p3-prediction-only-v1``) and P4 (``p4-multimodal-v1``) suites are separate and unchanged.

Firewall (same design as P3/P4):

* agent-visible: ``<scenarios_root>/<suite>/<case>/`` with ``meta.json`` and per-layer
  ``<layer>.values.npy`` / ``<layer>.covered.npy`` under an opaque case id;
* evaluator-only: ``<eval_root>/<suite>/sealed_key.json`` (fault ids, parameters, seeds,
  pre-registered expectations, suite seed) and ``salt.hex``;
* the public ``manifest.json`` holds per-file SHA256s and a salted commitment to the sealed key.

The expert target ``PRO`` is not used by this suite and never enters either root. Every case is a
copy of ONE tile: cases share pixels, so they are not statistically independent samples, and the
faults are synthetic corruptions, not genuine lunar anomalies.

This module never imports ``iceclaw.agents`` or ``iceclaw.swarm`` (P3 import firewall).
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from iceclaw.contracts import AgentRegionContext, RunBudget, SourceMode
from iceclaw.data.catalog import PREDICTION_MODALITY, available_tool_names, proj_string
from iceclaw.evaluation.faults import FaultSpec
from iceclaw.evaluation.fixtures import (
    GENERATOR_VERSION,
    AgentScenario,
    CaseEntry,
    SuiteAlreadyFrozen,
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

SUITE_ID = "p7-paired-v1"
PRED = PREDICTION_MODALITY
TEMP, SHADOW, TERRAIN = "pixelwise_temperature", "pixelwise_shadow_mask", "pixelwise_terrain"
LAYER_ORDER = (PRED, TEMP, SHADOW, TERRAIN)
EXPECTED_UNITS = {PRED: "prospectivity_index", TEMP: "K", SHADOW: "fraction", TERRAIN: "deg"}
# Public, identical for every case (the benchmark runner replaces it with the arm budget it
# declares; this value only documents the suite's intended scale).
SUITE_BUDGET = RunBudget(
    max_events=48, max_hops=5, max_tool_calls=16, max_llm_calls=16, timeout_s=360
)


# ------------------------------------------------------------------------------ case data
@dataclass(frozen=True)
class PairedCase:
    layers: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (values, covered)
    units: dict[str, str]
    footprint: tuple[tuple[float, float], ...]
    grid: dict[str, Any]
    projection: dict[str, Any]
    resolution_m: float
    pole: str

    def copy(self) -> PairedCase:
        return replace(self, layers={k: (v.copy(), c.copy()) for k, (v, c) in self.layers.items()})


@dataclass(frozen=True)
class PairedSource:
    """The single genuine paired tile (immutable) and its public provenance."""

    region_id: str  # sealed: never written to the agent-visible root
    case: PairedCase
    provenance: dict[str, str]  # public provenance (dataset revisions, source mode, hashes)

    def __post_init__(self) -> None:
        for v, c in self.case.layers.values():
            v.setflags(write=False)
            c.setflags(write=False)


def paired_source_from_arrays(
    arrays: RegionArrays, pole: str, provenance: dict[str, str]
) -> PairedSource:
    missing = [n for n in LAYER_ORDER if n not in arrays.layers]
    if missing:
        raise ValueError(f"paired source needs all four layers; missing {missing}")
    for n in LAYER_ORDER:
        if arrays.layers[n].unit != EXPECTED_UNITS[n]:
            raise ValueError(f"{n} unit {arrays.layers[n].unit!r} != {EXPECTED_UNITS[n]!r}")
    case = PairedCase(
        {n: (arrays.layers[n].values.copy(), arrays.layers[n].covered.copy()) for n in LAYER_ORDER},
        {n: arrays.layers[n].unit for n in LAYER_ORDER},
        tuple(tuple(c) for c in arrays.footprint),
        dict(arrays.grid),
        dict(arrays.projection),
        float(arrays.resolution_m),
        pole,
    )
    return PairedSource(arrays.region_id, case, dict(provenance))


def load_real_paired_source(settings: Any, region_id: str = "ice-s-0001-0002") -> PairedSource:
    """The verified PAIRED catalog tile (checksums re-verified by the catalog)."""
    from iceclaw.data.catalog import discover

    catalog = discover(settings)
    rec = catalog.get(region_id)
    if rec.status != "PAIRED" or rec.entry.readiness.value != "verified":
        raise ValueError(f"{region_id} is not a verified paired region ({rec.status})")
    arrays = catalog.load_arrays(region_id)
    e = rec.entry
    prov = {
        "source_mode": e.source_mode.value,
        "source_dataset": e.source_dataset,
        "source_revision": e.source_revision,
        "source_split": e.source_split,
        "prediction_sha256": e.prediction_sha256,
        "observation_layers_sha256": canonical_json(
            {n: arrays.layers[n].sha256 for n in (TEMP, SHADOW, TERRAIN)}
        ),
    }
    return paired_source_from_arrays(arrays, e.pole, prov)


# ------------------------------------------------------------------------------- faults
Builder = Callable[[PairedCase, np.random.Generator], tuple[PairedCase, dict[str, Any]]]


def _set(c: PairedCase, name: str, values: np.ndarray, covered: np.ndarray | None = None):
    out = c.copy()
    out.layers[name] = (values, out.layers[name][1] if covered is None else covered)
    return out


def _clean_identical(c, rng):
    return c.copy(), {}


def _clean_float32(c, rng):
    out = c.copy()
    for n, (v, cov) in list(out.layers.items()):
        out.layers[n] = (v.astype(np.float32).astype(np.float64), cov)
    return out, {"dtype": "float32"}


def _clean_noise(c, rng):
    sig = {PRED: 1e-6, TEMP: 1e-3, SHADOW: 0.0, TERRAIN: 1e-4}
    out = c.copy()
    for n, s in sig.items():
        if s:
            v, cov = out.layers[n]
            out.layers[n] = (v + rng.normal(0.0, s, v.shape), cov)
    return out, {"sigma": sig}


def _pred_percent(c, rng):
    v, _ = c.layers[PRED]
    return _set(c, PRED, v * 100.0), {"factor": 100.0}


def _pred_offset(c, rng):
    v, _ = c.layers[PRED]
    return _set(c, PRED, v + 0.03), {"offset": 0.03}


def _roll(v: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return np.roll(np.roll(v, dr, axis=0), dc, axis=1)


def _pred_misregistered(c, rng):
    dr, dc = int(rng.integers(15, 25)), int(rng.integers(15, 25))
    v, cov = c.layers[PRED]
    return _set(c, PRED, _roll(v, dr, dc), _roll(cov, dr, dc)), {"shift_px": [dr, dc]}


def _block(shape, size, rng):
    r = int(rng.integers(size, shape[0] - 2 * size))
    q = int(rng.integers(size, shape[1] - 2 * size))
    return r, q, size


def _pred_nan_block(c, rng):
    v, cov = c.layers[PRED]
    r, q, s = _block(v.shape, max(4, v.shape[0] // 10), rng)
    v = v.copy()
    v[r : r + s, q : q + s] = np.nan  # coverage mask NOT updated
    return _set(c, PRED, v, cov), {"block": [r, q, s]}


def _temp_celsius(c, rng):
    v, _ = c.layers[TEMP]
    return _set(c, TEMP, v - 273.15), {"offset_k": -273.15}


def _shadow_inverted(c, rng):
    v, _ = c.layers[SHADOW]
    return _set(c, SHADOW, 1.0 - v), {"transform": "1 - mask"}


def _temp_misregistered(c, rng):
    dr, dc = int(rng.integers(25, 35)), int(rng.integers(25, 35))
    v, cov = c.layers[TEMP]
    return _set(c, TEMP, _roll(v, dr, dc), _roll(cov, dr, dc)), {"shift_px": [dr, dc]}


def _slope_radians(c, rng):
    v, _ = c.layers[TERRAIN]
    return _set(c, TERRAIN, np.deg2rad(v)), {"transform": "deg2rad"}


def _spec(fid, cat, is_fault, intent, sev, exp, why=""):
    return FaultSpec(fid, cat, is_fault, intent, sev, exp, why)


# Pre-registered BEFORE any agent or baseline run on this suite. ``expected_detectable`` is the
# author's expectation for the deterministic baseline only (same author as the baseline, so a
# plumbing check, not an independent difficulty estimate).
PAIRED_FAULTS: dict[str, tuple[FaultSpec, Builder]] = {
    s.fault_id: (s, b)
    for s, b in [
        (_spec("clean_identical", "clean", False, "unmodified copy", "none", None), _clean_identical),
        (_spec("clean_identical_repeat", "clean", False, "second unmodified copy", "none", None), _clean_identical),
        (_spec("clean_float32_roundtrip", "clean", False, "benign float32 round-trip of all layers", "none", None), _clean_float32),
        (_spec("clean_subthreshold_noise", "clean", False, "benign tiny noise (1e-6 index, 1e-3 K, 1e-4 deg)", "none", None), _clean_noise),
        (_spec("prediction_percent_scale", "value", True, "prediction index multiplied by 100", "high", True,
               "raw index leaves any plausible range (rule prediction_range)"), _pred_percent),
        (_spec("prediction_offset_small", "value", True, "constant +0.03 bias on the prediction index", "low", False,
               "stays near the natural range; no reference-free rule can see a small bias"), _pred_offset),
        (_spec("prediction_misregistered", "geometry", True, "prediction shifted 15-24 px relative to the observation layers", "medium", False,
               "per-layer statistics are unchanged; needs a cross-layer spatial comparison"), _pred_misregistered),
        (_spec("prediction_nan_block", "value", True, "block of covered prediction pixels set to NaN, mask not updated", "medium", True,
               "non-finite covered pixels (rule missing_data)"), _pred_nan_block),
        (_spec("temperature_unit_celsius", "value", True, "TMAX in degrees Celsius instead of K", "high", True,
               "temperatures fall below any physical Kelvin value (rule temperature_range)"), _temp_celsius),
        (_spec("shadow_mask_inverted", "value", True, "LPSR replaced by 1 - LPSR", "high", True,
               "shadowed pixels become warmer than sunlit ones (rule psr_temperature_sign)"), _shadow_inverted),
        (_spec("temperature_misregistered", "geometry", True, "TMAX shifted 25-34 px relative to LPSR/SLOPE/prediction", "medium", True,
               "decorrelating the ~7% PSR pixels from their cold temperatures should collapse the "
               "PSR/non-PSR contrast above the -20 K sign threshold (rule psr_temperature_sign)"), _temp_misregistered),
        (_spec("slope_unit_radians", "value", True, "SLOPE in radians instead of degrees", "high", True,
               "slope maximum collapses below 1 deg (rule slope_range)"), _slope_radians),
    ]
}  # fmt: skip
PAIRED_V1_FAULT_IDS = tuple(PAIRED_FAULTS)

# P7 v2 (pre-registered before the v2 suite was generated): 2 clean controls + 4 faults, chosen to
# span one single-layer unit error and three CROSS-LAYER faults (where combining specialists'
# evidence could matter). Same builders as v1; the v2 suite gets a fresh private seed and salt.
PAIRED_V2_SUITE_ID = "p7-paired-v2"
PAIRED_V2_FAULT_IDS = (
    "clean_identical",
    "clean_float32_roundtrip",
    "temperature_unit_celsius",
    "shadow_mask_inverted",
    "temperature_misregistered",
    "prediction_misregistered",
)


def apply_paired_fault(fault_id: str, src: PairedCase, seed: int) -> tuple[PairedCase, dict]:
    if fault_id not in PAIRED_FAULTS:
        raise KeyError(fault_id)
    return PAIRED_FAULTS[fault_id][1](src, np.random.default_rng(seed))


# --------------------------------------------------------------------------- build / load
def _write_case(scen_dir: Path, case_id: str, case: PairedCase, source_mode: str) -> dict[str, str]:
    d = scen_dir / case_id
    d.mkdir(parents=True)
    meta = {
        "case_id": case_id,
        "layers": list(LAYER_ORDER),
        "units": case.units,
        "shape": list(case.layers[PRED][0].shape),
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


def build_paired_suite(
    source: PairedSource,
    scenarios_root: Path,
    eval_root: Path,
    *,
    suite_seed: int | None = None,
    salt: bytes | None = None,
    suite_id: str = SUITE_ID,
    fault_ids: tuple[str, ...] = PAIRED_V1_FAULT_IDS,
    suite_version: str = "1",
) -> SuiteManifest:
    """Create and freeze the suite. ``suite_seed`` defaults to a fresh PRIVATE random value that
    is stored only in the sealed key. Refuses to overwrite an existing suite."""
    scen_dir, seal_dir = _suite_dir(scenarios_root, suite_id), _suite_dir(eval_root, suite_id)
    if scen_dir.exists() or seal_dir.exists():
        raise SuiteAlreadyFrozen(f"suite {suite_id!r} already exists; create a new versioned suite")
    if Path(scenarios_root).resolve() == Path(eval_root).resolve():
        raise ValueError("scenario and evaluator roots must be different directories")
    suite_seed = suite_seed if suite_seed is not None else secrets.randbits(63)
    salt = salt if salt is not None else secrets.token_bytes(16)
    order = list(range(len(fault_ids)))
    np.random.default_rng(np.random.SeedSequence([suite_seed, 0])).shuffle(order)
    scen_dir.mkdir(parents=True)
    seal_dir.mkdir(parents=True)
    source_mode = source.provenance.get("source_mode", SourceMode.UPSTREAM_PUBLISHED_EXAMPLE.value)
    entries, truth = [], {}
    for pos, ci in enumerate(order, start=1):
        fid = fault_ids[ci]
        case_id = f"case-{pos:02d}"
        seed = int(np.random.SeedSequence([suite_seed, 1, ci]).generate_state(1)[0])
        case, params = apply_paired_fault(fid, source.case, seed)
        files = _write_case(scen_dir, case_id, case, source_mode)
        spec = PAIRED_FAULTS[fid][0]
        entries.append(
            CaseEntry(
                case_id=case_id,
                task="fault_detection",
                modalities=list(LAYER_ORDER),
                budget=SUITE_BUDGET,
                repeat_index=0,
                source_group="g1",
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
            "source_region": source.region_id,
        }
    sealed = {"suite_id": suite_id, "suite_seed": suite_seed, "cases": truth, "secondary": {}}
    sealed_text = canonical_json(sealed)
    (seal_dir / "sealed_key.json").write_text(sealed_text)
    (seal_dir / "salt.hex").write_text(salt.hex())
    manifest = SuiteManifest(
        suite_id=suite_id,
        suite_version=suite_version,
        generator_version=GENERATOR_VERSION,
        source={
            **{k: v for k, v in source.provenance.items()},
            "scope": "paired: NASA-IBM prediction + TMAX K + LPSR fraction + SLOPE deg; every "
            "case is a copy of ONE source tile",
        },
        independence={
            "independent_source_tiles": 1,
            "statistical_unit": "case",
            "warning": (
                "all cases derive from one tile and share its pixels; synthetic faults are not "
                "lunar anomalies; metrics are exploratory and do not generalize beyond this tile"
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


def load_paired_scenario(scenarios_root: Path, suite_id: str, case_id: str) -> AgentScenario:
    """Load one agent-visible case, verifying every file against the frozen manifest."""
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
    layers = {
        n: RasterLayer(
            n,
            np.load(d / f"{n}.values.npy", allow_pickle=False),
            np.load(d / f"{n}.covered.npy", allow_pickle=False),
            meta["units"][n],
            f"{case_id}/{n}.values.npy",
            entry.files[f"{n}.values.npy"],
        )
        for n in meta["layers"]
    }
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
        available_tools=available_tool_names(arrays),
        budget=entry.budget,
    )
    return AgentScenario(case_id, context, arrays)


def verify_paired_suite(
    scenarios_root: Path, eval_root: Path | None = None, suite_id: str = SUITE_ID
) -> list[str]:
    problems: list[str] = []
    try:
        manifest = load_manifest(scenarios_root, suite_id)
    except (SuiteIntegrityError, OSError, ValueError) as e:
        return [f"manifest: {e}"]
    for entry in manifest.cases:
        try:
            load_paired_scenario(scenarios_root, suite_id, entry.case_id)
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
class PairedBaselineConfig:
    """Pre-registered deterministic rules (v1, frozen before any run on ``p7-paired-v1``).

    Physical bounds are screening limits, not laws. The shadow tolerance follows the P5 v2
    lesson (1e-4, well above float32 round-off and the clean-noise control).
    """

    version: str = "paired-v1"
    prediction_min: float = -0.5
    prediction_max: float = 1.5
    prediction_outside_unit_max_fraction: float = 0.10
    temp_min_k: float = 20.0
    temp_max_k: float = 450.0
    slope_max_floor_deg: float = 1.0
    slope_max_ceiling_deg: float = 90.0
    shadow_tol: float = 1e-4
    psr_sign_min_difference_k: float = -20.0  # PSR must be at least 20 K colder than non-PSR

    def sha256(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode())


@dataclass(frozen=True)
class PairedRule:
    rule_id: str
    triggered: bool
    detail: str


@dataclass(frozen=True)
class PairedBaselineResult:
    case_id: str
    flagged: bool
    rules: tuple[PairedRule, ...]
    tool_run_ids: tuple[str, ...]

    @property
    def triggered_rules(self) -> list[str]:
        return [r.rule_id for r in self.rules if r.triggered]


def run_paired_baseline(
    scenario: AgentScenario, cfg: PairedBaselineConfig | None = None
) -> PairedBaselineResult:
    """Reference-free QC rules over the existing P2 tools. Flagged = any rule triggers."""
    cfg = cfg or PairedBaselineConfig()
    a = scenario.arrays
    ids: list[str] = []
    rules: list[PairedRule] = []

    def tool(tool_id: str, **kw):
        try:
            m = run_tool(a, tool_id, **kw)
        except ScienceToolError:
            return None
        ids.append(m.tool_run_id)
        return m.value

    nonfinite, empty = [], []
    valid: dict[str, float] = {}
    for n in LAYER_ORDER:
        m = tool("missingness", modality=n)
        if m is None:
            continue
        valid[n] = m["valid_pixels"]
        if m["nonfinite_covered_pixels"] > 0:
            nonfinite.append(n)
        if m["valid_pixels"] == 0:
            empty.append(n)
    rules.append(PairedRule("missing_data", bool(nonfinite), f"non-finite covered: {nonfinite}"))
    rules.append(PairedRule("layer_empty", bool(empty), f"no valid pixels: {empty}"))

    p = tool("valid_pixel_summary", modality=PRED)
    rules.append(
        PairedRule(
            "prediction_range",
            p is not None and (p["min"] < cfg.prediction_min or p["max"] > cfg.prediction_max),
            "n/a" if p is None else f"min {p['min']:.4g} max {p['max']:.4g}",
        )
    )
    u = tool("unit_interval_check", modality=PRED)
    frac = None
    if u is not None and valid.get(PRED):
        frac = u["outside_unit_interval_pixels"] / valid[PRED]
    rules.append(
        PairedRule(
            "prediction_outside_unit_fraction",
            frac is not None and frac > cfg.prediction_outside_unit_max_fraction,
            "n/a" if frac is None else f"{frac:.4f} of valid pixels outside [0,1]",
        )
    )
    t, s, sh = (
        tool("valid_pixel_summary", modality=TEMP),
        tool("valid_pixel_summary", modality=TERRAIN),
        tool("valid_pixel_summary", modality=SHADOW),
    )
    rules.append(
        PairedRule(
            "temperature_range",
            t is not None and (t["min"] < cfg.temp_min_k or t["max"] > cfg.temp_max_k),
            "n/a" if t is None else f"min {t['min']:.4g} max {t['max']:.4g} K",
        )
    )
    rules.append(
        PairedRule(
            "slope_range",
            s is not None
            and (s["max"] < cfg.slope_max_floor_deg or s["max"] > cfg.slope_max_ceiling_deg),
            "n/a" if s is None else f"max {s['max']:.4g} deg",
        )
    )
    rules.append(
        PairedRule(
            "shadow_range",
            sh is not None and (sh["min"] < -cfg.shadow_tol or sh["max"] > 1 + cfg.shadow_tol),
            "n/a" if sh is None else f"min {sh['min']:.4g} max {sh['max']:.4g}",
        )
    )
    th = tool("thermal_psr_consistency")
    diff = None if th is None else th.get("psr_minus_non_psr_k")
    rules.append(
        PairedRule(
            "psr_temperature_sign",
            diff is not None and diff > cfg.psr_sign_min_difference_k,
            "undefined" if diff is None else f"PSR minus non-PSR {diff:.1f} K",
        )
    )
    return PairedBaselineResult(
        scenario.case_id, any(r.triggered for r in rules), tuple(rules), tuple(ids)
    )
