"""Suite construction, freezing, and the physical split between agent-visible and sealed data.

Layout (two physically separate roots; ``load_scenario`` only ever receives the first):

    <scenarios_root>/<suite_id>/manifest.json, manifest.sha256   public case list + commitments
    <scenarios_root>/<suite_id>/<case_id>/{values.npy,covered.npy,meta.json}   agent-visible case
    <eval_root>/<suite_id>/{sealed_key.json,salt.hex,reference.npy}            evaluator only

The public manifest carries a SHA256 *commitment* to the sealed key (salted, so the tiny label
space cannot be brute-forced) and its own SHA256; both are fixed before any benchmark run.
Nothing here uses timestamps, so identical inputs/seed/salt reproduce identical hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from iceclaw.config import Settings
from iceclaw.contracts import AgentRegionContext, RunBudget
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.data.catalog import PREDICTION_MODALITY, arrays_for_record, discover, proj_string
from iceclaw.evaluation.faults import FAULTS, P3_SUITE_V1_FAULT_IDS, CaseData, apply_fault
from iceclaw.science.common import RasterLayer, RegionArrays
from iceclaw.science.model_audit import PREDICTION_TOOLS

SUITE_ID = "p3-prediction-only-v1"
GENERATOR_VERSION = "iceclaw-eval/0.1.0"
DEFAULT_SUITE_SEED = 20261031
DEFAULT_WINDOW_PX = 32
DEFAULT_TOP_K = 8
PUBLIC_BUDGET = RunBudget(
    max_events=40, max_hops=4, max_tool_calls=12, max_llm_calls=12, timeout_s=300
)


class SuiteIntegrityError(RuntimeError):
    """A frozen suite file, commitment or manifest no longer matches its recorded hash."""


class SuiteAlreadyFrozen(RuntimeError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _npy_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(a), allow_pickle=False)
    return buf.getvalue()


# ------------------------------------------------------------------------------- source
@dataclass(frozen=True)
class SourceBundle:
    """Authentic source (prediction + metadata) and the evaluator-only reference array."""

    region_id: str
    pole: str
    source_mode: str
    source_dataset: str
    source_revision: str
    source_split: str
    prediction_sha256: str
    data: CaseData
    reference: np.ndarray  # evaluator-only; NaN where not covered
    reference_sha256: str

    def __post_init__(self) -> None:
        for a in (self.data.values, self.data.covered, self.reference):
            a.setflags(write=False)  # inputs are immutable; faults act on copies


def load_reference(src: ma.SectorSource) -> tuple[np.ndarray, str]:
    """Decode the published expert-reference raster (EVALUATOR USE ONLY).

    Checksum-verified against the processing manifest and compared with upstream's published
    reference statistics within the encoding's quantization tolerance.
    """
    from PIL import Image

    root = src.root
    sector = next(
        s for s in json.loads((root / "ice-sectors.json").read_text()) if s["id"] == src.sector_id
    )
    heat = sector["reference"]["heatmap"]
    checksums = json.loads((root / "processing-manifest.json").read_text())["checksums"]
    rel = heat["rawPath"]
    sha = ma.verify_checksum(root, rel, checksums[rel])
    with Image.open(root / rel) as im:
        rgba = np.asarray(im.convert("RGBA") if im.mode != "RGB" else im.convert("RGB"))
    dec = ma.decode_rgba(rgba, heat["rawEncoding"], expected_shape=(src.height, src.width))
    ref = np.where(dec.covered, dec.values, np.nan)
    stats = sector["reference"]["raw"]
    tol = heat["rawEncoding"]["scale"] / 2 + ma.STATS_ROUNDING
    v = dec.values[dec.covered]
    for key, got in (("min", v.min()), ("max", v.max()), ("mean", v.mean())):
        if abs(float(got) - stats[key]) > tol:
            raise ma.SourceError("CORRUPT_RASTER", f"reference {key} differs from upstream")
    return ref, sha


def load_real_source_bundle(settings: Settings, region_id: str) -> SourceBundle:
    """Authentic source from the real catalog (``ice-s-0001-0002`` published example)."""
    catalog = discover(settings)
    rec = catalog.get(region_id)
    arrays = arrays_for_record(rec)
    layer = arrays.layers[PREDICTION_MODALITY]
    ref, ref_sha = load_reference(rec.source)
    data = CaseData(
        values=layer.values.copy(),
        covered=layer.covered.copy(),
        footprint=tuple(rec.entry.footprint),
        grid=dict(rec.source.grid),
        projection=dict(rec.source.projection),
        resolution_m=rec.entry.spatial_resolution_m,
    )
    return SourceBundle(
        region_id=region_id,
        pole=rec.entry.pole,
        source_mode=rec.entry.source_mode.value,
        source_dataset=rec.entry.source_dataset,
        source_revision=rec.entry.source_revision,
        source_split=rec.entry.source_split,
        prediction_sha256=rec.entry.prediction_sha256,
        data=data,
        reference=ref,
        reference_sha256=ref_sha,
    )


# ------------------------------------------------------------------------------ manifest
class CaseEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    task: str
    modalities: list[str]
    budget: RunBudget
    repeat_index: int = Field(ge=0)
    source_group: str
    files: dict[str, str]  # file name -> sha256


class SuiteManifest(BaseModel):
    """Public, frozen case list. Contains NO fault labels, seeds, reference data or sealed paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    suite_id: str
    suite_version: str
    generator_version: str
    source: dict[str, str]
    independence: dict[str, Any]
    cases: list[CaseEntry]
    secondary_task: dict[str, Any]
    sealed_commitment: str
    metrics: list[str]

    def body_sha256(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")).encode())

    def case(self, case_id: str) -> CaseEntry:
        for c in self.cases:
            if c.case_id == case_id:
                return c
        raise KeyError(case_id)

    @property
    def detection_case_ids(self) -> list[str]:
        return [c.case_id for c in self.cases if c.task == "fault_detection"]


@dataclass(frozen=True)
class SealedKey:
    suite_seed: int
    cases: dict[str, dict[str, Any]]  # case_id -> truth record
    secondary: dict[str, Any]

    @property
    def labels(self) -> dict[str, bool]:
        return {cid: bool(c["is_fault"]) for cid, c in self.cases.items()}


def _suite_dir(root: Path, suite_id: str) -> Path:
    return Path(root) / suite_id


def window_grid(shape: tuple[int, int], window_px: int) -> list[tuple[int, int, int, int]]:
    """Non-overlapping windows (row0,row1,col0,col1) in row-major order; remainders dropped."""
    if window_px < 2:
        raise ValueError("window_px must be >= 2")
    nr, nc = shape[0] // window_px, shape[1] // window_px
    if nr < 2 or nc < 2:
        raise ValueError("grid too small for the requested window size")
    return [
        (r * window_px, (r + 1) * window_px, c * window_px, (c + 1) * window_px)
        for r in range(nr)
        for c in range(nc)
    ]


def _write_case(
    scen_dir: Path, case_id: str, case: CaseData, source: SourceBundle
) -> dict[str, str]:
    d = scen_dir / case_id
    d.mkdir(parents=True)
    meta = {
        "case_id": case_id,
        "shape": list(case.values.shape),
        "unit": "prospectivity_index",
        "pole": source.pole,
        "source_mode": source.source_mode,
        "footprint": [list(c) for c in case.footprint],
        "grid": case.grid,
        "projection": case.projection,
        "resolution_m": case.resolution_m,
    }
    blobs = {
        "values.npy": _npy_bytes(case.values.astype(np.float64)),
        "covered.npy": _npy_bytes(case.covered.astype(np.bool_)),
        "meta.json": canonical_json(meta).encode(),
    }
    for name, b in blobs.items():
        (d / name).write_bytes(b)
    return {name: sha256_bytes(b) for name, b in blobs.items()}


def build_suite(
    source: SourceBundle,
    scenarios_root: Path,
    eval_root: Path,
    *,
    suite_seed: int = DEFAULT_SUITE_SEED,
    salt: bytes | None = None,
    suite_id: str = SUITE_ID,
    window_px: int = DEFAULT_WINDOW_PX,
    top_k: int = DEFAULT_TOP_K,
    fault_ids: tuple[str, ...] = P3_SUITE_V1_FAULT_IDS,
) -> SuiteManifest:
    """Create and freeze a suite. Refuses to overwrite an existing frozen suite.

    ``salt`` defaults to a fresh random value stored (only) in the sealed root.
    """
    import secrets

    scen_dir, seal_dir = _suite_dir(scenarios_root, suite_id), _suite_dir(eval_root, suite_id)
    if scen_dir.exists() or seal_dir.exists():
        raise SuiteAlreadyFrozen(f"suite {suite_id!r} already exists; create a new versioned suite")
    if Path(scenarios_root).resolve() == Path(eval_root).resolve():
        raise ValueError("scenario and evaluator roots must be different directories")
    salt = salt if salt is not None else secrets.token_bytes(16)
    scen_dir.mkdir(parents=True)
    seal_dir.mkdir(parents=True)

    order = list(fault_ids)
    np.random.default_rng(np.random.SeedSequence([suite_seed, 0])).shuffle(order)
    canonical_index = {f: i for i, f in enumerate(fault_ids)}

    entries: list[CaseEntry] = []
    truth: dict[str, dict[str, Any]] = {}
    for pos, fault_id in enumerate(order, start=1):
        case_id = f"case-{pos:02d}"
        seed = int(
            np.random.SeedSequence([suite_seed, 1, canonical_index[fault_id]]).generate_state(1)[0]
        )
        case, params = apply_fault(fault_id, source.data, seed)
        files = _write_case(scen_dir, case_id, case, source)
        spec = FAULTS[fault_id][0]
        entries.append(
            CaseEntry(
                case_id=case_id,
                task="fault_detection",
                modalities=[PREDICTION_MODALITY],
                budget=PUBLIC_BUDGET,
                repeat_index=0,
                source_group="g1",
                files=files,
            )
        )
        truth[case_id] = {
            "fault_id": fault_id,
            "category": spec.category,
            "is_fault": spec.is_fault,
            "severity": spec.severity,
            "intent": spec.intent,
            "expected_detectable": spec.expected_detectable,
            "detectable_rationale": spec.detectable_rationale,
            "params": params,
            "seed": seed,
        }

    # Secondary task: the UNTOUCHED prediction is ranked; the reference stays sealed.
    rank_id = "rank-01"
    entries.append(
        CaseEntry(
            case_id=rank_id,
            task="reference_disagreement_ranking",
            modalities=[PREDICTION_MODALITY],
            budget=PUBLIC_BUDGET,
            repeat_index=0,
            source_group="g1",
            files=_write_case(scen_dir, rank_id, source.data.copy(), source),
        )
    )
    windows = window_grid(source.data.values.shape, window_px)
    ref_bytes = _npy_bytes(source.reference)
    (seal_dir / "reference.npy").write_bytes(ref_bytes)
    sealed = {
        "suite_id": suite_id,
        "suite_seed": suite_seed,
        "cases": truth,
        "secondary": {
            "case_id": rank_id,
            "reference_file": "reference.npy",
            "reference_sha256": sha256_bytes(ref_bytes),
            "window_px": window_px,
            "top_k": top_k,
        },
    }
    sealed_text = canonical_json(sealed)
    (seal_dir / "sealed_key.json").write_text(sealed_text)
    (seal_dir / "salt.hex").write_text(salt.hex())
    commitment = sha256_bytes(salt + sealed_text.encode())

    manifest = SuiteManifest(
        suite_id=suite_id,
        suite_version="1",
        generator_version=GENERATOR_VERSION,
        source={
            "region_id": source.region_id,
            "source_mode": source.source_mode,
            "source_dataset": source.source_dataset,
            "source_revision": source.source_revision,
            "source_split": source.source_split,
            "prediction_sha256": source.prediction_sha256,
            "scope": "prediction-only; every case is a copy of ONE source tile",
        },
        independence={
            "independent_source_tiles": 1,
            "statistical_unit": "case",
            "warning": (
                "all cases derive from one tile and share its pixels; metrics are exploratory "
                "and do not generalize beyond this tile"
            ),
        },
        cases=entries,
        secondary_task={
            "task": "reference_disagreement_ranking",
            "case_id": rank_id,
            "window_px": window_px,
            "n_windows": len(windows),
            "windows": [list(w) for w in windows],
            "top_k": top_k,
            "target": "mean |prediction - expert reference| per window (reference sealed)",
            "interpretation": "model/expert-reference disagreement, not ice detection",
        },
        sealed_commitment=commitment,
        metrics=["precision", "recall", "f1", "false_positive_rate", "top_k_recovery"],
    )
    (scen_dir / "manifest.json").write_text(canonical_json(manifest.model_dump(mode="json")))
    (scen_dir / "manifest.sha256").write_text(manifest.body_sha256())
    return manifest


# --------------------------------------------------------------------------- loading
def load_manifest(scenarios_root: Path, suite_id: str = SUITE_ID) -> SuiteManifest:
    d = _suite_dir(scenarios_root, suite_id)
    text = (d / "manifest.json").read_text()
    manifest = SuiteManifest.model_validate_json(text)
    if manifest.body_sha256() != (d / "manifest.sha256").read_text().strip():
        raise SuiteIntegrityError("manifest content does not match manifest.sha256")
    if canonical_json(manifest.model_dump(mode="json")) != text:
        raise SuiteIntegrityError("manifest is not in canonical form")
    return manifest


@dataclass(frozen=True)
class AgentScenario:
    """The ONLY object an agent/benchmark runner may be given for a case."""

    case_id: str
    context: AgentRegionContext
    arrays: RegionArrays

    def public_dict(self) -> dict[str, Any]:
        return {"case_id": self.case_id, "context": self.context.model_dump(mode="json")}


def load_scenario(scenarios_root: Path, suite_id: str, case_id: str) -> AgentScenario:
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
    values = np.load(d / "values.npy", allow_pickle=False)
    covered = np.load(d / "covered.npy", allow_pickle=False)
    footprint = tuple(tuple(c) for c in meta["footprint"])
    file_id = f"{case_id}/values.npy"
    layer = RasterLayer(
        PREDICTION_MODALITY, values, covered, meta["unit"], file_id, entry.files["values.npy"]
    )
    arrays = RegionArrays(
        region_id=case_id,
        footprint=footprint,
        resolution_m=meta["resolution_m"],
        projection=meta["projection"],
        grid=meta["grid"],
        layers={PREDICTION_MODALITY: layer},
    )
    context = AgentRegionContext(
        region_id=case_id,
        pole=meta["pole"],
        footprint=list(footprint),
        width=values.shape[1],
        height=values.shape[0],
        spatial_resolution_m=meta["resolution_m"],
        lunar_crs=proj_string(meta["projection"]),
        source_ids=[file_id],
        source_mode=meta["source_mode"],
        available_modalities=[PREDICTION_MODALITY],
        available_tools=sorted(PREDICTION_TOOLS),
        budget=entry.budget,
    )
    return AgentScenario(case_id, context, arrays)


def load_sealed(eval_root: Path, manifest: SuiteManifest) -> SealedKey:
    """Evaluator-only. Verifies the salted commitment recorded in the frozen manifest."""
    d = _suite_dir(eval_root, manifest.suite_id)
    text = (d / "sealed_key.json").read_text()
    salt = bytes.fromhex((d / "salt.hex").read_text().strip())
    if sha256_bytes(salt + text.encode()) != manifest.sealed_commitment:
        raise SuiteIntegrityError("sealed key does not match the commitment in the manifest")
    raw = json.loads(text)
    return SealedKey(raw["suite_seed"], raw["cases"], raw["secondary"])


def load_sealed_reference(
    eval_root: Path, manifest: SuiteManifest, sealed: SealedKey
) -> np.ndarray:
    d = _suite_dir(eval_root, manifest.suite_id)
    b = (d / sealed.secondary["reference_file"]).read_bytes()
    if sha256_bytes(b) != sealed.secondary["reference_sha256"]:
        raise SuiteIntegrityError("sealed reference raster was modified")
    return np.load(io.BytesIO(b), allow_pickle=False)


def verify_suite(
    scenarios_root: Path, eval_root: Path | None = None, suite_id: str = SUITE_ID
) -> list[str]:
    """Return a list of integrity problems (empty means the frozen suite is intact)."""
    problems: list[str] = []
    try:
        manifest = load_manifest(scenarios_root, suite_id)
    except (SuiteIntegrityError, OSError, ValueError) as e:
        return [f"manifest: {e}"]
    for entry in manifest.cases:
        try:
            load_scenario(scenarios_root, suite_id, entry.case_id)
        except (SuiteIntegrityError, OSError, ValueError) as e:
            problems.append(f"{entry.case_id}: {e}")
    if eval_root is not None:
        try:
            sealed = load_sealed(eval_root, manifest)
            load_sealed_reference(eval_root, manifest, sealed)
        except (SuiteIntegrityError, OSError, ValueError) as e:
            problems.append(f"sealed: {e}")
    return problems
