"""Content-addressed SOURCE PACKAGE: the minimal permitted inputs needed to recompute a bundle.

A package holds the exact original bytes of the non-evaluator arrays the deterministic tools read
(the NASA-IBM prediction raster and the SomBench TMAX/LPSR/SLOPE GeoTIFFs), plus the metadata
needed to decode them without any catalog, upstream checkout or network: grid, projection,
footprint, decoding parameters, units and per-file SHA256. Its identity is
``package_sha256`` (SHA256 of the canonical manifest), recorded in the bundle's input manifest.

Deliberately excluded (policy, tested): the expert target ``PRO``; the reference raster; the
upstream ``ice-sectors.json`` (it carries statistics of the evaluator-only reference, so only a
filtered excerpt is written); fault labels; ``.env``; absolute paths.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iceclaw.contracts import AgentRegionContext, RunBudget
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.science.common import RasterLayer, RegionArrays, ScalarMetadata

FORMAT = "iceclaw-source-package/1"
MANIFEST_NAME = "package_manifest.json"
PREDICTION = "nasa_ibm_prediction_raw"
KIND_PREDICTION = "nasa_ibm_model_prediction"
KIND_OBSERVATION = "lunar_observation"
FORBIDDEN_NAMES = ("PRO", "reference", "sealed", "ice-sectors.json", ".env", "fault")


class PackageError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def contained_path(root: Path, name: str) -> Path:
    """Resolve an untrusted relative path without allowing escape through symlinks."""
    if not isinstance(name, str) or not name or "\\" in name:
        raise PackageError("INVALID_PATH", "expected a nonempty relative POSIX path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise PackageError("INVALID_PATH", f"not a package-relative path: {name!r}")
    base = Path(root).resolve()
    try:
        path = (base / relative).resolve()
    except (OSError, RuntimeError) as e:
        raise PackageError("INVALID_PATH", f"cannot resolve {name!r}") from e
    if not path.is_relative_to(base) or path == base:
        raise PackageError("INVALID_PATH", f"path escapes package: {name!r}")
    return path


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@dataclass(frozen=True)
class SourceFile:
    modality: str
    kind: str  # KIND_PREDICTION | KIND_OBSERVATION
    path: Path  # original file (read only while writing a package)
    file_id: str  # identifier the science tools cite
    unit: str
    decoding: dict[str, Any]
    origin: str  # human-readable provenance


@dataclass
class SourceSpec:
    """Everything needed to write a package and to build an ``AgentRegionContext``."""

    region_id: str
    pole: str
    source_mode: str
    source_dataset: str
    source_split: str
    source_revision: str
    processing_version: str
    original_tile_id: str
    prediction_checkpoint_id: str
    upstream_sectors_sha256: str  # sha of the upstream file that is NOT included (see module doc)
    lunar_crs: str
    projection: dict[str, Any]
    grid: dict[str, Any]
    footprint: list[tuple[float, float]]  # (lat, lon) NW, NE, SE, SW
    width: int
    height: int
    resolution_m: float
    files: list[SourceFile]
    scalars: list[dict[str, Any]] = field(default_factory=list)
    upstream: dict[str, Any] = field(default_factory=dict)
    attribution: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------ writing
def _file_name(sf: SourceFile) -> str:
    return sf.path.name


def build_manifest(spec: SourceSpec, shas: dict[str, tuple[str, int]]) -> dict[str, Any]:
    files = []
    for sf in sorted(spec.files, key=lambda f: f.modality):
        sha, size = shas[sf.modality]
        files.append(
            {
                "modality": sf.modality,
                "kind": sf.kind,
                "file": f"files/{_file_name(sf)}",
                "file_id": sf.file_id,
                "sha256": sha,
                "size": size,
                "unit": sf.unit,
                "decoding": sf.decoding,
                "origin": sf.origin,
            }
        )
    body = {
        "format": FORMAT,
        "region": {
            "region_id": spec.region_id,
            "pole": spec.pole,
            "source_mode": spec.source_mode,
            "source_dataset": spec.source_dataset,
            "source_split": spec.source_split,
            "source_revision": spec.source_revision,
            "processing_version": spec.processing_version,
            "original_tile_id": spec.original_tile_id,
            "prediction_checkpoint_id": spec.prediction_checkpoint_id,
            "upstream_sectors_sha256": spec.upstream_sectors_sha256,
            "upstream_sectors_note": (
                "NOT included: the upstream file carries statistics of the evaluator-only "
                "reference raster. The hash is recorded only so it can be checked against "
                "the upstream repository."
            ),
        },
        "geometry": {
            "lunar_crs": spec.lunar_crs,
            "projection": spec.projection,
            "grid": spec.grid,
            "footprint_lat_lon_NW_NE_SE_SW": [list(c) for c in spec.footprint],
            "width": spec.width,
            "height": spec.height,
            "resolution_m": spec.resolution_m,
        },
        "files": files,
        "tile_scalars_and_anchors_not_arrays": spec.scalars,
        "upstream": spec.upstream,
        "attribution_and_modification_notice": spec.attribution,
        "excluded_by_policy": [
            "the evaluator-only expert target layer and any reference raster",
            "fault labels / sealed evaluation data",
            "layers with unestablished units (CUR, DICE, LPSR_DEN, LPSR_DIS, ASP_SIN_COS)",
            ".env and any credential",
        ],
    }
    return {**body, "package_sha256": sha256_bytes(canonical(body).encode())}


def write_package(spec: SourceSpec, out_dir: Path) -> dict[str, Any]:
    """Copy the exact original bytes + write the manifest. Returns the manifest."""
    out_dir = Path(out_dir)
    (out_dir / "files").mkdir(parents=True, exist_ok=True)
    names = [_file_name(f) for f in spec.files]
    if len(set(names)) != len(names):
        raise PackageError("DUPLICATE_FILE_NAME", str(names))
    for f in spec.files:
        low = f.path.name.lower()
        if any(tok.lower() in low for tok in ("_pro.", "reference", "sealed", ".env")):
            raise PackageError("FORBIDDEN_SOURCE", f"{f.path.name} must never be packaged")
    shas: dict[str, tuple[str, int]] = {}
    for f in spec.files:
        data = f.path.read_bytes()
        (out_dir / "files" / _file_name(f)).write_bytes(data)
        shas[f.modality] = (sha256_bytes(data), len(data))
    manifest = build_manifest(spec, shas)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


# ------------------------------------------------------------------------------ loading
@dataclass
class LoadedPackage:
    manifest: dict[str, Any]
    arrays: RegionArrays
    root: Path


def read_manifest(root: Path) -> dict[str, Any]:
    p = contained_path(root, MANIFEST_NAME)
    if not p.is_file():
        raise PackageError("MISSING_FILE", f"{MANIFEST_NAME} not found in {root}")
    try:
        m = json.loads(p.read_text())
    except (ValueError, OSError) as e:
        raise PackageError("MANIFEST_TAMPERED", f"manifest is not valid JSON: {e}") from e
    if not isinstance(m, dict):
        raise PackageError("INVALID_MANIFEST", "manifest must be a JSON object")
    if m.get("format") != FORMAT:
        raise PackageError("UNSUPPORTED_FORMAT", str(m.get("format")))
    body = {k: v for k, v in m.items() if k != "package_sha256"}
    if sha256_bytes(canonical(body).encode()) != m.get("package_sha256"):
        raise PackageError("MANIFEST_TAMPERED", "package_sha256 does not match manifest content")
    return m


def _decode_prediction(path: Path, dec: dict[str, Any], shape: tuple[int, int]):
    from PIL import Image

    try:
        with Image.open(path) as im:
            rgba = np.asarray(im.convert("RGBA") if im.mode != "RGB" else im.convert("RGB"))
    except Exception as e:
        raise PackageError("CORRUPT_RASTER", f"{path.name}: {e}") from e
    try:
        d = ma.decode_rgba(rgba, {"offset": dec["offset"], "scale": dec["scale"]}, shape)
    except ma.SourceError as e:
        raise PackageError("CORRUPT_RASTER", str(e)) from e
    return d.values, d.covered


def _read_geotiff(path: Path):
    import rasterio

    try:
        with rasterio.open(path) as ds:
            data = ds.read().astype(np.float64)
    except Exception as e:
        raise PackageError("CORRUPT_RASTER", f"{path.name}: {e}") from e
    return data[0], np.isfinite(data).all(axis=0)


def load_package(root: Path) -> LoadedPackage:
    """Verify every hash, decode with the same code path as the catalog, build ``RegionArrays``."""
    try:
        return _load_package(Path(root))
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OSError) as e:
        raise PackageError("INVALID_MANIFEST", f"invalid source package: {e}") from e


def _load_package(root: Path) -> LoadedPackage:
    root = Path(root)
    m = read_manifest(root)
    geo = m["geometry"]
    shape = (geo["height"], geo["width"])
    layers: dict[str, RasterLayer] = {}
    seen_files: set[Path] = set()
    seen_ids: set[str] = set()
    for f in m["files"]:
        p = contained_path(root, f["file"])
        if p in seen_files or f["file_id"] in seen_ids or f["modality"] in layers:
            raise PackageError("INVALID_MANIFEST", "duplicate source file, file ID or modality")
        seen_files.add(p)
        seen_ids.add(f["file_id"])
        if f["kind"] not in (KIND_PREDICTION, KIND_OBSERVATION):
            raise PackageError("INVALID_MANIFEST", f"unknown source kind: {f['kind']!r}")
        if not p.is_file():
            raise PackageError("MISSING_FILE", f["file"])
        data = p.read_bytes()
        if len(data) != f["size"] or sha256_bytes(data) != f["sha256"]:
            raise PackageError("CHECKSUM_MISMATCH", f"{f['file']} differs from package manifest")
        if f["kind"] == KIND_PREDICTION:
            values, covered = _decode_prediction(p, f["decoding"], shape)
        else:
            values, covered = _read_geotiff(p)
        if values.shape != shape:
            raise PackageError("SHAPE_MISMATCH", f"{f['file']}: {values.shape} vs {shape}")
        layers[f["modality"]] = RasterLayer(
            f["modality"], values, covered, f["unit"], f["file_id"], f["sha256"]
        )
    scalars = {
        s["name"]: ScalarMetadata(s["name"], s["kind"], s["value"], s["unit"])
        for s in m["tile_scalars_and_anchors_not_arrays"]
    }
    arrays = RegionArrays(
        region_id=m["region"]["region_id"],
        footprint=tuple(tuple(c) for c in geo["footprint_lat_lon_NW_NE_SE_SW"]),
        resolution_m=geo["resolution_m"],
        projection=geo["projection"],
        grid=geo["grid"],
        layers=layers,
        scalars=scalars,
        source_verified=True,
    )
    return LoadedPackage(m, arrays, root)


def copy_package(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst)


# ------------------------------------------------------------ catalog -> spec / context
def spec_from_catalog_record(rec: Any) -> SourceSpec:
    """Build a ``SourceSpec`` from a PAIRED catalog record (real data)."""
    from iceclaw.data import sombench_adapter as sa
    from iceclaw.data.catalog import checkpoint_id, proj_string

    if rec.observation is None:
        raise PackageError("NOT_PAIRED", f"{rec.entry.region_id} has no verified observations")
    src, patch = rec.source, rec.observation
    files = [
        SourceFile(
            PREDICTION,
            KIND_PREDICTION,
            src.prediction_path,
            src.prediction_relpath,
            "prospectivity_index",
            {
                "format": "u16-rg-linear-webp",
                "offset": src.encoding["offset"],
                "scale": src.encoding["scale"],
                "alpha_is_coverage": True,
            },
            "NASA-IBM Lunar Foundation Model ice-prospectivity output as published in the "
            "MOONATLAS science-smoke example (not a fresh IceClaw inference)",
        )
    ]
    for modality, layer in sorted(patch.exposed().items()):
        lf = patch.layers[layer]
        files.append(
            SourceFile(
                modality,
                KIND_OBSERVATION,
                patch.root / lf.relpath,
                f"sombench:{lf.relpath}",
                sa.LAYER_SPECS[layer].unit or "",
                {"format": "geotiff-float32-band1", "nodata": "nan", "somb_layer": layer},
                f"SomBench Ice-Prospectivity-Regression {layer}, unit basis: "
                f"{sa.LAYER_SPECS[layer].unit_basis}",
            )
        )
    scalars = [
        {"name": a.name, "kind": a.status, "value": a.value, "unit": a.unit}
        for a in rec.availability
        if a.status in ("tile_scalar", "single_anchor_point")
    ]
    dl = json.loads((patch.root / "download_manifest.json").read_text())["files"]
    wanted = {patch.layers[layer].relpath for layer in patch.exposed().values()}
    return SourceSpec(
        region_id=rec.entry.region_id,
        pole=rec.entry.pole,
        source_mode=rec.entry.source_mode.value,
        source_dataset=rec.entry.source_dataset,
        source_split=rec.entry.source_split,
        source_revision=rec.entry.source_revision,
        processing_version=rec.entry.processing_version,
        original_tile_id=rec.entry.original_tile_id,
        prediction_checkpoint_id=checkpoint_id(src),
        upstream_sectors_sha256=src.sectors_json_sha256,
        lunar_crs=proj_string(src.projection),
        projection=dict(src.projection),
        grid=dict(src.grid),
        footprint=[tuple(c) for c in rec.entry.footprint],
        width=rec.entry.width,
        height=rec.entry.height,
        resolution_m=rec.entry.spatial_resolution_m,
        files=files,
        scalars=scalars,
        upstream={
            "moonatlas_commit": "945fb5196679cf6927354c53be624d745083f9e9",
            "moonatlas_upstream_nasa_ibm_code_commit": src.upstream_commit,
            "model_revision": src.model_revision,
            "checkpoint_file": src.checkpoint,
            "checkpoint_sha256_as_published_upstream_not_verified_locally": src.checkpoint_sha256,
            "prediction_upstream_statistics": src.upstream_stats,
            "sombench_dataset_revision": patch.dataset_revision,
            "sombench_download_records": {k: dl[k] for k in sorted(wanted)},
        },
        attribution=[
            "NASA-IBM Lunar Foundation Model, ice-prospectivity task checkpoint "
            "(NASA-IBM AI4Science; Apache-2.0). Prediction raster is the model output published "
            "in the MOONATLAS science-smoke example, re-encoded losslessly (16-bit linear WebP) "
            "and georeferenced by MOONATLAS; not a fresh IceClaw inference. No NASA/IBM "
            "endorsement of IceClaw.",
            "SomBench Ice-Prospectivity-Regression (nasa-ibm-ai4science), CC BY 4.0, revision "
            f"{patch.dataset_revision}. GeoTIFFs are byte-exact copies of TMAX, LPSR and SLOPE; "
            "no modification.",
            "Fraccaro, P. et al. (2026). Multimodal-Multiresolution Foundation Model for Lunar "
            "Remote Sensing. NASA-IBM AI4Science.",
        ],
    )


def context_from_spec(
    spec: SourceSpec, arrays: RegionArrays, budget: RunBudget
) -> AgentRegionContext:
    """The agent-visible context, built from the package-loaded arrays (allowlist only)."""
    from iceclaw.contracts import SourceMode
    from iceclaw.data.catalog import available_tool_names

    return AgentRegionContext(
        region_id=spec.region_id,
        pole=spec.pole,  # type: ignore[arg-type]
        footprint=list(spec.footprint),
        width=spec.width,
        height=spec.height,
        spatial_resolution_m=spec.resolution_m,
        lunar_crs=spec.lunar_crs,
        source_ids=[spec.source_dataset, spec.original_tile_id, spec.prediction_checkpoint_id],
        source_mode=SourceMode(spec.source_mode),
        available_modalities=sorted(arrays.layers),
        available_tools=sorted(available_tool_names(arrays)),
        budget=budget,
    )
