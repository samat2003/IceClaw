"""Region catalog discovered from real MOONATLAS exports (nothing is hardcoded).

Discovery roots: ``Settings.moonatlas_roots`` and, below ``Settings.data_root``, any directory
(depth <= 2) holding ``ice-sectors.json`` + ``processing-manifest.json``. Every sector in each
export is loaded through the P1 adapter and validated; failures are recorded in
``RegionCatalog.rejections`` and never become catalog entries.

P4: when ``Settings.sombench_root`` is set, original SomBench observation GeoTIFFs are paired with
a prediction ONLY if every alignment check passes. Status values:

* ``PAIRED``            verified NASA-IBM prediction + aligned pixelwise TMAX/LPSR/SLOPE
* ``PREDICTION_ONLY``   prediction without (verified) observations; thermal/terrain unavailable
* ``OBSERVATION_ONLY``  verified observations, NO prediction: listed separately, never investigable
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from iceclaw.config import Settings
from iceclaw.contracts import ModalityInfo, Readiness, RegionCatalogEntry
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.data import sombench_adapter as sa
from iceclaw.data.validation import Check, ValidationResult, validate_sector, with_verified_pairing
from iceclaw.science.common import RasterLayer, RegionArrays, ScalarMetadata

PREDICTION_MODALITY = "nasa_ibm_prediction_raw"
PREDICTION_UNIT = "prospectivity_index"


class RegionNotSupported(LookupError):
    code = "OUT_OF_COVERAGE"


class PredictionUnavailable(RegionNotSupported):
    """Observations exist for this tile but no verified NASA-IBM prediction."""

    code = "PREDICTION_UNAVAILABLE"


@dataclass(frozen=True)
class Rejection:
    export_root: str
    sector_id: str | None
    code: str
    message: str


@dataclass(frozen=True)
class CatalogRecord:
    """Private record: the catalog entry plus the validated source and availability list."""

    entry: RegionCatalogEntry
    source: ma.SectorSource
    validation: ValidationResult
    availability: tuple[ma.ModalityAvailability, ...]
    observation: sa.SombenchPatch | None = None  # set only when pairing PASSED
    alignment: sa.AlignmentReport | None = None
    pairing_error: str | None = None  # observations exist but pairing failed

    @property
    def prediction_ready(self) -> bool:
        return True  # a record exists only for a validated prediction

    @property
    def observation_ready(self) -> bool:
        return self.observation is not None

    @property
    def status(self) -> str:
        return "PAIRED" if self.observation_ready else "PREDICTION_ONLY"

    @property
    def unit_unknown_layers(self) -> list[str]:
        """Downloaded layers deliberately NOT exposed because their unit is not established."""
        return self.observation.unit_unknown_layers() if self.observation else []


@dataclass(frozen=True)
class ObservationRecord:
    """Verified observations for a tile with no verified prediction (not investigable)."""

    patch: sa.SombenchPatch
    reason: str = "no verified NASA-IBM prediction exists for this tile"
    prediction_ready: bool = False
    observation_ready: bool = True
    status: str = "OBSERVATION_ONLY"


@dataclass
class RegionCatalog:
    records: dict[str, CatalogRecord] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)
    observation_only: dict[str, ObservationRecord] = field(default_factory=dict)

    def list_entries(self) -> list[RegionCatalogEntry]:
        return [self.records[k].entry for k in sorted(self.records)]

    def get(self, region_id: str) -> CatalogRecord:
        if region_id in self.observation_only:
            raise PredictionUnavailable(
                f"PREDICTION_UNAVAILABLE: {region_id!r} has verified observations but no "
                "verified NASA-IBM prediction, so it is not an investigable region."
            )
        if region_id not in self.records:
            raise RegionNotSupported(
                f"OUT_OF_COVERAGE: region {region_id!r} is not in the validated catalog "
                f"(supported: {sorted(self.records) or 'none'})."
            )
        return self.records[region_id]

    def load_arrays(self, region_id: str) -> RegionArrays:
        """Decode raw prediction (+ paired observations); checksums re-verified at read time."""
        return arrays_for_record(self.get(region_id))


# ----------------------------------------------------------------------------- building
def proj_string(projection: dict) -> str:
    """PROJ definition of the (spherical) lunar polar stereographic CRS from contract metadata."""
    if projection.get("kind") != "polar-stereographic":
        raise ma.SourceError("UNSUPPORTED_PROJECTION", f"kind={projection.get('kind')!r}")

    def g(key: str) -> str:
        return f"{projection[key]:.12g}"  # plain decimal, no exponent form

    return (
        f"+proj=stere +lat_0={g('latitudeOfOriginDeg')} +lon_0={g('centralMeridianDeg')} "
        f"+k={g('scaleFactor')} +x_0={g('falseEastingM')} "
        f"+y_0={g('falseNorthingM')} +R={g('radiusM')} +units=m +no_defs"
    )


def affine_of(grid: dict) -> tuple[float, float, float, float, float, float]:
    """Rasterio/GDAL order (a, b, c, d, e, f): x = a*col + b*row + c; y = d*col + e*row + f."""
    return (
        float(grid["pixelSizeXM"]),
        0.0,
        float(grid["xMinM"]),
        0.0,
        -float(grid["pixelSizeYM"]),
        float(grid["yMaxM"]),
    )


def checkpoint_id(src: ma.SectorSource) -> str:
    """Checkpoint filename plus the upstream-published sha256 (not verified locally)."""
    return f"{src.checkpoint}@sha256:{src.checkpoint_sha256}"


def _observation_modalities(patch: sa.SombenchPatch) -> dict[str, ModalityInfo]:
    out = {}
    for modality, layer in patch.exposed().items():
        lf, spec = patch.layers[layer], sa.LAYER_SPECS[layer]
        out[modality] = ModalityInfo(
            name=modality,
            unit=spec.unit,
            nodata=None,  # NaN nodata; coverage = finite pixels
            resolution_m=patch.grid["pixelSizeXM"],
            path=str(patch.root / lf.relpath),
            sha256=lf.sha256,
        )
    return out


def entry_from_sector(
    src: ma.SectorSource,
    validation: ValidationResult | None = None,
    observation: sa.SombenchPatch | None = None,
) -> RegionCatalogEntry:
    """Build the private ``RegionCatalogEntry`` from a validated ``SectorSource``.

    Only pixelwise layers are ``modalities``; tile scalars and single-anchor metadata are
    exposed via ``ma.describe_modalities`` and are never listed as layers. ``observation`` must
    only be passed after a PASSED alignment check. ``reference_path`` stays None: the
    evaluator-owned store is the only place it is set.
    """
    validation = validation or validate_sector(src)
    readiness, reason = validation.readiness
    modalities = {
        PREDICTION_MODALITY: ModalityInfo(
            name=PREDICTION_MODALITY,
            unit=PREDICTION_UNIT,
            nodata=None,  # coverage comes from the raster alpha mask, not a nodata value
            resolution_m=src.pixel_size_m,
            path=str(src.prediction_path),
            sha256=src.prediction_sha256,
        )
    }
    if observation is not None:
        modalities.update(_observation_modalities(observation))
    return RegionCatalogEntry(
        region_id=src.sector_id,
        pole=src.pole,  # type: ignore[arg-type]
        source_dataset=src.source_dataset,
        source_split=src.dataset_split,
        original_tile_id=src.patch_id,
        source_mode=src.source_mode,
        source_revision=src.dataset_revision,
        processing_version=src.processing_version,
        source_manifest_path=str(src.root / "ice-sectors.json"),
        source_sha256=src.sectors_json_sha256,
        lunar_crs=proj_string(src.projection),
        affine=affine_of(src.grid),
        footprint=[tuple(c) for c in src.footprint_corners_lat_lon],  # (lat, lon): NW,NE,SE,SW
        width=src.width,
        height=src.height,
        spatial_resolution_m=src.pixel_size_m,
        modalities=modalities,
        prediction_raw_path=str(src.prediction_path),
        prediction_sha256=src.prediction_sha256,
        prediction_checkpoint_id=checkpoint_id(src),
        reference_path=None,
        readiness=readiness,
        readiness_reason=reason,
    )


def observation_layers(patch: sa.SombenchPatch) -> dict[str, RasterLayer]:
    """Pixelwise layers with an ESTABLISHED unit, keyed by IceClaw modality name.

    Raw values (float64), NaN preserved, covered = finite; the file sha256 is the source id.
    Unit-unknown layers are never returned.
    """
    out = {}
    for modality, layer in patch.exposed().items():
        values, finite = sa.read_layer(patch, layer)
        lf = patch.layers[layer]
        out[modality] = RasterLayer(
            name=modality,
            values=values[0],
            covered=finite,
            unit=sa.LAYER_SPECS[layer].unit,
            file_id=f"sombench:{lf.relpath}",
            sha256=lf.sha256,
        )
    return out


def arrays_for_record(rec: CatalogRecord) -> RegionArrays:
    src = rec.source
    pred = ma.decode_prediction(src)
    layer = RasterLayer(
        name=PREDICTION_MODALITY,
        values=pred.values,
        covered=pred.covered,
        unit=PREDICTION_UNIT,
        file_id=src.prediction_relpath,
        sha256=src.prediction_sha256,
    )
    layers = {layer.name: layer}
    if rec.observation is not None:
        layers.update(observation_layers(rec.observation))
    scalars = {}
    for m in rec.availability:
        if m.status in ("tile_scalar", "single_anchor_point"):
            scalars[m.name] = ScalarMetadata(m.name, m.status, m.value, m.unit)
    return RegionArrays(
        region_id=rec.entry.region_id,
        footprint=tuple(rec.entry.footprint),
        resolution_m=rec.entry.spatial_resolution_m,
        projection=src.projection,
        grid=src.grid,
        layers=layers,
        scalars=scalars,
        source_verified=rec.validation.source_ok,
    )


def available_tool_names(arrays: RegionArrays) -> list[str]:
    """Tool names whose required pixelwise inputs exist (prediction tools + thermal/terrain)."""
    from iceclaw.science.model_audit import PREDICTION_TOOLS

    names = list(PREDICTION_TOOLS) if PREDICTION_MODALITY in arrays.layers else []
    if {"pixelwise_temperature", "pixelwise_shadow_mask"} <= set(arrays.layers):
        names.append("thermal_psr_consistency")
    if "pixelwise_terrain" in arrays.layers:
        names.append("terrain_slope_summary")
    return sorted(names)


def projection_from_patch(patch: sa.SombenchPatch) -> dict:
    """Contract-style projection dict from the GeoTIFF's own CRS (no external assumptions)."""
    import rasterio.crs

    d = rasterio.crs.CRS.from_string(patch.crs_proj4).to_dict()
    if d.get("proj") != "stere":
        raise sa.SombenchError("UNSUPPORTED_PROJECTION", patch.crs_proj4)
    return {
        "kind": "polar-stereographic",
        "latitudeOfOriginDeg": float(d["lat_0"]),
        "centralMeridianDeg": float(d["lon_0"]),
        "scaleFactor": float(d["k"]),
        "falseEastingM": float(d["x_0"]),
        "falseNorthingM": float(d["y_0"]),
        "radiusM": float(d["R"]),
    }


def observation_arrays(patch: sa.SombenchPatch) -> RegionArrays:
    """Observation-only ``RegionArrays`` (no prediction layer), geometry from the GeoTIFF."""
    proj = projection_from_patch(patch)
    w, h = patch.grid["widthPx"], patch.grid["heightPx"]
    footprint = tuple(
        ma.pixel_to_lat_lon(proj, patch.grid, c, r) for c, r in ((0, 0), (w, 0), (w, h), (0, h))
    )
    return RegionArrays(
        region_id=patch.sector_id,
        footprint=footprint,
        resolution_m=patch.grid["pixelSizeXM"],
        projection=proj,
        grid=dict(patch.grid),
        layers=observation_layers(patch),
        scalars={},
        source_verified=True,
    )


# --------------------------------------------------------------------------- discovery
def _is_export(p: Path) -> bool:
    return (p / "ice-sectors.json").is_file() and (p / "processing-manifest.json").is_file()


def find_export_roots(settings: Settings) -> list[Path]:
    found: list[Path] = []
    candidates = list(settings.moonatlas_roots)
    data_root = settings.data_root
    if data_root.is_dir():
        candidates += [data_root]
        candidates += [d for d in sorted(data_root.iterdir()) if d.is_dir()]
        candidates += [
            g
            for d in sorted(data_root.iterdir())
            if d.is_dir()
            for g in sorted(d.iterdir())
            if g.is_dir()
        ]
    for c in candidates:
        c = c.resolve()
        if c not in found and _is_export(c):
            found.append(c)
    return found


def discover(settings: Settings) -> RegionCatalog:
    """Build the catalog from real exports. Unusable sectors are rejected, not listed."""
    import json

    catalog = RegionCatalog()
    for root in find_export_roots(settings):
        try:
            sectors = json.loads((root / "ice-sectors.json").read_text(encoding="utf-8"))
            ids = [s["id"] for s in sectors]
        except (ValueError, KeyError, TypeError, OSError) as e:
            catalog.rejections.append(Rejection(str(root), None, "MALFORMED_METADATA", str(e)))
            continue
        for sid in ids:
            try:
                src = ma.load_sector(root, sid)
                validation = validate_sector(src)
                entry = entry_from_sector(src, validation)
            except ma.SourceError as e:
                catalog.rejections.append(Rejection(str(root), sid, e.code, str(e)))
                continue
            if entry.readiness is Readiness.UNUSABLE:
                catalog.rejections.append(
                    Rejection(
                        str(root), sid, "SOURCE_VALIDATION_FAILED", entry.readiness_reason or ""
                    )
                )
                continue
            if sid in catalog.records:
                catalog.rejections.append(
                    Rejection(
                        str(root), sid, "DUPLICATE_REGION_ID", "already loaded from another root"
                    )
                )
                continue
            catalog.records[sid] = CatalogRecord(
                entry, src, validation, tuple(ma.describe_modalities(src))
            )
    if settings.sombench_root is not None:
        _pair_observations(catalog, Path(settings.sombench_root))
    return catalog


def _patch_ids(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name.removesuffix("_TMAX.tif") for p in root.glob("patch_*_TMAX.tif"))


def _pair_observations(catalog: RegionCatalog, root: Path) -> None:
    """Attach observations to matching prediction records, or list them as observation-only."""
    by_patch = {r.source.patch_id: sid for sid, r in catalog.records.items()}
    for patch_id in _patch_ids(root):
        try:
            patch = sa.load_patch(root, patch_id)
        except sa.SombenchError as e:
            catalog.rejections.append(Rejection(str(root), patch_id, e.code, str(e)))
            continue
        sid = by_patch.get(patch_id)
        if sid is None:
            catalog.observation_only[patch.sector_id] = ObservationRecord(patch)
            continue
        rec = catalog.records[sid]
        try:
            pred = ma.decode_prediction(rec.source).values
            report = sa.check_alignment(patch, rec.source, pred)
        except (ma.SourceError, sa.SombenchError) as e:
            catalog.records[sid] = dataclasses.replace(rec, pairing_error=str(e))
            catalog.rejections.append(Rejection(str(root), patch_id, "PAIRING_FAILED", str(e)))
            continue
        if not report.ok:  # never pair on a failed check; the prediction stays valid alone
            msg = "; ".join(f"{c.name}: {c.detail}" for c in report.failures())
            catalog.records[sid] = dataclasses.replace(
                rec, alignment=report, pairing_error=f"ALIGNMENT_FAILED: {msg}"
            )
            catalog.rejections.append(Rejection(str(root), patch_id, "PAIRING_FAILED", msg))
            continue
        exposed = patch.exposed()
        validation = with_verified_pairing(
            rec.validation,
            [Check(c.name, c.passed, c.detail) for c in report.checks],
            sorted(exposed),
        )
        availability = tuple(
            a for a in rec.availability if not (a.status == "unavailable" and a.name in exposed)
        ) + tuple(
            ma.ModalityAvailability(
                name,
                "pixelwise",
                unit=sa.LAYER_SPECS[layer].unit,
                note=f"SomBench {layer}: {sa.LAYER_SPECS[layer].unit_basis}",
            )
            for name, layer in sorted(exposed.items())
        )
        catalog.records[sid] = dataclasses.replace(
            rec,
            entry=entry_from_sector(rec.source, validation, observation=patch),
            validation=validation,
            availability=availability,
            observation=patch,
            alignment=report,
        )
