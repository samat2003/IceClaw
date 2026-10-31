"""Region catalog discovered from real MOONATLAS exports (nothing is hardcoded).

Discovery roots: ``Settings.moonatlas_roots`` and, below ``Settings.data_root``, any directory
(depth <= 2) holding ``ice-sectors.json`` + ``processing-manifest.json``. Every sector in each
export is loaded through the P1 adapter and validated; failures are recorded in
``RegionCatalog.rejections`` and never become catalog entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from iceclaw.config import Settings
from iceclaw.contracts import ModalityInfo, Readiness, RegionCatalogEntry
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.data.validation import ValidationResult, validate_sector
from iceclaw.science.common import RasterLayer, RegionArrays, ScalarMetadata

PREDICTION_MODALITY = "nasa_ibm_prediction_raw"
PREDICTION_UNIT = "prospectivity_index"


class RegionNotSupported(LookupError):
    code = "OUT_OF_COVERAGE"


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


@dataclass
class RegionCatalog:
    records: dict[str, CatalogRecord] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)

    def list_entries(self) -> list[RegionCatalogEntry]:
        return [self.records[k].entry for k in sorted(self.records)]

    def get(self, region_id: str) -> CatalogRecord:
        if region_id not in self.records:
            raise RegionNotSupported(
                f"OUT_OF_COVERAGE: region {region_id!r} is not in the validated catalog "
                f"(supported: {sorted(self.records) or 'none'})."
            )
        return self.records[region_id]

    def load_arrays(self, region_id: str) -> RegionArrays:
        """Decode the raw prediction (checksum re-verified) into the science-tool input."""
        rec = self.get(region_id)
        return arrays_for_record(rec)


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


def entry_from_sector(
    src: ma.SectorSource, validation: ValidationResult | None = None
) -> RegionCatalogEntry:
    """Build the private ``RegionCatalogEntry`` from a validated ``SectorSource``.

    Only pixelwise layers are ``modalities``; tile scalars and single-anchor metadata are
    exposed via ``ma.describe_modalities`` and are never listed as layers.
    ``reference_path`` stays None: the evaluator-owned store (P3) is the only place it is set.
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
        layers={layer.name: layer},
        scalars=scalars,
        source_verified=rec.validation.source_ok,
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
    return catalog
