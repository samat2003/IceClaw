"""Read-only adapter for a MOONATLAS ``science-smoke`` style export (data contract v2).

P1 scope: load one ice-prospectivity sector, verify checksums/provenance/geometry,
decode the *raw* numeric prediction (never the display heatmap), summarize it, and state
exactly which scientific measurements exist. The reference raster is evaluator-only: this
module never reads, decodes, or exposes it.

Source mode of anything loaded here is ``upstream_published_example`` -- it is NOT a fresh
IceClaw/NASA checkpoint inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iceclaw.contracts import SourceMode

DEFAULT_SECTOR_ID = "ice-s-0001-0002"
SUPPORTED_ENCODING = "u16-rg-linear"
SUPPORTED_SCHEMA = "2"
GEOLOCATION_TOLERANCE_DEG = 1e-5  # upstream stores coordinates with 6 decimals
STATS_ROUNDING = 1e-6  # upstream rounds summary statistics to 6 decimals


class SourceError(RuntimeError):
    """Source validation failure with a machine-readable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


# ----------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class SectorSource:
    sector_id: str
    pole: str
    patch_id: str
    source_mode: SourceMode
    source_dataset: str
    dataset_revision: str
    dataset_split: str
    task: str
    checkpoint: str
    checkpoint_sha256: str
    model_revision: str
    upstream_commit: str
    processing_version: str
    catalog_version: str
    projection: dict[str, Any]
    grid: dict[str, Any]
    width: int
    height: int
    pixel_size_m: float
    nominal_resolution_m: float
    center_lat_lon: tuple[float, float]
    footprint_corners_lat_lon: tuple[tuple[float, float], ...]  # NW, NE, SE, SW
    psr_fraction: float | None
    at_anchor: dict[str, Any]
    prediction_path: Path
    prediction_relpath: str
    prediction_sha256: str
    encoding: dict[str, Any]
    upstream_stats: dict[str, float]
    covered_pixels: int
    clipped_pixels: int
    raw_peak: dict[str, Any]
    sectors_json_sha256: str
    root: Path

    @property
    def quantization_tolerance(self) -> float:
        """Max |decoded - original model output| (scale/2), the encoding's own error bound."""
        return float(self.encoding["scale"]) / 2

    @property
    def stats_tolerance(self) -> float:
        """Documented tolerance for comparing recomputed vs upstream summary statistics:
        half a quantization step plus upstream's 6-decimal rounding."""
        return self.quantization_tolerance + STATS_ROUNDING


@dataclass(frozen=True)
class DecodedPrediction:
    values: np.ndarray  # float64 (H, W), raw regression output, NOT clipped to [0, 1]
    covered: np.ndarray  # bool (H, W)


@dataclass(frozen=True)
class ModalityAvailability:
    name: str
    status: str  # "pixelwise" | "tile_scalar" | "single_anchor_point" | "unavailable"
    unit: str | None = None
    value: Any = None
    note: str = ""


@dataclass(frozen=True)
class Discrepancy:
    quantity: str
    recomputed: float
    upstream: float
    abs_diff: float
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.abs_diff <= self.tolerance


@dataclass
class ComparisonReport:
    items: list[Discrepancy] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(i.ok for i in self.items)


# ----------------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise SourceError("MISSING_FILE", f"required file not found: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SourceError("MALFORMED_METADATA", f"{path.name} is not valid JSON: {e}") from e


def _need(d: dict, *keys: str, where: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            raise SourceError("MALFORMED_METADATA", f"{where}: missing '{'.'.join(keys)}'")
        cur = cur[k]
    return cur


def verify_checksum(root: Path, relpath: str, expected: str) -> str:
    """Return actual SHA256; raise MISSING_FILE / CHECKSUM_MISMATCH."""
    path = root / relpath
    if not path.is_file():
        raise SourceError("MISSING_FILE", f"required file not found: {relpath}")
    actual = sha256_file(path)
    if actual != expected:
        raise SourceError(
            "CHECKSUM_MISMATCH",
            f"{relpath}: sha256 {actual[:12]}... != expected {expected[:12]}...",
        )
    return actual


# --------------------------------------------------------------- projection (spherical)
_DEG = math.pi / 180.0


def pixel_to_lat_lon(projection: dict, grid: dict, col: float, row: float) -> tuple[float, float]:
    """Polar-stereographic (sphere) inverse for a fractional pixel position.

    Independent re-implementation of the contract's formula; ``col+0.5,row+0.5`` is a pixel centre.
    """
    if projection.get("kind") != "polar-stereographic":
        raise SourceError("UNSUPPORTED_PROJECTION", f"kind={projection.get('kind')!r}")
    x = grid["xMinM"] + col * grid["pixelSizeXM"]
    y = grid["yMaxM"] - row * grid["pixelSizeYM"]
    rk = projection["radiusM"] * projection["scaleFactor"]
    dx, dy = x - projection["falseEastingM"], y - projection["falseNorthingM"]
    south = projection["latitudeOfOriginDeg"] < 0
    c = 2 * math.atan(math.hypot(dx, dy) / (2 * rk))
    lat = (c - math.pi / 2) if south else (math.pi / 2 - c)
    lon = projection["centralMeridianDeg"] + math.atan2(dx, dy if south else -dy) / _DEG
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat / _DEG, lon


def lat_lon_to_pixel(projection: dict, grid: dict, lat: float, lon: float) -> tuple[float, float]:
    """Forward projection to fractional (col, row); used for round-trip geometry checks."""
    lam = (lon - projection["centralMeridianDeg"]) * _DEG
    phi = lat * _DEG
    south = projection["latitudeOfOriginDeg"] < 0
    rk = projection["radiusM"] * projection["scaleFactor"]
    rho = 2 * rk * math.tan(math.pi / 4 + (phi if south else -phi) / 2)
    x = projection["falseEastingM"] + rho * math.sin(lam)
    y = projection["falseNorthingM"] + (1 if south else -1) * rho * math.cos(lam)
    return (x - grid["xMinM"]) / grid["pixelSizeXM"], (grid["yMaxM"] - y) / grid["pixelSizeYM"]


def lon_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ------------------------------------------------------------------------------ loading
def load_sector(root: str | Path, sector_id: str = DEFAULT_SECTOR_ID) -> SectorSource:
    """Parse and validate one ice sector from a MOONATLAS export root.

    Verifies the processing-manifest checksum of ``ice-sectors.json``, ``manifest.json`` and the
    raw prediction raster, plus provenance and geometry consistency. Raises ``SourceError``.
    """
    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    processing = _read_json(root / "processing-manifest.json")
    if str(manifest.get("schemaVersion")) != SUPPORTED_SCHEMA:
        raise SourceError("UNSUPPORTED_SCHEMA", f"schemaVersion={manifest.get('schemaVersion')!r}")
    if manifest.get("dataMode") != "real-smoke" or processing.get("scope") != "smoke":
        raise SourceError(
            "UNSUPPORTED_SOURCE_MODE",
            "P1 only supports the published real-smoke export "
            f"(dataMode={manifest.get('dataMode')!r}, scope={processing.get('scope')!r}).",
        )
    checksums = _need(processing, "checksums", where="processing-manifest.json")
    for rel in ("manifest.json", "ice-sectors.json"):
        if rel not in checksums:
            raise SourceError("MALFORMED_METADATA", f"no checksum recorded for {rel}")
    verify_checksum(root, "manifest.json", checksums["manifest.json"])
    sectors_sha = verify_checksum(root, "ice-sectors.json", checksums["ice-sectors.json"])

    sectors = _read_json(root / "ice-sectors.json")
    sector = next((s for s in sectors if s.get("id") == sector_id), None)
    if sector is None:
        raise SourceError("NOT_FOUND", f"sector {sector_id!r} not in ice-sectors.json")

    w = f"sector {sector_id}"
    src = _need(sector, "source", where=w)
    pred = _need(sector, "prediction", where=w)
    prov = _need(sector, "provenance", where=w)
    grid = _need(src, "grid", where=w)
    projection = _need(src, "projection", where=w)
    heat = _need(pred, "heatmap", where=w)
    enc = _need(heat, "rawEncoding", where=w)
    rel = _need(heat, "rawPath", where=w)
    if enc.get("kind") != SUPPORTED_ENCODING:
        raise SourceError("UNSUPPORTED_ENCODING", f"rawEncoding.kind={enc.get('kind')!r}")
    if not (isinstance(enc.get("scale"), (int, float)) and enc["scale"] > 0):
        raise SourceError("MALFORMED_METADATA", "rawEncoding.scale must be positive")
    if rel not in checksums:
        raise SourceError("MALFORMED_METADATA", f"no checksum recorded for {rel}")
    width, height = int(grid["widthPx"]), int(grid["heightPx"])
    if width <= 0 or height <= 0:
        raise SourceError("MALFORMED_METADATA", "grid dimensions must be positive")
    if grid["pixelSizeXM"] <= 0 or grid["pixelSizeYM"] <= 0:
        raise SourceError("MALFORMED_METADATA", "pixel size must be positive")

    pred_sha = verify_checksum(root, rel, checksums[rel])

    # Provenance cross-checks against the processing manifest.
    model = next(
        (m for m in processing.get("models", []) if m.get("sourceId") == "lfm-ice-prospectivity"),
        None,
    )
    if model is None:
        raise SourceError("MALFORMED_METADATA", "no lfm-ice-prospectivity model in manifest")
    if model.get("checkpoint") != prov.get("checkpoint") or model.get("revision") != prov.get(
        "modelRevision"
    ):
        raise SourceError(
            "PROVENANCE_MISMATCH", "sector checkpoint/revision != processing manifest"
        )
    if prov.get("processingVersion") != processing.get("processingVersion"):
        raise SourceError("PROVENANCE_MISMATCH", "processingVersion differs from manifest")
    dataset = next(
        (
            d
            for d in processing.get("datasets", [])
            if d.get("sourceId") == prov.get("sourceDataset")
        ),
        None,
    )
    if dataset is None:
        raise SourceError("PROVENANCE_MISMATCH", "sourceDataset not in processing manifest")

    fg = _need(src, "footprintGrid", where=w)
    try:
        corners = tuple(
            (float(fg[r][c]["latitude"]), float(fg[r][c]["longitude"]))
            for r, c in (
                (0, 0),
                (0, len(fg[0]) - 1),
                (len(fg) - 1, len(fg[0]) - 1),
                (len(fg) - 1, 0),
            )
        )
    except (KeyError, IndexError, TypeError) as e:
        raise SourceError("MALFORMED_METADATA", f"footprintGrid unreadable: {e}") from e

    display = _need(pred, "display", where=w)
    return SectorSource(
        sector_id=sector_id,
        pole=sector["pole"],
        patch_id=src["patchId"],
        source_mode=SourceMode.UPSTREAM_PUBLISHED_EXAMPLE,
        source_dataset=prov["sourceDataset"],
        dataset_revision=dataset["revision"],
        dataset_split=prov["datasetSplit"],
        task=prov["task"],
        checkpoint=prov["checkpoint"],
        checkpoint_sha256=model["checkpointSha256"],
        model_revision=prov["modelRevision"],
        upstream_commit=_need(processing, "upstream", "commit", where="processing-manifest.json"),
        processing_version=prov["processingVersion"],
        catalog_version=str(manifest.get("catalogVersion")),
        projection=projection,
        grid=grid,
        width=width,
        height=height,
        pixel_size_m=float(grid["pixelSizeXM"]),
        nominal_resolution_m=float(src["resolutionM"]),
        center_lat_lon=(float(sector["latitude"]), float(sector["longitude"])),
        footprint_corners_lat_lon=corners,
        psr_fraction=src.get("psrFraction"),
        at_anchor=dict(src.get("atAnchor") or {}),
        prediction_path=root / rel,
        prediction_relpath=rel,
        prediction_sha256=pred_sha,
        encoding=dict(enc),
        upstream_stats={k: float(v) for k, v in _need(pred, "raw", where=w).items()},
        covered_pixels=int(display["coveredPixels"]),
        clipped_pixels=int(display["clippedPixels"]),
        raw_peak=dict(_need(pred, "rawPeak", where=w)),
        sectors_json_sha256=sectors_sha,
        root=root,
    )


def decode_prediction(src: SectorSource) -> DecodedPrediction:
    """Decode the lossless raw raster: ``value = offset + scale * (R*256 + G)``; alpha = coverage.

    Re-verifies the file checksum at read time so a swapped file is caught.
    """
    actual = sha256_file(src.prediction_path) if src.prediction_path.is_file() else None
    if actual is None:
        raise SourceError("MISSING_FILE", f"required file not found: {src.prediction_relpath}")
    if actual != src.prediction_sha256:
        raise SourceError("CHECKSUM_MISMATCH", f"{src.prediction_relpath} changed since load")
    try:
        with Image.open(src.prediction_path) as im:
            if im.format != "WEBP":
                raise SourceError("CORRUPT_RASTER", f"expected WEBP, got {im.format}")
            rgba = np.asarray(im.convert("RGBA") if im.mode != "RGB" else im.convert("RGB"))
    except SourceError:
        raise
    except Exception as e:  # PIL raises many types for corrupt data
        raise SourceError("CORRUPT_RASTER", f"cannot decode {src.prediction_relpath}: {e}") from e
    return decode_rgba(rgba, src.encoding, expected_shape=(src.height, src.width))


def decode_rgba(
    rgba: np.ndarray, encoding: dict, expected_shape: tuple[int, int] | None = None
) -> DecodedPrediction:
    """Pure decoder for an (H, W, 3|4) uint8 array; libwebp drops an all-opaque alpha channel."""
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4) or rgba.dtype != np.uint8:
        raise SourceError("CORRUPT_RASTER", f"unexpected raster array {rgba.shape} {rgba.dtype}")
    if expected_shape is not None and rgba.shape[:2] != expected_shape:
        raise SourceError(
            "SHAPE_MISMATCH", f"raster is {rgba.shape[:2]}, metadata says {expected_shape}"
        )
    levels = rgba[..., 0].astype(np.float64) * 256 + rgba[..., 1]
    covered = rgba[..., 3] == 255 if rgba.shape[2] == 4 else np.ones(rgba.shape[:2], dtype=bool)
    values = float(encoding["offset"]) + float(encoding["scale"]) * levels
    return DecodedPrediction(values=values, covered=covered)


def summarize_prediction(pred: DecodedPrediction) -> dict[str, float | int]:
    """Deterministic summary over covered pixels. Values outside [0, 1] are kept, not clipped."""
    v = pred.values[pred.covered]
    if v.size == 0:
        raise SourceError("NO_VALID_PIXELS", "prediction has no covered pixels")
    return {
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "p90": float(np.percentile(v, 90)),
        "valid_pixel_count": int(v.size),
        "outside_unit_interval_count": int(((v < 0) | (v > 1)).sum()),
    }


def compare_to_upstream(src: SectorSource, summary: dict[str, float | int]) -> ComparisonReport:
    """Compare a recomputed summary with upstream's published values.

    Tolerance for min/max/mean/p90 is ``scale/2 + 1e-6`` (see ``SectorSource.stats_tolerance``).
    Pixel counts must match exactly except ``outside_unit_interval_count``, which upstream
    computed on the unquantized model output and may differ by quantization at the 0/1 edges;
    it is reported as a count difference, not hidden.
    """
    rep = ComparisonReport()
    tol = src.stats_tolerance
    for k in ("min", "max", "mean", "p90"):
        rep.items.append(
            Discrepancy(
                k,
                float(summary[k]),
                src.upstream_stats[k],
                abs(summary[k] - src.upstream_stats[k]),
                tol,
            )
        )
    rep.items.append(
        Discrepancy(
            "valid_pixel_count",
            float(summary["valid_pixel_count"]),
            float(src.covered_pixels),
            abs(summary["valid_pixel_count"] - src.covered_pixels),
            0.0,
        )
    )
    return rep


def verify_peak_and_geometry(src: SectorSource, pred: DecodedPrediction) -> list[str]:
    """Return a list of failure messages (empty = pass) for peak location/value and geolocation."""
    problems: list[str] = []
    p = src.raw_peak
    col, row = int(p["col"]), int(p["row"])
    if not (0 <= col < src.width and 0 <= row < src.height):
        return [f"rawPeak ({row},{col}) outside grid"]
    got = float(pred.values[row, col])
    if abs(got - float(p["value"])) > src.stats_tolerance:
        problems.append(f"value at rawPeak {got:.6f} != {p['value']}")
    masked = np.where(pred.covered, pred.values, -np.inf)
    arg_row, arg_col = np.unravel_index(int(np.argmax(masked)), masked.shape)
    if (int(arg_row), int(arg_col)) != (row, col):
        problems.append(f"argmax ({arg_row},{arg_col}) != rawPeak ({row},{col})")
    lat, lon = pixel_to_lat_lon(src.projection, src.grid, col + 0.5, row + 0.5)
    if (
        abs(lat - p["latitude"]) > GEOLOCATION_TOLERANCE_DEG
        or lon_diff(lon, p["longitude"]) > GEOLOCATION_TOLERANCE_DEG
    ):
        problems.append(
            f"rawPeak pixel centre -> ({lat:.6f},{lon:.6f}) != ({p['latitude']},{p['longitude']})"
        )
    # Footprint corners are the outer corners of the pixel grid.
    w, h = src.width, src.height
    for (clat, clon), (c, r) in zip(
        src.footprint_corners_lat_lon, ((0, 0), (w, 0), (w, h), (0, h)), strict=True
    ):
        lat, lon = pixel_to_lat_lon(src.projection, src.grid, c, r)
        if (
            abs(lat - clat) > GEOLOCATION_TOLERANCE_DEG
            or lon_diff(lon, clon) > GEOLOCATION_TOLERANCE_DEG
        ):
            problems.append(
                f"footprint corner ({c},{r}) -> ({lat:.6f},{lon:.6f}) != ({clat},{clon})"
            )
    return problems


# --------------------------------------------------------------------------- modalities
def describe_modalities(src: SectorSource) -> list[ModalityAvailability]:
    """What the published fixture really contains, and what it does not.

    ``atAnchor`` values are single-point metadata, NOT pixelwise physical layers. No pixelwise
    input-modality rasters ship in the published example, so thermal/terrain tools are unavailable.
    """
    out = [
        ModalityAvailability(
            "nasa_ibm_prediction_raw",
            "pixelwise",
            unit="prospectivity index (dimensionless; not a probability or ice abundance)",
            note=(
                f"{src.width}x{src.height} @ {src.pixel_size_m} m/px; "
                "raw values may lie outside [0,1]"
            ),
        )
    ]
    if src.psr_fraction is not None:
        out.append(
            ModalityAvailability(
                "psr_fraction",
                "tile_scalar",
                unit="fraction",
                value=src.psr_fraction,
                note="whole-tile scalar from upstream metadata; no spatial PSR mask",
            )
        )
    units = {"slopeDeg": "deg", "maxTemperatureK": "K", "permanentShadow": None}
    for key, unit in units.items():
        if key in src.at_anchor:
            out.append(
                ModalityAvailability(
                    f"anchor_{key}",
                    "single_anchor_point",
                    unit=unit,
                    value=src.at_anchor[key],
                    note="single-anchor metadata, not a spatial field",
                )
            )
    for name in ("pixelwise_temperature", "pixelwise_shadow_mask", "pixelwise_terrain"):
        out.append(
            ModalityAvailability(
                name,
                "unavailable",
                note="UNAVAILABLE_MODALITY: not included in the published example; real layer "
                "names/shapes must be discovered from pinned SomBench config (P6)",
            )
        )
    return out
