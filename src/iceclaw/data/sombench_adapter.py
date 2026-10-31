"""Reader/validator for original SomBench ice-prospectivity GeoTIFFs (observation inputs).

Source: ``nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression`` at the pinned revision.
Files are fetched by ``scripts/prepare_sombench.py`` (hub-verified, hashes recorded in
``download_manifest.json``). This module never downloads anything, never resamples, and never
reads the expert target ``PRO`` (evaluator-only, stored elsewhere).

Pairing with a MOONATLAS prediction is *verified*, not assumed: see ``check_alignment``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iceclaw.data import moonatlas_adapter as ma

REPO = "nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression"
PINNED_REVISION = "78040503ceaa68107829f48114774112e8ee940d"
INPUT_LAYERS = ("ASP_SIN_COS", "CUR", "DICE", "LPSR", "LPSR_DEN", "LPSR_DIS", "SLOPE", "TMAX")
SPLIT_FILES = {
    "train": "train_filtered.txt",
    "validation": "val_filtered.txt",
    "test": "test_filtered.txt",
}
HUB_VERIFIED = {"lfs-sha256", "git-blob-sha1"}
GRID_TOL_M = 1e-6  # observed float noise ~5e-13 m


class SombenchError(RuntimeError):
    """Observation-source failure with a machine-readable ``code``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class LayerSpec:
    """What is and is not established about a layer. ``unit=None`` means UNKNOWN: such a layer
    gets no physical interpretation and is not exposed to tools."""

    layer: str
    bands: int
    unit: str | None
    unit_basis: str
    exposed_as: str | None  # IceClaw pixelwise modality name, or None


LAYER_SPECS: dict[str, LayerSpec] = {
    "TMAX": LayerSpec(
        "TMAX",
        1,
        "K",
        "MOONATLAS publishes this layer as maxTemperatureK and its value at the published anchor "
        "pixel matches; range 53-330 K is physical. The GeoTIFF itself has no unit tag.",
        "pixelwise_temperature",
    ),
    "LPSR": LayerSpec(
        "LPSR",
        1,
        "fraction",
        "values in [0,1]; MOONATLAS treats >=0.5 as permanently shadowed and the tile mean equals "
        "its published psrFraction. No unit tag in the file.",
        "pixelwise_shadow_mask",
    ),
    "SLOPE": LayerSpec(
        "SLOPE",
        1,
        "deg",
        "MOONATLAS publishes this layer as slopeDeg and its value at the anchor pixel matches; "
        "range 0-34 is a plausible slope in degrees. No unit tag in the file.",
        "pixelwise_terrain",
    ),
    "CUR": LayerSpec("CUR", 1, None, "curvature; unit undocumented upstream", None),
    "DICE": LayerSpec("DICE", 1, None, "ice-stability depth(?); unit undocumented upstream", None),
    "LPSR_DEN": LayerSpec("LPSR_DEN", 1, None, "PSR density; unit undocumented upstream", None),
    "LPSR_DIS": LayerSpec(
        "LPSR_DIS",
        1,
        None,
        "distance to PSR; values 0-11050 suggest metres but this is not documented upstream",
        None,
    ),
    "ASP_SIN_COS": LayerSpec(
        "ASP_SIN_COS",
        2,
        None,
        "two bands in [-1,1]; band 2 statistics equal the ONLYCOS (COSASPECT) band statistics, so "
        "band 1 = sin, band 2 = cos; dimensionless but not exposed",
        None,
    ),
}
assert set(LAYER_SPECS) == set(INPUT_LAYERS)


@dataclass(frozen=True)
class LayerFile:
    layer: str
    relpath: str
    sha256: str
    size: int
    hub_verification: str
    count: int
    height: int
    width: int
    dtype: str
    nodata: float | None
    transform: tuple[float, float, float, float, float, float]
    crs_proj4: str
    tags: dict[str, str]
    finite_fraction: float


@dataclass(frozen=True)
class SombenchPatch:
    patch_id: str
    sector_id: str
    pole: str
    split: str
    root: Path
    dataset_revision: str
    layers: dict[str, LayerFile]
    grid: dict[str, Any]  # xMinM, yMaxM, pixelSizeXM, pixelSizeYM, widthPx, heightPx
    crs_proj4: str

    def exposed(self) -> dict[str, str]:
        """IceClaw modality name -> SomBench layer, for layers with an established unit."""
        return {s.exposed_as: s.layer for s in LAYER_SPECS.values() if s.exposed_as}

    def unit_unknown_layers(self) -> list[str]:
        return sorted(n for n, s in LAYER_SPECS.items() if s.unit is None)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class AlignmentReport:
    checks: list[Check] = field(default_factory=list)
    dihedral: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


# ----------------------------------------------------------------------------- loading
_PATCH_RE = re.compile(r"^patch_(\d{4})_(\d{4})_(S_80S|80N)$")


def parse_patch_id(patch_id: str) -> tuple[str, str]:
    """(sector_id, pole) using the same rule as MOONATLAS ``ice_normalize.sector_id``."""
    m = _PATCH_RE.match(patch_id)
    if not m:
        raise SombenchError("MALFORMED_PATCH_ID", patch_id)
    pole = "south" if m.group(3) == "S_80S" else "north"
    return f"ice-{pole[0]}-{m.group(1)}-{m.group(2)}", pole


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_of(root: Path, patch_id: str) -> str:
    found = []
    for split, name in SPLIT_FILES.items():
        f = root / name
        if not f.is_file():
            raise SombenchError("MISSING_FILE", f"split file {name} not found")
        if patch_id in f.read_text().split():
            found.append(split)
    if len(found) != 1:
        raise SombenchError("SPLIT_MISMATCH", f"{patch_id} is in splits {found or 'none'}")
    return found[0]


def _download_manifest(root: Path) -> dict[str, Any]:
    f = root / "download_manifest.json"
    if not f.is_file():
        raise SombenchError("MISSING_FILE", "download_manifest.json not found")
    m = json.loads(f.read_text())
    if m.get("repo") != REPO or m.get("revision") != PINNED_REVISION:
        raise SombenchError(
            "REVISION_MISMATCH", f"manifest is for {m.get('repo')}@{m.get('revision')}"
        )
    return m["files"]


def load_patch(root: str | Path, patch_id: str) -> SombenchPatch:
    """Open and validate all eight input layers of one patch with Rasterio."""
    import rasterio

    root = Path(root)
    sector_id, pole = parse_patch_id(patch_id)
    split = _split_of(root, patch_id)
    files = _download_manifest(root)
    m = _PATCH_RE.match(patch_id)
    layers: dict[str, LayerFile] = {}
    for layer, spec in LAYER_SPECS.items():
        rel = f"{patch_id}_{layer}.tif"
        path = root / rel
        if not path.is_file():
            raise SombenchError("MISSING_FILE", f"{rel} not found")
        rec = files.get(rel)
        if rec is None or rec.get("hub_verification") not in HUB_VERIFIED:
            raise SombenchError("UNVERIFIED_DOWNLOAD", f"{rel} was not hub-verified at download")
        sha = _sha256(path)
        if sha != rec["sha256"]:
            raise SombenchError("CHECKSUM_MISMATCH", f"{rel}: {sha[:12]}... != recorded")
        try:
            ds = rasterio.open(path)
        except Exception as e:  # rasterio raises RasterioIOError and others
            raise SombenchError("CORRUPT_RASTER", f"{rel}: {e}") from e
        with ds:
            if ds.count != spec.bands:
                raise SombenchError(
                    "UNEXPECTED_BAND_COUNT", f"{rel}: {ds.count} bands, expected {spec.bands}"
                )
            if set(ds.dtypes) != {"float32"}:
                raise SombenchError("UNEXPECTED_DTYPE", f"{rel}: {ds.dtypes}")
            if ds.crs is None:
                raise SombenchError("MISSING_CRS", rel)
            tags = ds.tags()
            if tags.get("variable_name") != layer:
                raise SombenchError(
                    "LAYER_NAME_MISMATCH", f"{rel}: variable_name={tags.get('variable_name')!r}"
                )
            if (tags.get("patch_row"), tags.get("patch_col")) != (
                str(int(m.group(1))),
                str(int(m.group(2))),
            ):
                raise SombenchError("TILE_TAG_MISMATCH", f"{rel}: row/col tags {tags}")
            if tags.get("hemisphere") != pole[0].upper():
                raise SombenchError("POLE_MISMATCH", f"{rel}: hemisphere={tags.get('hemisphere')}")
            data = ds.read()
            nodata = ds.nodata
            layers[layer] = LayerFile(
                layer=layer,
                relpath=rel,
                sha256=sha,
                size=path.stat().st_size,
                hub_verification=rec["hub_verification"],
                count=ds.count,
                height=ds.height,
                width=ds.width,
                dtype="float32",
                nodata=None if nodata is None else float(nodata),
                transform=tuple(ds.transform)[:6],
                crs_proj4=ds.crs.to_proj4(),
                tags=dict(tags),
                finite_fraction=float(np.isfinite(data).mean()),
            )
    ref = layers["TMAX"]
    for lf in layers.values():
        if (lf.height, lf.width) != (ref.height, ref.width):
            raise SombenchError("SHAPE_MISMATCH", f"{lf.relpath}: {(lf.height, lf.width)}")
        if lf.transform != ref.transform or lf.crs_proj4 != ref.crs_proj4:
            raise SombenchError("GRID_MISMATCH", f"{lf.relpath} is on a different grid/CRS")
    t = ref.transform
    grid = {
        "xMinM": t[2],
        "yMaxM": t[5],
        "pixelSizeXM": t[0],
        "pixelSizeYM": -t[4],
        "widthPx": ref.width,
        "heightPx": ref.height,
    }
    return SombenchPatch(
        patch_id, sector_id, pole, split, root, PINNED_REVISION, layers, grid, ref.crs_proj4
    )


def read_layer(patch: SombenchPatch, layer: str) -> tuple[np.ndarray, np.ndarray]:
    """(values float64 shaped (bands, H, W), finite mask (H, W)); sha256 re-verified at read time.

    Nodata is NaN in these files; pixels are never filled.
    """
    import rasterio

    if layer not in LAYER_SPECS:
        raise SombenchError("UNKNOWN_LAYER", f"{layer!r} is not one of {sorted(LAYER_SPECS)}")
    lf = patch.layers[layer]
    path = patch.root / lf.relpath
    if not path.is_file():
        raise SombenchError("MISSING_FILE", f"{lf.relpath} not found")
    if _sha256(path) != lf.sha256:
        raise SombenchError("CHECKSUM_MISMATCH", f"{lf.relpath} changed since load")
    with rasterio.open(path) as ds:
        data = ds.read().astype(np.float64)
    return data, np.isfinite(data).all(axis=0)


# --------------------------------------------------------------------------- alignment
def _crs_matches(proj_string: str, projection: dict[str, Any]) -> list[str]:
    import rasterio.crs

    d = rasterio.crs.CRS.from_string(proj_string).to_dict()
    want = {
        "proj": "stere",
        "lat_0": projection["latitudeOfOriginDeg"],
        "lon_0": projection["centralMeridianDeg"],
        "k": projection["scaleFactor"],
        "x_0": projection["falseEastingM"],
        "y_0": projection["falseNorthingM"],
        "R": projection["radiusM"],
    }
    bad = []
    for k, v in want.items():
        got = d.get(k)
        if (got != v) if isinstance(v, str) else (got is None or abs(float(got) - float(v)) > 1e-9):
            bad.append(f"{k}: {got} != {v}")
    return bad


def _dihedral(a: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for k in range(4):
        r = np.rot90(a, k)
        out[f"rot{90 * k}"] = r
        out[f"rot{90 * k}+flipud"] = np.flipud(r)
    return out


def orientation_scan(pred: np.ndarray, layer: np.ndarray, finite: np.ndarray) -> dict[str, float]:
    """Pearson r between the prediction and each of the 8 dihedral transforms of ``layer``."""
    out = {}
    for name, arr in _dihedral(layer).items():
        fin = _dihedral(finite.astype(np.uint8))[name].astype(bool)
        if fin.sum() < 100:
            out[name] = float("nan")
            continue
        out[name] = float(np.corrcoef(pred[fin], arr[fin])[0, 1])
    return out


def check_alignment(
    patch: SombenchPatch,
    src: ma.SectorSource,
    prediction: np.ndarray,
    min_identity_r: float = 0.5,
    min_margin: float = 2.0,
) -> AlignmentReport:
    """Verify that ``patch`` observations are the same tile, grid and orientation as ``src``.

    Pre-registered orientation rule: for TMAX and LPSR the IDENTITY transform must give the
    largest |r| with the prediction, with |r| >= ``min_identity_r`` and >= ``min_margin`` times
    the best non-identity |r|. Nothing is resampled; any failure disables pairing.
    """
    rep = AlignmentReport()

    def add(name: str, ok: bool, detail: str = "") -> None:
        rep.checks.append(Check(name, bool(ok), detail))

    add(
        "tile_identity",
        patch.sector_id == src.sector_id
        and patch.patch_id == src.patch_id
        and patch.split == src.dataset_split
        and patch.pole == src.pole,
        f"{patch.patch_id}/{patch.split} vs {src.patch_id}/{src.dataset_split}",
    )
    add(
        "dataset_revision",
        patch.dataset_revision == src.dataset_revision,
        f"{patch.dataset_revision} vs {src.dataset_revision}",
    )
    add(
        "shape",
        (patch.grid["heightPx"], patch.grid["widthPx"]) == (src.height, src.width),
        f"{(patch.grid['heightPx'], patch.grid['widthPx'])} vs {(src.height, src.width)}",
    )
    bad = _crs_matches(patch.crs_proj4, src.projection)
    add("crs", not bad, "; ".join(bad))
    g, sg = patch.grid, src.grid
    diffs = {
        k: abs(float(g[k]) - float(sg[k])) for k in ("xMinM", "yMaxM", "pixelSizeXM", "pixelSizeYM")
    }
    add("affine", max(diffs.values()) <= GRID_TOL_M, f"max |diff| {max(diffs.values()):.3g} m")

    # corners / pixel centres computed from the GeoTIFF's OWN transform, compared with the
    # coordinates MOONATLAS published.
    w, h = g["widthPx"], g["heightPx"]
    worst = 0.0
    for (clat, clon), (c, r) in zip(
        src.footprint_corners_lat_lon, ((0, 0), (w, 0), (w, h), (0, h)), strict=True
    ):
        lat, lon = ma.pixel_to_lat_lon(src.projection, g, c, r)
        worst = max(worst, abs(lat - clat), ma.lon_diff(lon, clon))
    add("footprint_corners", worst <= ma.GEOLOCATION_TOLERANCE_DEG, f"worst {worst:.3g} deg")
    pk = src.raw_peak
    row, col = int(pk["row"]), int(pk["col"])
    lat, lon = ma.pixel_to_lat_lon(src.projection, g, col + 0.5, row + 0.5)
    err = max(abs(lat - pk["latitude"]), ma.lon_diff(lon, pk["longitude"]))
    add("peak_pixel_center", err <= ma.GEOLOCATION_TOLERANCE_DEG, f"{err:.3g} deg")

    tmax, tfin = read_layer(patch, "TMAX")
    slope, _ = read_layer(patch, "SLOPE")
    lpsr, lfin = read_layer(patch, "LPSR")
    a = src.at_anchor
    if tfin[row, col] and "maxTemperatureK" in a:
        add(
            "anchor_temperature",
            round(float(tmax[0, row, col]), 1) == a["maxTemperatureK"],
            f"{tmax[0, row, col]:.4f} vs {a['maxTemperatureK']}",
        )
    else:
        add("anchor_temperature", False, "anchor pixel not finite or metadata missing")
    add(
        "anchor_slope",
        round(float(slope[0, row, col]), 2) == a.get("slopeDeg"),
        f"{slope[0, row, col]:.4f} vs {a.get('slopeDeg')}",
    )
    add(
        "anchor_shadow",
        bool(lpsr[0, row, col] >= 0.5) == a.get("permanentShadow"),
        f"{lpsr[0, row, col]} vs {a.get('permanentShadow')}",
    )
    if src.psr_fraction is not None:
        add(
            "psr_fraction",
            round(float(lpsr[0][lfin].mean()), 4) == src.psr_fraction,
            f"{lpsr[0][lfin].mean():.6f} vs {src.psr_fraction}",
        )

    for name, arr, fin in (("TMAX", tmax[0], tfin), ("LPSR", lpsr[0], lfin)):
        scan = orientation_scan(prediction, arr, fin)
        rep.dihedral[name] = scan
        ident = scan["rot0"]
        others = [abs(v) for k, v in scan.items() if k != "rot0" and not np.isnan(v)]
        best_other = max(others) if others else float("nan")
        ok = (
            not np.isnan(ident)
            and abs(ident) >= min_identity_r
            and abs(ident) >= min_margin * best_other
        )
        add(f"orientation_{name}", ok, f"identity r={ident:.3f}, best other |r|={best_other:.3f}")
    return rep
