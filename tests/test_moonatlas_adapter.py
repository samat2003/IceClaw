"""P1 adapter tests.

* ``test_synthetic_*``: tiny hand-built arrays; verify decoder math only (not lunar science).
* ``test_upstream_*``: the real published MOONATLAS smoke export in ``.upstream/``; skipped
  (and reported as skipped) when that checkout is absent.
* ``test_corrupt_*``: copies of the real export that are deliberately damaged in a tmp dir.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from iceclaw.contracts import SourceMode
from iceclaw.data import moonatlas_adapter as ma

REPO = Path(__file__).resolve().parents[1]
SMOKE = Path(
    os.environ.get(
        "ICECLAW_MOONATLAS_SMOKE_ROOT",
        REPO / ".upstream/moonatlas-science/examples/science-smoke",
    )
)
needs_upstream = pytest.mark.skipif(
    not (SMOKE / "manifest.json").is_file(), reason="MOONATLAS smoke export not checked out"
)

RAW_REL = "overlays/ice/ice-s-0001-0002-raw.webp"


# ------------------------------------------------------------------ synthetic decoder
def test_synthetic_decode_formula_and_coverage():
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0, 0, :2] = (0, 0)  # level 0
    rgba[0, 1, :2] = (0, 1)  # level 1
    rgba[0, 2, :2] = (1, 0)  # level 256
    rgba[1, 0, :2] = (255, 255)  # level 65535
    rgba[1, 1, 3] = 0  # uncovered
    enc = {"kind": "u16-rg-linear", "offset": -0.5, "scale": 0.001}
    d = ma.decode_rgba(rgba, enc, expected_shape=(2, 3))
    assert d.values[0, 0] == pytest.approx(-0.5)
    assert d.values[0, 1] == pytest.approx(-0.499)
    assert d.values[0, 2] == pytest.approx(-0.5 + 0.256)
    assert d.values[1, 0] == pytest.approx(-0.5 + 65.535)  # > 1 is preserved, not clipped
    assert d.covered.tolist() == [[True, True, True], [True, False, True]]


def test_synthetic_decode_rgb_without_alpha_is_fully_covered():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    d = ma.decode_rgba(rgb, {"offset": 0.0, "scale": 1.0})
    assert d.covered.all()


def test_synthetic_decode_rejects_bad_arrays():
    enc = {"offset": 0.0, "scale": 1.0}
    with pytest.raises(ma.SourceError) as e:
        ma.decode_rgba(np.zeros((2, 2, 4), dtype=np.uint8), enc, expected_shape=(3, 3))
    assert e.value.code == "SHAPE_MISMATCH"
    with pytest.raises(ma.SourceError) as e:
        ma.decode_rgba(np.zeros((2, 2), dtype=np.uint8), enc)
    assert e.value.code == "CORRUPT_RASTER"
    with pytest.raises(ma.SourceError):
        ma.decode_rgba(np.zeros((2, 2, 4), dtype=np.float32), enc)


def test_synthetic_summary_ignores_uncovered_and_keeps_out_of_range():
    values = np.array([[-0.1, 0.5], [1.2, 99.0]])
    covered = np.array([[True, True], [True, False]])
    s = ma.summarize_prediction(ma.DecodedPrediction(values, covered))
    assert (s["min"], s["max"]) == (-0.1, 1.2)
    assert s["valid_pixel_count"] == 3 and s["outside_unit_interval_count"] == 2
    assert s["mean"] == pytest.approx((-0.1 + 0.5 + 1.2) / 3)
    with pytest.raises(ma.SourceError):
        ma.summarize_prediction(ma.DecodedPrediction(values, np.zeros_like(covered)))


def test_synthetic_projection_roundtrip():
    proj = {
        "kind": "polar-stereographic",
        "latitudeOfOriginDeg": -90,
        "centralMeridianDeg": 0,
        "scaleFactor": 0.994,
        "falseEastingM": 500000.0,
        "falseNorthingM": 500000.0,
        "radiusM": 1737400.0,
    }
    grid = {"xMinM": 320000.0, "yMaxM": 740000.0, "pixelSizeXM": 240.0, "pixelSizeYM": 240.0}
    for col, row in [(0, 0), (10.5, 200.5), (255.5, 3.25)]:
        lat, lon = ma.pixel_to_lat_lon(proj, grid, col, row)
        assert lat < -75
        c2, r2 = ma.lat_lon_to_pixel(proj, grid, lat, lon)
        assert (c2, r2) == pytest.approx((col, row), abs=1e-6)
    # the south pole maps to the false origin
    lat, _ = ma.pixel_to_lat_lon(proj, {**grid, "xMinM": 500000.0, "yMaxM": 500000.0}, 0, 0)
    assert lat == pytest.approx(-90.0)
    with pytest.raises(ma.SourceError):
        ma.pixel_to_lat_lon({**proj, "kind": "mercator"}, grid, 0, 0)


# ------------------------------------------------------------- real upstream artifact
@pytest.fixture(scope="module")
def sector():
    return ma.load_sector(SMOKE)


@pytest.fixture(scope="module")
def prediction(sector):
    return ma.decode_prediction(sector)


@needs_upstream
def test_upstream_metadata_and_provenance(sector):
    s = sector
    assert s.sector_id == "ice-s-0001-0002" and s.pole == "south"
    assert s.source_mode is SourceMode.UPSTREAM_PUBLISHED_EXAMPLE
    assert s.patch_id == "patch_0001_0002_S_80S"
    assert s.dataset_split == "test"
    assert s.source_dataset == "sombench-ice-prospectivity-regression"
    assert s.checkpoint == "ni_lfm_ps8_all_modalities_s42.ckpt"
    assert len(s.checkpoint_sha256) == 64 and len(s.prediction_sha256) == 64
    assert (s.width, s.height) == (256, 256)
    assert s.pixel_size_m == pytest.approx(238.56)
    assert s.projection["kind"] == "polar-stereographic"
    assert s.center_lat_lon == pytest.approx((-81.450066, -35.371514))
    assert s.encoding["kind"] == "u16-rg-linear"
    assert s.processing_version == "science-0.3.1"


@needs_upstream
def test_upstream_raw_decode_matches_published_statistics(sector, prediction):
    assert prediction.values.shape == (256, 256) and prediction.covered.all()
    summary = ma.summarize_prediction(prediction)
    report = ma.compare_to_upstream(sector, summary)
    assert report.ok, [i for i in report.items if not i.ok]
    # documented tolerance = scale/2 + 1e-6 (~9.2e-6), far tighter than display quantization
    assert sector.stats_tolerance < 1e-5
    assert summary["valid_pixel_count"] == 65536
    # raw values are preserved outside [0, 1] (display clipping must not be applied)
    assert summary["min"] < 0 < 1 < summary["max"]
    assert summary["outside_unit_interval_count"] == sector.clipped_pixels == 825


@needs_upstream
def test_upstream_peak_pixels_and_geolocation(sector, prediction):
    assert ma.verify_peak_and_geometry(sector, prediction) == []
    p = sector.raw_peak
    assert (p["row"], p["col"]) == (194, 33)
    assert prediction.values[194, 33] == pytest.approx(p["value"], abs=sector.stats_tolerance)
    # sector centre is the centre of the pixel grid (footprint centre)
    lat, lon = ma.pixel_to_lat_lon(sector.projection, sector.grid, 128, 128)
    assert (lat, lon) == pytest.approx(sector.center_lat_lon, abs=1e-4)


@needs_upstream
def test_upstream_display_preview_is_not_the_science_source(sector, prediction):
    display = SMOKE / "overlays/ice/ice-s-0001-0002.webp"
    with Image.open(display) as im:
        gray = np.asarray(im.convert("LA"))[..., 0] / 255.0
    # display is clipped to [0,1] and 8-bit: it cannot reproduce the raw range
    assert gray.max() <= 1.0 < prediction.values.max()
    assert np.abs(gray - np.clip(prediction.values, 0, 1)).max() <= 1 / 255 + 1e-9
    assert sector.prediction_relpath.endswith("-raw.webp")


@needs_upstream
def test_upstream_modality_availability(sector):
    m = {x.name: x for x in ma.describe_modalities(sector)}
    assert m["nasa_ibm_prediction_raw"].status == "pixelwise"
    assert m["psr_fraction"].status == "tile_scalar" and m["psr_fraction"].value == 0.0665
    assert m["anchor_maxTemperatureK"].status == "single_anchor_point"
    assert m["anchor_maxTemperatureK"].value == 84.9
    assert m["anchor_slopeDeg"].value == 6.56 and m["anchor_permanentShadow"].value is True
    for name in ("pixelwise_temperature", "pixelwise_shadow_mask", "pixelwise_terrain"):
        assert m[name].status == "unavailable" and "UNAVAILABLE_MODALITY" in m[name].note
    # single-anchor metadata must never be classified as a pixelwise layer
    assert not any(x.status == "pixelwise" for x in m.values() if x.name.startswith("anchor_"))


@needs_upstream
def test_upstream_reference_not_exposed_by_adapter(sector):
    dumped = repr(sector) + repr(ma.describe_modalities(sector))
    assert "reference" not in dumped.lower()


# ------------------------------------------------------- corrupted / missing sources
@pytest.fixture
def smoke_copy(tmp_path):
    if not (SMOKE / "manifest.json").is_file():
        pytest.skip("MOONATLAS smoke export not checked out")
    dst = tmp_path / "smoke"
    (dst / "overlays/ice").mkdir(parents=True)
    for f in ("manifest.json", "processing-manifest.json", "ice-sectors.json"):
        shutil.copy(SMOKE / f, dst / f)
    shutil.copy(SMOKE / RAW_REL, dst / RAW_REL)
    ma.load_sector(dst)  # the copy itself is valid
    return dst


def _rewrite_checksum(root: Path, rel: str) -> None:
    pm = root / "processing-manifest.json"
    data = json.loads(pm.read_text())
    data["checksums"][rel] = ma.sha256_file(root / rel)
    pm.write_text(json.dumps(data))


def _code(root: Path) -> str:
    with pytest.raises(ma.SourceError) as e:
        ma.decode_prediction(ma.load_sector(root))
    return e.value.code


def test_corrupt_missing_raster(smoke_copy):
    (smoke_copy / RAW_REL).unlink()
    assert _code(smoke_copy) == "MISSING_FILE"


def test_corrupt_flipped_byte_fails_checksum(smoke_copy):
    p = smoke_copy / RAW_REL
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0xFF
    p.write_bytes(bytes(b))
    assert _code(smoke_copy) == "CHECKSUM_MISMATCH"


def test_corrupt_truncated_raster_with_matching_checksum(smoke_copy):
    p = smoke_copy / RAW_REL
    p.write_bytes(p.read_bytes()[:200])
    _rewrite_checksum(smoke_copy, RAW_REL)
    assert _code(smoke_copy) == "CORRUPT_RASTER"


def test_corrupt_wrong_dimensions_in_metadata(smoke_copy):
    f = smoke_copy / "ice-sectors.json"
    data = json.loads(f.read_text())
    data[0]["source"]["grid"]["widthPx"] = 128
    f.write_text(json.dumps(data))
    _rewrite_checksum(smoke_copy, "ice-sectors.json")
    assert _code(smoke_copy) == "SHAPE_MISMATCH"


def test_corrupt_edited_metadata_fails_checksum(smoke_copy):
    f = smoke_copy / "ice-sectors.json"
    f.write_text(f.read_text().replace("-81.450066", "-81.450067", 1))
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy)
    assert e.value.code == "CHECKSUM_MISMATCH"


def test_corrupt_raster_swapped_after_load(smoke_copy):
    src = ma.load_sector(smoke_copy)
    other = np.zeros((256, 256, 4), dtype=np.uint8)
    Image.fromarray(other, "RGBA").save(smoke_copy / RAW_REL, "WEBP", lossless=True)
    with pytest.raises(ma.SourceError) as e:
        ma.decode_prediction(src)
    assert e.value.code == "CHECKSUM_MISMATCH"


def test_corrupt_malformed_json_and_missing_manifest(smoke_copy):
    (smoke_copy / "ice-sectors.json").write_text("{not json")
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy)
    assert e.value.code in {"MALFORMED_METADATA", "CHECKSUM_MISMATCH"}
    (smoke_copy / "manifest.json").unlink()
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy)
    assert e.value.code == "MISSING_FILE"


def test_corrupt_unknown_sector_and_unsupported_encoding(smoke_copy):
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy, "ice-s-9999-9999")
    assert e.value.code == "NOT_FOUND"
    f = smoke_copy / "ice-sectors.json"
    data = json.loads(f.read_text())
    data[0]["prediction"]["heatmap"]["rawEncoding"]["kind"] = "f32"
    f.write_text(json.dumps(data))
    _rewrite_checksum(smoke_copy, "ice-sectors.json")
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy)
    assert e.value.code == "UNSUPPORTED_ENCODING"


def test_corrupt_provenance_mismatch_and_non_smoke(smoke_copy):
    f = smoke_copy / "ice-sectors.json"
    data = json.loads(f.read_text())
    data[0]["provenance"]["checkpoint"] = "other.ckpt"
    f.write_text(json.dumps(data))
    _rewrite_checksum(smoke_copy, "ice-sectors.json")
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy)
    assert e.value.code == "PROVENANCE_MISMATCH"
    pm = smoke_copy / "processing-manifest.json"
    d = json.loads(pm.read_text())
    d["scope"] = "full"
    pm.write_text(json.dumps(d))
    with pytest.raises(ma.SourceError) as e:
        ma.load_sector(smoke_copy)
    assert e.value.code == "UNSUPPORTED_SOURCE_MODE"
