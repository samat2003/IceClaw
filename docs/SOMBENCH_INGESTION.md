# SomBench observation ingestion and pairing (P4)

Facts below were measured on the actual downloaded files (2026-10-31) unless marked otherwise.
Nothing here is a model result: the only NASA–IBM prediction is still the **published** MOONATLAS
output for `ice-s-0001-0002` (`source_mode=upstream_published_example`); no checkpoint was
downloaded and no inference was run.

## 1. Source and download

| Item | Value |
| --- | --- |
| Dataset | `nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression`, CC BY 4.0, public |
| Pinned revision | `78040503ceaa68107829f48114774112e8ee940d` (hub `sha` matched) |
| Command | `python scripts/prepare_sombench.py --patches-from-split test --pole south --layers inputs` |
| Credentials | none: `token=False` (a cached login on the machine is not used) |
| Fetched | 100 files, **21 299 972 bytes (21.3 MB)**: 12 south test patches × 8 input layers (96 files) + 3 split lists + `band_statistics.json` |
| Verification | every file checked against the hub: 71 by LFS sha256, 29 small files by git blob SHA1; sizes/sha256 in `data/sombench/prospectivity_dataset/download_manifest.json` (git-ignored) |
| Bytes per layer (12 tiles) | ASP_SIN_COS 7 400 640 · CUR 3 675 422 · SLOPE 3 582 106 · TMAX 3 103 196 · DICE 1 694 748 · LPSR_DIS 1 506 476 · LPSR_DEN 258 812 · LPSR 73 846 |
| Evaluator-only target | `patch_0001_0002_S_80S_PRO.tif` (298 308 B, sha256 `fb247a13…ed018`, LFS-verified) stored under `data/evaluation_inputs/prospectivity_target/`, **outside** the observation directory; the catalog never reads it |

South test patches downloaded: `patch_0001_0002`, `0003_0002`, `0005_0008`, `0007_0006`,
`0000_0003`, `0003_0000`, `0007_0008`, `0008_0003`, `0001_0008`, `0003_0001`, `0003_0008`,
`0007_0001` (all `_S_80S`, all in `test_filtered.txt`; split membership is re-checked on load).
`ONLYCOS` (the alternative aspect file) was not fetched.

## 2. What the real GeoTIFFs contain (Rasterio)

Every layer of every tile: 256 × 256, `float32`, nodata = NaN, LZW, no unit/description tags,
tags `variable_name`, `patch_row`, `patch_col`, `hemisphere=S`, `tile_size=256`.
CRS (identical in all files): `+proj=stere +lat_0=-90 +lon_0=0 +k=0.994 +x_0=500000
+y_0=500000 +R=1737400 +units=m` ("Moon (2015) Sphere / Ocentric / South Polar").
Each tile has its own affine origin; pixel size 238.56 m. Paired tile affine
`(238.56, 0, 320006.48, 0, −238.56, 741064.88)`.

| Layer | bands | Paired-tile min / max / mean | Unit status |
| --- | --- | --- | --- |
| `TMAX` | 1 | 83.03 / 323.87 / 239.82 | **K** (established, see §3) → `pixelwise_temperature` |
| `LPSR` | 1 | 0 / 1 / 0.0665 (219 distinct values, not binary) | **fraction**; PSR threshold 0.5 (upstream's) → `pixelwise_shadow_mask` |
| `SLOPE` | 1 | 0.143 / 29.82 / 8.73 | **deg** (established) → `pixelwise_terrain` |
| `ASP_SIN_COS` | **2** | −1.00 / 1.00 / −0.005 | band 1 = sin, band 2 = cos (band 2 stats equal the `ONLYCOS`/`COSASPECT` stats); not exposed |
| `CUR` | 1 | −2.6e-3 / 1.2e-3 / −4e-6 | **unknown**, not exposed |
| `DICE` | 1 | 0 / 1 / 0.587 | **unknown**, not exposed |
| `LPSR_DEN` | 1 | 0 / 1 / 0.0664 | **unknown**, not exposed |
| `LPSR_DIS` | 1 | 0 / 11 050 / 2 491 | **unknown** (metres plausible, undocumented), not exposed |

**Resolved:** the dataset README's "single-band" claim is wrong for `ASP_SIN_COS` (2 bands).
**Still open:** README says 11 layers / 1 782 files while the hub lists 10 layer names / 1 620
patch files; units of `CUR`, `DICE`, `LPSR_DEN`, `LPSR_DIS` remain undocumented. Layers with an
unknown unit are downloaded and hashed but get no physical interpretation and are not available
to any tool.

**Natural nodata is real:** 6 of the 12 tiles contain NaN. Finite fraction of `TMAX`:
0001_0008 0.474 · 0000_0003 0.773 · 0003_0000 0.773 · 0007_0008 0.983 · 0007_0001 0.998 · the
other seven 1.000. NaN is kept as NaN and uncovered; nothing is filled.

## 3. How units were established (and their limits)

The GeoTIFFs carry no unit tags. A unit is accepted only when (a) MOONATLAS publishes the layer
under a unit-named field, (b) its value at the published anchor pixel reproduces that field from
the original array, and (c) the range is physically sensible. Anchor pixel = published raw-peak
pixel (row 194, col 33):

| Published (`atAnchor`) | Original array at (194, 33) |
| --- | --- |
| `maxTemperatureK` 84.9 | `TMAX` 84.9448 K |
| `slopeDeg` 6.56 | `SLOPE` 6.5616 |
| `permanentShadow` true | `LPSR` 1.0 (≥ 0.5) |
| `psrFraction` 0.0665 | mean `LPSR` 0.066528 → 0.0665 |

This establishes the *labels and pairing*; it is not an independent calibration of the
instruments (the temperature is a derived Diviner-based product per MOONATLAS).

## 4. Spatial pairing with the validated prediction (verified, not assumed)

`sombench_adapter.check_alignment` must pass every check or the region stays
`PREDICTION_ONLY` (tested with a vertically flipped but re-hashed TMAX: not paired):

| Check | Result for `ice-s-0001-0002` |
| --- | --- |
| tile identity / split / dataset revision | `patch_0001_0002_S_80S`, test, pinned revision |
| shape | 256 × 256 = 256 × 256 |
| CRS parameters | equal (lat_0, lon_0, k, x_0, y_0, R) |
| affine (xMin, yMax, pixel size) | max |Δ| 4.7e-10 m |
| footprint corners from the GeoTIFF's own transform vs published | worst 4.7e-7° |
| raw-peak pixel-centre coordinate vs published | 2.7e-7° |
| anchor values, PSR fraction | match (§3) |
| orientation: Pearson r(prediction, layer) over the 8 rotations/flips | TMAX identity **r = −0.930**, best non-identity \|r\| 0.113; LPSR identity r = +0.536, best other 0.086 |

Orientation rule (pre-registered): identity must be the largest |r|, with |r| ≥ 0.5 and ≥ 2× the
best non-identity, for both TMAX and LPSR. Nothing is resampled or reprojected.

## 5. Real thermal / terrain measurements (paired tile)

Computed by the unchanged P2 tools on the real arrays; reproduced independently with NumPy
straight from the GeoTIFFs (`abs` ≤ 1e-9) and by `recompute()`.

| Measurement | Value |
| --- | --- |
| PSR pixels (LPSR ≥ 0.5) | 4 360 of 65 536 |
| mean TMAX in PSR / outside PSR | 139.215 K / 246.989 K (Δ −107.774 K) |
| PSR pixels with TMAX < 110 K (caller parameter) | 6.54 % (1.0 % at 90 K, 32.1 % at 130 K) |
| slope mean / median / p95 / max | 8.734° / 7.456° / 20.416° / 29.824° |
| pixels steeper than 20° (caller parameter) | 5.60 % |

The 110 K and 20° values are parameters, not validated physical thresholds. The PSR/temperature
contrast is a descriptive statistic of these inputs, not evidence of ice.

## 6. Catalog status model

| Status | Meaning | Selectable for investigation |
| --- | --- | --- |
| `PAIRED` (readiness `verified`) | verified prediction **and** aligned TMAX/LPSR/SLOPE | yes |
| `PREDICTION_ONLY` (readiness `partial`) | prediction without verified observations (none present, or pairing failed → `pairing_error`) | prediction tools only |
| `OBSERVATION_ONLY` | verified observations, no NASA–IBM prediction (11 tiles) | **no** (`PREDICTION_UNAVAILABLE`) |

`prediction_ready` / `observation_ready` are reported separately in `list-regions --json`.
Observation-only tiles are not `RegionCatalogEntry` objects (the contract requires a prediction).
**Coverage label: `PARTIAL_ONE_REGION`** — one paired region; the other 11 south test tiles have
observations but no verified prediction, and none was fabricated.

## 7. Contract and code changes

* Additive only: `SourceMode.OBSERVATION_ONLY` (for multimodal scenarios that contain no
  prediction), `Settings.sombench_root` (`ICECLAW_SOMBENCH_ROOT`; `None` disables pairing so
  P0–P3 behaviour/tests are unchanged), new `data/sombench_adapter.py`, catalog/service/CLI
  fields. P1/P2 decoding, projection code and tool formulas are untouched.

## 8. Remaining limitations

* One paired region. Predictions for the other tiles need the NASA–IBM checkpoint (1.69 GB +
  backbone/tokenizers); not attempted. RTX 5070 Ti compatibility with the pinned torch is untested.
* Units for four layers unknown; `TMAX`/`SLOPE` units rest on upstream naming + anchor matches.
* Orientation was verified for the paired tile only; the other tiles' pairing is not applicable.
* The published prediction was trained on the same evidential layers; agreement between
  inputs and prediction is not independent evidence (see `SOMBENCH_FEASIBILITY.md` §5).
