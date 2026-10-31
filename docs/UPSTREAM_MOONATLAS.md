# MOONATLAS upstream acquisition notes (P1)

IceClaw reads a MOONATLAS export as **read-only validated data**. MOONATLAS runs in its own
environment so its dependencies never enter the IceClaw/Streamlit/agent environment.

## Pinned revision

| Item | Value |
| --- | --- |
| Repository | https://github.com/RRG1312/moonatlas-science |
| Commit (HEAD of `main` when acquired, 2026-10-31) | `945fb5196679cf6927354c53be624d745083f9e9` |
| Commit subject | `fix: regenerate science 0.3.1 published metadata` (2026-09-18) |
| Export used | `examples/science-smoke` (`dataMode: real-smoke`, `processingVersion: science-0.3.1`) |

Re-record with `scripts/record_moonatlas_validation.sh`; output goes to the git-ignored
`data/processed/p1/moonatlas_validation.txt`.

## Setup (separate environment)

```bash
cd ~/IceClaw
git clone https://github.com/RRG1312/moonatlas-science.git .upstream/moonatlas-science
cd .upstream/moonatlas-science
git checkout 945fb5196679cf6927354c53be624d745083f9e9
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
moonatlas-science validate --root examples/science-smoke --scope published
```

Scope `published` does not check raw inference artifacts (`checkpoint run` files are not in the
repository); the checkpoint was **not** downloaded and no model was executed.

## What the fixture is (and is not)

- Sector `ice-s-0001-0002`: patch `patch_0001_0002_S_80S`, south pole, centre
  -81.450066°, -35.371514°, split `test`, dataset `sombench-ice-prospectivity-regression`,
  checkpoint `ni_lfm_ps8_all_modalities_s42.ckpt`. Prediction is a **published upstream
  output** (`source_mode=upstream_published_example`), not a fresh IceClaw inference.
- Grid 256×256, 238.56 m/px, spherical polar stereographic (R = 1 737 400 m, k = 0.994).
- Raw prediction: lossless WebP, `u16-rg-linear`: `value = offset + scale·(R·256 + G)`, alpha =
  coverage. Max quantization error is `scale/2` (≈ 8.2e-6). Display WebPs are clipped 8-bit
  previews and are never read for science.
- Pixelwise input modalities (temperature, shadow, terrain) are **absent**. `atAnchor` (slope,
  max temperature, permanent-shadow flag) is single-point metadata and `psrFraction` is a
  whole-tile scalar; neither is a spatial field.
- The reference raster is evaluator-only and is not read by the adapter.

## Adapter tolerance

Recomputed min/max/mean/p90 are compared with upstream's published values with tolerance
`scale/2 + 1e-6` (half a quantization step plus upstream's 6-decimal rounding), ≈ 9.2e-6.
Covered-pixel count must match exactly.

## Contract notes

No change to the canonical contracts was needed. The adapter returns its own `SectorSource`;
`data/catalog.py` (P2) maps it into `RegionCatalogEntry`: `footprint` is `(latitude, longitude)`
for the NW, NE, SE, SW outer pixel corners; `affine` is GDAL order; `prediction_checkpoint_id` is
`<filename>@sha256:<upstream-published sha>` (not verified against a local download). Only
pixelwise layers appear in `modalities`, so the published example has readiness `partial`.
See `docs/SOMBENCH_FEASIBILITY.md` for acquiring real input layers.
