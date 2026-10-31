# SomBench ice-prospectivity data feasibility (P2 check; informs P4)

Purpose: decide how to obtain original **test-split observation arrays** so thermal/terrain
specialists have real pixelwise inputs. Nothing here was downloaded except small text/JSON
metadata; no checkpoint was fetched and no inference was run.

**Sources inspected (2026-10-31):** pinned MOONATLAS commit
`945fb5196679cf6927354c53be624d745083f9e9` (`src/moonatlas_science/config.py`, `hub.py`,
`ice_inference.py`, `ice_normalize.py`, `validate.py`, `docs/datasets/*`, README), plus
unauthenticated, metadata-only Hugging Face API/`resolve` requests at the *same pinned
revisions* (dataset file listing with sizes, `README.md`, `*_filtered.txt` split lists,
`band_statistics.json`, model-repo file sizes). Facts marked **[HF]** come from those requests;
**[repo]** from the pinned MOONATLAS code; **[unverified]** means not yet confirmed on real
files.

## 1. Where the arrays live

| Item | Value |
| --- | --- |
| Dataset | `nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression` **[repo]** |
| Pinned revision | `78040503ceaa68107829f48114774112e8ee940d` **[repo, HF sha matches]** |
| Access | public, **not gated**, not private; CC BY 4.0 **[HF]** |
| Download tool | `huggingface_hub.hf_hub_download(..., repo_type="dataset", revision=...)`; MOONATLAS wraps it in `hub.fetch_ice_samples` **[repo]**. No token is required for a public dataset. |
| File naming | one GeoTIFF per layer per patch: `{patch}_{LAYER}.tif`, patch = `patch_RRRR_CCCC_S_80S` (south) or `patch_RRRR_CCCC_80N` (north) **[repo, HF]** |
| Split files | `train_filtered.txt`, `val_filtered.txt`, `test_filtered.txt`, plus `band_statistics.json` **[repo, HF]** |
| Total size | 1 626 files, **≈ 0.38 GB** for the entire dataset **[HF]**; one patch's 8 input layers ≲ 2 MB (per-file sizes: 3.6 kB–0.63 MB) |
| Local path expected by upstream | `data/prospectivity_dataset/` under the MOONATLAS data root **[repo]** |

**Splits [HF]:** test 25 patches (12 south, 13 north), validation 23 listed (README text says
24), train 108. The README says "162 patches"; the file list holds 81 south + 81 north.

South **test** patches (candidates for P4; one of them is already the published sector
`ice-s-0001-0002`):
`patch_0001_0002_S_80S`, `patch_0003_0002_S_80S`, `patch_0005_0008_S_80S`,
`patch_0007_0006_S_80S`, `patch_0000_0003_S_80S`, `patch_0003_0000_S_80S`,
`patch_0007_0008_S_80S`, `patch_0008_0003_S_80S`, `patch_0001_0008_S_80S`,
`patch_0003_0001_S_80S`, `patch_0003_0008_S_80S`, `patch_0007_0001_S_80S`.
Sector ids derive as `ice-s-RRRR-CCCC` from the patch name **[repo: `ice_normalize.sector_id`]**.

## 2. Layers, names and expected shapes

Layer files present for every patch **[HF]**: `ASP_SIN_COS`, `CUR`, `DICE`, `LPSR`, `LPSR_DEN`,
`LPSR_DIS`, `ONLYCOS`, `SLOPE`, `TMAX`, and the target `PRO`.

Model inputs used by the released checkpoint **[repo: `ICE_INPUT_LAYERS`]**: `ASP_SIN_COS`,
`CUR`, `DICE`, `LPSR`, `LPSR_DEN`, `LPSR_DIS`, `SLOPE`, `TMAX`. `ONLYCOS` is the alternative
aspect encoding and is not an input of this checkpoint. `PRO` is the reference target (the
inference code stages inputs only and never copies `PRO` **[repo]**).

| Layer | Meaning (dataset README / repo) | Unit | Notes |
| --- | --- | --- | --- |
| `TMAX` | maximum temperature (Diviner-derived per upstream report) | K (upstream reports `maxTemperatureK`) | band stats mean ≈ 228 K **[HF]** |
| `LPSR` | permanently shadowed region indicator | 0..1 (upstream uses `>= 0.5` as shadow) | |
| `LPSR_DEN` | PSR density | **undocumented upstream** | |
| `LPSR_DIS` | distance to PSRs | **undocumented upstream** (mean ≈ 2 680) | likely metres; unverified |
| `SLOPE` | slope | degrees (upstream reports `slopeDeg`) | mean ≈ 8.4 **[HF]** |
| `CUR` | curvature | **undocumented upstream** | mean ≈ 0 |
| `DICE` | ice stability depth | **undocumented upstream** | |
| `ASP_SIN_COS` | aspect as sin/cos | dimensionless | **band count discrepancy**, see below |
| `PRO` | prospectivity target (Coyan et al. 2025 workflow), continuous 0–1 | index | evaluator-only in IceClaw |

**Shape/CRS (expected, [unverified] until a real file is opened):** 256 × 256 px at ≈ 240 m/px
(README); the pinned code requires all layers of a patch to share one CRS and grid, and the
published sector is 256 × 256 at 238.56 m/px in polar stereographic **[repo, P1]**. MOONATLAS
reads layers with `rasterio` and `.read()` yielding `(bands, H, W)`, treating no-data as
non-finite; coverage is "every input layer finite" **[repo: `coverage`]**. TIFF no-data tags,
exact dtypes and per-layer band counts must be read from the files in P4.

**Discrepancies to resolve in P4 (do not assume either side):**
1. README says "single-band GeoTIFF" and "11 layers / 1 782 files", but `band_statistics.json`
   lists `ASP_SIN_COS` with **two** bands and the file listing holds **10 layer names / 1 620 patch
   files**.
2. README says validation = 24 patches; `val_filtered.txt` lists 23.
3. `ONLYCOS` has band statistics under the key `COSASPECT` **[HF]**.

## 3. Proposed IceClaw modality mapping (for P4, not implemented)

| IceClaw layer name | SomBench file | Required unit check |
| --- | --- | --- |
| `pixelwise_temperature` | `TMAX` | `K` |
| `pixelwise_shadow_mask` | `LPSR` | `fraction` |
| `pixelwise_terrain` | `SLOPE` | `deg` |
| (later, once units are known) | `CUR`, `DICE`, `LPSR_DEN`, `LPSR_DIS`, `ASP_SIN_COS` | — |

The P2 thermal/terrain tools already refuse layers whose unit string differs, so undocumented
layers cannot silently feed a tool.

## 4. Getting predictions for the new patches (the expensive part)

The arrays alone are cheap; **predictions on them require the NASA–IBM checkpoint** unless
upstream already published them (only `ice-s-0001-0002` is published).

| Artifact | Size **[HF]** |
| --- | --- |
| `ni_lfm_ps8_all_modalities_s42.ckpt` (ice task, revision `fb0510d0…`) | 1.69 GB |
| backbone `backbone/checkpoint.pt` (revision `f29ade2b…`) | 2.43 GB |
| 11 tokenizer checkpoints in the backbone repo | ≈ 1.15 GB each (≈ 12.6 GB total) |
| MOONATLAS's own estimate for all models in smoke level | ≈ 6.3 GB **[repo README]** |

Which tokenizer files the ice task needs is **not established** (`official.load_task` was not
traced); MOONATLAS's smoke path downloads what it needs via `hub.fetch_models`. Both model repos
are public and ungated **[HF]**.

Compute: a separate environment with PyTorch + TerraTorch (pinned: torch 2.12.1, terratorch
1.2.13, lightning 2.6.6 per the processing manifest) is required; the upstream smoke run used a
CUDA GPU, the README says "GPU recommended, CPU works". This machine reports an RTX 5070 Ti
Laptop GPU (12 GB), 19 GB RAM and ~700 GB free disk; whether the pinned torch build supports this
GPU generation is **not tested**. Per-patch inference time upstream is recorded in
`processing-manifest.json` only as timestamps; no timing was measured here.

## 5. Scientific caveats for the data decision

- **Circularity:** the dataset README states the prospectivity target encodes assumed
  relationships between ice concentration and the *evidential layers*. The model is trained to
  reproduce `PRO` from those same layers. Agent claims that "inputs corroborate the prediction"
  are therefore not independent evidence; keep this in the report limitations (PROJECT_SPEC §7,
  Benchmark B).
- Only **test**-split patches may support held-out claims; validation/train predictions are
  in-sample (`docs/datasets/splits.md`). All 12 candidate south patches above are test split.
- Published outputs for new patches do not exist; any new region's `source_mode` would be
  `local_checkpoint_run` and must record the checkpoint SHA actually verified locally (the P2
  catalog currently carries only the upstream-published checkpoint SHA).

## 6. Recommended next data decision

1. **First priority (no checkpoint):** fetch the eight original input layers for **`patch_0001_0002_S_80S`**, which maps to the already verified published predicted sector `ice-s-0001-0002`, and open each GeoTIFF with `rasterio`. Resolve all band counts, dtypes, no-data masks, CRS and physical units. Verify pixel-center/affine/array alignment with the actual P1 prediction *before* enabling pixelwise thermal/geology claims. Only after this works, fetch the other south test observation packages (the previous ~25 MB estimate is metadata-derived, not an executed download).
   `PRO` is an **evaluator-only target**: if downloaded, place it under an isolated evaluator root, never in agent-visible modalities, prompts, science tools or event logs.
2. Only if checkpoint inference proves feasible on local hardware (≈ 6–15 GB download, pinned
   torch environment), produce predictions for the same patches for Benchmark B and the
   prediction auditor.
3. If inference is infeasible, mark extra regions `observation_ready=True` and `prediction_ready=False`. They may be inspected, but are **not** complete user-selectable prospectivity investigations. The prediction auditor and model-reference-disagreement Benchmark B cover only `ice-s-0001-0002`. A physical-layer audit without a matching prediction is a separate, explicitly labeled task; do not claim full four-specialist operation or substitute fabricated model predictions.
