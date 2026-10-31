# Evaluation protocol — P3 (prediction-only, preliminary)

Status: implemented in `src/iceclaw/evaluation/`; no LLM has seen any case. Everything below
describes what the code does, not results of agents.

## 1. Sealed boundary

| Visible to agents/runners (`<scenarios_root>/<suite>/<case>/`) | Evaluator only (`<eval_root>/<suite>/`) |
| --- | --- |
| `values.npy`, `covered.npy`, `meta.json` of an **opaque** case id (`case-NN`, `rank-01`) | `sealed_key.json` (fault id, category, parameters, seed, expectations), `salt.hex`, `reference.npy` |

* `AgentScenario` (case id + `AgentRegionContext` + `RegionArrays`) is the only object handed to
  an agent side. It carries no fault label, seed, reference, evaluator path or real tile id.
* The two roots must differ; `load_scenario()` receives only the scenarios root and works with
  the evaluator root deleted (tested).
* The science tools take arrays, never paths (tested by signature inspection).
* AST test: `iceclaw.science|data|agents|swarm|reports`, `contracts` and `config` never import
  `iceclaw.evaluation`; evaluation never imports agents/swarm.
* Sealed data are loaded only inside `score_*`, and only for a **frozen** output object
  (`FrozenDetections`, `FrozenRanking`) bound to the manifest hash.
* **Runner requirement for P7:** all cases are copies of one tile. Present each case to an
  agent in an isolated run; a run that sees several cases could use the clean one as a reference.

## 2. Frozen suite `p3-prediction-only-v1`

* Built by `build_suite()` from the authentic published prediction of `ice-s-0001-0002`
  (`source_mode=upstream_published_example`, split `test`); the source arrays are read-only and
  faults act on copies (tested: source hash unchanged).
* 12 detection cases: **3 clean controls** (identical copy; float32 round-trip; noise below the raw
  encoding's quantization step) and **9 synthetic faults**:

| Fault | Intent | Pre-registered detectable by baseline |
| --- | --- | --- |
| `value_unit_scale_x100` | percent/fraction conversion | yes (`value_range`) |
| `value_unit_scale_div100` | inverse conversion | yes (`dynamic_range`) |
| `value_nan_block` | NaN block in covered pixels | yes (`missing_data`) |
| `value_saturation_block` | block set to 5.0 | yes (`value_range`) |
| `value_offset_small` | +0.05 bias | **no** (stays in range) |
| `value_spatial_roll` | columns rolled (wrap seam) | **no** (seam ≈ natural gradients) |
| `metadata_footprint_shift` | footprint +0.4° latitude | yes (`geometry_consistency`) |
| `metadata_grid_origin_shift` | grid origin +25 px | yes (`geometry_consistency`) |
| `metadata_resolution_unit_error` | resolution ×0.1 | yes (`geometry_consistency`) |

  Expectations are written in `faults.py` before the baseline is run. They were written by the
  same author as the baseline, so they are a plumbing/sanity check, **not** an independent
  estimate of difficulty. No thermal/PSR/terrain faults exist yet (no pixelwise inputs; P4).
* Case order is a seeded shuffle; per-case seeds derive from `SeedSequence([suite_seed, 1, i])`.
  The suite seed is sealed. (The default seed is a constant in the repo: blindness relies on
  agents having no file access; use a private `--seed` for the official P7 suite.)
* The public `manifest.json` has per-case file SHA256s, the secondary-task definition, a **salted
  SHA256 commitment** to the sealed key and its own SHA256 (`manifest.sha256`). It has no
  timestamps, so the same source, seed and salt reproduce the same hash (tested).
* Never silently edit a frozen suite: `build_suite` refuses to overwrite; use a new `suite_id`.
  `verify_suite()` detects tampered case files, manifest, sealed key and reference.

## 3. Primary task: fault detection (Benchmark A, preliminary)

* Unit of analysis is the **case**. Metrics: TP/FP/FN/TN, precision, recall, F1, clean
  false-positive rate. An empty denominator gives `None` (undefined), never 0 or 1; no confidence
  intervals are reported (too few cases, one source tile).
* **Baseline** (`baselines.py`): reference-free rules over existing P2 tools —
  `missing_data`, `value_range`, `outside_unit_fraction`, `dynamic_range`, `geometry_consistency`
  (P1 projection code), `seam_line`. Thresholds live in `BaselineConfig` and its SHA256 is
  recorded. Flagged = any rule triggers.

## 4. Secondary task: reference-disagreement ranking (Benchmark B, preliminary)

* Candidate set: 64 non-overlapping 32×32-pixel windows of the untouched prediction (`rank-01`).
* Submission: exactly K=8 distinct window ids (a frozen ranking).
* Target (sealed): per-window mean |prediction − published expert reference|. Score: top-K recovery
  vs the K truly largest windows, random expectation K/N = 0.125, and the fraction of the best
  attainable residual captured (undefined if the best is 0).
* This measures agreement with **model/expert-reference disagreement**. The expert reference
  (`PRO`, Coyan et al. 2025 workflow) is not physical truth, and the model was trained on the
  same evidential layers; this says nothing about ice detection.
* Windows are spatially correlated pixels of one tile: results are exploratory.

## 5. Offline reproduction

```bash
python scripts/p3_build_and_score.py --scenarios-root data/scenarios --eval-root data/evaluation
python -m pytest -q
```

The script builds (or reuses) the suite, verifies integrity, runs the baseline, freezes its
output, scores it, and writes the report (which contains sealed labels) under the evaluator root.
