> **ARCHIVED — superseded pre-implementation draft (kept for history).** It describes plans and placeholders, not results; some envisioned behaviour (for example agents challenging and revising each other) did not occur in the measured runs. For the current state see the [README](../../README.md), [PROJECT_SPEC](../PROJECT_SPEC.md) and [BENCHMARK_P7](../BENCHMARK_P7.md).

# IceClaw scientific protocol and integration contract

**Status: proposal, not executed.** This document defines the acceptance gates and prevents overstated science.

## Verified upstream properties

- NASA–IBM Lunar Foundation Model supplies inference and fine-tuning for its ice-prospectivity task.
- MOONATLAS Science v0.3.1 supports georeferencing/normalizing ice predictions and source verification. The real published example is in `examples/science-smoke`. See its `docs/provenance/data-contract.md`.
- Official polar ice task: 8 input modalities (9 channels), 256×256 patches at 240 m/pixel, native lunar polar stereographic projection. Raw regression outputs may fall outside [0,1], and must never be silently clipped for scientific analysis.
- The target is a reference *prospectivity* map, not ground-truth ice presence. A good RMSE to the expert reference does not demonstrate water discovery.

## Concrete first input

Use an exact, verified MOONATLAS published example:
- `ice-sectors.json` holds region/pole, footprint and grid, source tile, projection, anchor measurements, predicted raster paths and provenance.
- `ice-mosaics.json` holds the mosaic raster references and grid.
- `overlays/ice/*-raw.webp` are losslessly encoded numeric rasters with offset/scale metadata; `*-display.webp` are *only* visuals.
- `sources.json`, `processing-manifest.json`, `manifest.json` identify the build and upstream source revisions.

The published example includes a south-polar patch `ice-s-0001-0002`; its center is approx 81.45°S, 35.37°W. **Do not call it Cabeus without spatial verification.**

For full science: download SomBench held-out source modalities and run a pinned NASA–IBM checkpoint through MOONATLAS. Only report successful runs backed by generated logs and verified checksums.

## Proposed IceClaw I/O JSON (new contract)

```json
{
  "investigation_id": "uuid",
  "region_id": "ice-s-0001-0002",
  "dataset_split": "test",
  "dataset_revision": "pinned-source-sha",
  "model_checkpoint": "pinned-checkpoint-sha",
  "processing_version": "moonatlas-science-version",
  "grid": {
    "projection": "lunar-polar-stereographic",
    "width_px": 256,
    "height_px": 256,
    "resolution_m": 240
  },
  "evidence": [],
  "hypotheses": [],
  "counterarguments": [],
  "next_observations": [],
  "status": "pending",
  "generated_artifacts": []
}
```

Each evidence event must include agent_id, timestamp, event type, region/footprint, claim, cited source product, quantity + units or null, measurement resolution, confidence interpretation, data dependency, and code/tool run id. Assertions without traceable support must be rejected by the report validator.

## Agent interaction

- Thermal scientist: checks thermal/PSR consistency using deterministic tools.
- Geology scientist: checks relief, roughness and slope effects.
- Prediction auditor: examines model output, missingness, edges, potential systematic errors.
- Skeptic / experiment planner: proposes counter-hypotheses and selects high-information follow-up analyses.

Microsoft Agent Framework supplies tool-using agents and concurrent workflows. Publish immutable events to an append-only SQLite store and stream them to the UI. Agents independently consume new events and can publish new tasks. Scheduler transports events, enforces budgets and terminates runs; it does **not** prescribe the scientific reasoning path.

Prevent infinite chatter via event deduplication (hash of region+claim+test), per-agent budgets, max follow-up depth, and a final freeze/snapshot point.

## Evaluation: avoid leakage and circular science

**Separate two things.**

1. *Model prospectivity reproduction:* compare NASA–IBM raw predictions to the held-out expert-reference regression raster using RMSE/MAE and residual maps. This is a model-to-reference consistency test, not a water-ice detection benchmark.
2. *Collective scientific capability:* hide reference rasters from all agents, provide the same permitted observations/tools and fixed inference budget, then compare how each approach prioritizes locations whose published model outputs show substantial held-out reference disagreement. Compute precision@K / cumulative reference error found by inspection budget, tool calls, cost, and unsupported-claim count. Reference is revealed ONLY to the evaluator after rankings are frozen.

Compare:
- deterministic risk/uncertainty heuristic;
- single agent with all specialist tools;
- agents acting independently without inter-agent messages;
- autonomous communicating swarm.

Limitations: expert reference shares scientific assumptions/modalities with model training, and is not independent proof of ice. Report any overlap explicitly. Use perturbation controls (with known injected errors and missing modalities) as an additional *software robustness* experiment, separately labeled synthetic.

## Scientific integrity acceptance gates

- NASA checkpoint and split IDs recorded; train/val data never described as held-out.
- Raster pixels are derived from raw numeric representation, never display colors.
- Lunar CRS / footprints / resolution correct and individually verified.
- Null/missing modalities never silently invented.
- Model prospectivity values not called probabilities or deposits.
- Reference maps are not presented as actual ice detections.
- Agent conclusions cite deterministic measurements and include alternative explanations.
- Include NASA/IBM, SomBench and MOONATLAS attribution; obey their Apache-2.0, CC BY 4.0 and trademark provisions.

## Sources

- https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model
- https://github.com/RRG1312/moonatlas-science
- https://github.com/RRG1312/moonatlas-science/blob/main/docs/provenance/data-contract.md
- https://github.com/microsoft/agent-framework
- https://scienceclawhack.ai/#competition
