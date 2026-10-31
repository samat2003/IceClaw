# IceClaw — Canonical Project Specification

**Version:** 1.0 design freeze (2026-10-31)  
**Status:** PLAN ONLY — implementation, model inference, agent runs and benchmark results are not claimed.  
**Repository:** https://github.com/samat2003/IceClaw  
**Implementation roadmap:** [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

> **Document precedence:** This document governs scope, architecture, scientific semantics, data contracts, and acceptance criteria. The implementation plan governs sequence and commands. Earlier README and docs/SCIENTIFIC_PROTOCOL.md / docs/DEMO_2MIN.md are historical planning notes wherever they differ. No implementation step may silently change this specification.

## 1. Purpose and testable claim

IceClaw is a **reusable lunar ice-prospectivity research workbench**, not a detector of confirmed lunar water deposits. A user selects a scientifically covered lunar south-polar tile; independent tool-using AI specialists audit NASA–IBM predictions, test competing explanations using accessible lunar observations, and return a source-backed research assessment.

Primary research question: **Under a matched inference/tool budget, does evidence sharing between independently acting scientific agents improve audit quality and useful investigation prioritization relative to isolated or single-agent approaches?**

**Scientific boundaries:**
- An ice-*prospectivity* regression is an expert-index/model prediction, **not** a measured ice abundance, calibrated physical probability, confirmed water-ice deposit, or newly discovered site.
- Reference prospectivity values are not independent physical ground truth. Published LCROSS findings are context, not per-pixel labels for this benchmark.
- We distinguish model predictions, instrument-derived inputs, deterministic IceClaw computations, and LLM hypotheses with an explicit evidence-type field.
- Missing data are null/unavailable. No interpolation beyond documented native resolution, invented observations, or fictional agent traces.
- The NASA–IBM, SomBench and MOONATLAS projects retain their credits; no NASA/IBM endorsement of IceClaw.

## 2. The user experience (one Streamlit page)

1. **Choose region** from a catalog of *actually validated* supported polar tiles, showing coverage, source/data split, and dataset revision. Uncovered coordinates cannot be analyzed.
2. **Choose action:** Investigate (live) or inspect/replay a clearly labeled saved authentic run. A fixed standard investigation budget is the default.
3. **Inspect** a 2D lunar-polar-georeferenced tile heatmap, measured layer availability, agent evidence events and proposed follow-up experiments.
4. **Verify** one finding by rerunning a deterministic analysis against recorded raw inputs, then download the complete investigation bundle.
5. **Evaluate** system-level benchmarks (not artificially presented as statistics for an individual region). A second verified region must produce different region-specific numerical evidence.

A simple dropdown and plot are sufficient; no custom React app, 3D Moon, mandatory cloud hosting or user file upload for the MVP. **No data coverage means a clear 'not supported' response.**

## 3. Three upstream systems and one novel system

| Source | Responsibility | Integration boundary |
| --- | --- | --- |
| NASA–IBM Lunar Foundation Model | Pretrained ice-prospectivity task/checkpoint and input modalities | Batch inference during preparation, NOT a live model invoked on every click |
| MOONATLAS Science | Pinned acquisition, normalizing and geolocating raw prediction rasters, verifying checksums and data contracts | Run in a **separate pinned environment**; IceClaw reads its validated versioned exported artifacts |
| Microsoft Agent Framework (Python) | Tool-using AI agents, concurrent workflows, streaming and orchestration | Installed inside the IceClaw application environment; does not perform numerical science itself |
| IceClaw (ours) | Input selection, deterministic scientific tests, decentralized evidence events, comparison harness, verification, report, Streamlit UI | Single Python application and portable research outputs |

Upstream canonical references:
- https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model
- https://github.com/RRG1312/moonatlas-science
- https://github.com/RRG1312/moonatlas-science/blob/main/docs/provenance/data-contract.md
- https://github.com/microsoft/agent-framework
- https://scienceclawhack.ai/#competition

Do not fork/rewrite the upstream science packages or copy large datasets/checkpoints into this repository. Pin and record upstream revision/versions during the first working installation.

## 4. Input sources, coverage and data separation

**Supported first input:** the real MOONATLAS published smoke-example south-polar sector `ice-s-0001-0002`, approximately 81.45°S, 35.37°W (verify exact coordinates in the source metadata). This is a **published upstream example**, not a fresh IceClaw checkpoint inference. It supplies a normalized prediction, reference raster and source/geolocation/provenance metadata, but **may not supply all original pixelwise input modalities**. The smoke fixture alone is insufficient to demonstrate every thermal/geology tool.

For full specialist analysis, obtain original held-out SomBench ice-prospectivity sample arrays and their declared metadata; process through the official pinned NASA–IBM/MOONATLAS path. The published task uses roughly 256×256 tiles with about 240 m/pixel native spatial resolution and multiple physical layers. Exact channel naming, splits, shapes, and available files are discovered from pinned official configs, **never presumed**.

**Catalog inclusion rules:** accepted only after source, split, raw numeric rasters, coordinate grid, footprint, hashes and available modalities are validated. Include incomplete regions with missing-modality badges only if tools correctly disable unsupported analyses. Target at least **two distinct validated south-polar regions** for a reusable hackathon demo; if only one is available, label the feature partial rather than fabricate a second region or coordinate.

**Data isolation:** Reference prospectivity arrays and injected-fault manifests are accessible only to the evaluator *after* a ranking/run is frozen. Agent tools cannot read those paths or call evaluator functions. UI may reveal reference/residual layers only in labeled post-run evaluation mode.

**Raster rules:** Read numerical values exclusively from raw lossless arrays/encodings; display previews are visualization only. Preserve raw values even when regression values lie outside [0,1]. Respect lunar polar stereographic projection, native resolution, true per-tile footprints, no-data mask, units and source revisions. Do not treat nearest tile as coverage.

## 5. Canonical interfaces (all steps must implement these)

These are *logical contracts*; Python dataclasses/Pydantic models must enforce them. Their field definitions must not diverge between backend, UI, tests and exported JSON.

### RegionCatalogEntry

- region_id, pole, source_dataset, source_split, original_tile_id
- source_revision, processing_version, source_manifest_path, source_sha256
- lunar_crs, affine/grid geometry, footprint, width, height, spatial_resolution_m
- modalities: explicit set of present scientific layers + per-layer units/nodata/resolution
- prediction_raw_path, prediction_checkpoint_id, reference_path **(evaluation-only)**
- readiness: verified / partial / unusable with machine-readable reason

### EvidenceEvent (append-only)

- event_id, investigation_id, region_id, agent_id, timestamp_utc, event_kind
- parent_event_ids, claim_or_request, evidence_type (observation / model_prediction / computed_metric / hypothesis)
- source_artifact_ids, tool_id, tool_run_id, spatial_footprint, metric_value, unit, analysis_resolution_m
- hypothesis_status (provisional / challenged / supported / unresolved), alternative_explanations, limitations
- error_code and null fields when information is absent; never invent evidence values

Event kinds: `claim`, `measurement`, `challenge`, `test_request`, `test_result`, `revision`, `completion`, `failure`. A challenge must reference a prior event; a revision must reference the evidence which changed the assessment.

### InvestigationBundle

Under `runs/<investigation_id>/`:
- `input_manifest.json` (region, source/checkpoint hashes, provider/model, code revision, visible modalities, budget)
- `events.jsonl` (authentic ordered tool/agent events; replay this exact file only)
- `evidence.json` (validated claims and supporting source/tool records)
- `report.md` (machine-assistive scientific report with caveats and recommendations)
- `rankings.csv` (spatial target identifiers and *investigation priority*, NOT ice probability)
- `verification.json` (checksum, recomputation outcomes, missing sources, validation statuses)
- `plots/` (optional generated figures derived from numeric data)

**CLI contract**, to be implemented in stages:
- `python -m iceclaw.cli list-regions`
- `python -m iceclaw.cli investigate --region REGION_ID`
- `python -m iceclaw.cli verify --run RUN_ID`
- `python -m iceclaw.cli benchmark --config PATH`
The UI must call these same Python services rather than implement a second analysis path.

## 6. Scientific tools and four autonomous agents

Each tool is a deterministic, tested Python function returning structured measurements and provenance. Tool candidates: numeric raster summaries, temperature/PSR consistency, terrain slope/curvature checks, prediction boundary/discontinuity and missingness checks, raw-data validity and geometry checks. Unsupported modalities disable corresponding tools rather than fabricate substitutes.

| Agent | Distinct resources | Independently decides |
| --- | --- | --- |
| Thermal Scientist | Temperature / shadow/ice-stability data and tools (if available) | Thermal claims and next thermal check |
| Geology Scientist | Terrain / slope / curvature data and tools (if available) | Geological counter-explanations and tests |
| Prediction Auditor | Raw NASA–IBM predicted raster, model/run metadata | Which prediction artifacts are questionable |
| Skeptic / Investigation Planner | Published evidence and test-request tools | Challenges, counter-tests, research priority changes |

Use **Microsoft Agent Framework** with concurrent tool-capable agents, typed events to an append-only **SQLite** evidence store, and asynchronous delivery to interested agents. They can publish without a central LLM approving their work. A neutral supervisor enforces max events/hops, deduplication, tool budgets, timeout and final snapshot, **not scientific decision order**. Same initial region can produce genuine new investigation events on each permitted run.

Use one configurable frontier LLM provider/model for all roles at first (OpenAI, Anthropic, or Gemini as supported by an adapter proven to work). Model-ID/version is configuration, not hardcoded; failure to supply a key must produce a useful clear setup error, while deterministic analysis/replay still works.

## 7. Scientific verification and research evaluation

### Verification contract

An independent judge can select a claim and:
1. Identify source dataset/split/file and checksum, numeric values, CRS, units, modality and resolution.
2. Recompute its deterministic tool calculation from original numeric inputs with a declared numeric tolerance.
3. See original agent event(s), references, critiques, tool executions and subsequent revisions.
4. Run `verify` against the stored bundle without needing the original LLM response or GPU.
5. See clear separation of *source output*, *reproduced inference*, *IceClaw computation*, and *hypothesis*.

Authentic LLM text need not reproduce verbatim, but the original transcript and chosen tools must be retained.

### Benchmark A — blind data-quality / scientific-audit faults (primary)

On genuine held-out observation packages, create a declared suite of **synthetic** corruptions with hidden fault manifests: unit/conversion problems, spatial alignment shifts, altered raw values, missing modalities, and cross-layer inconsistency. Include clean negative controls. Never label a seeded fault as a real NASA anomaly. Record seeded operations, severity, independent sample/case IDs and random seeds. Report fault-detection precision/recall/F1 and false-positive rate at case level, with confidence limits only if sample size supports them.

### Benchmark B — real model-reference disagreement triage (secondary)

Agents inspect untouched permitted observations and model predictions, rank candidate areas **without seeing expert-reference maps**, then frozen rankings are evaluated against reference prediction residuals. Report top-K recovery / cumulative residual discrepancy found per tool budget. Interpret strictly as *model/expert-reference disagreement*, NOT physical water-ice detection performance. Avoid circular independent-evidence claims.

### Comparison arms, identical information and total budgets

1. Deterministic science/rules heuristic.
2. Single LLM agent with all allowed tools.
3. Independent specialist agents with **messages disabled**.
4. Communicating specialist swarm (IceClaw).

Hold fixed input cases, split, permitted tools, accessible modalities, total LLM-token/call/tool budget, temperature/seed where supported. Keep test manifests sealed until outputs are frozen. Report API cost, timing, unsupported claim rate and limitations. Different coverage or a single patch does not justify broad scientific-generalization claims.

## 8. UI, security, deployment and release criteria

**UI:** One Streamlit process: validated region selector; live/replay start; 2D polar region raster (not a fictitious global map); status/activity feed; evidence report; Verify/Recompute and Download buttons; benchmark summary from real result files. No React, separate FastAPI, Redis/Postgres, mobile app, 3D globe or arbitrary file upload in MVP.

**Secrets:** Codex CLI authentication is separate from IceClaw's model API keys. Keys enter the app through untracked environment settings / secret manager (one provider at first), never GitHub, prompts, logs or generated artifacts. Ship `.env.example` containing variable names/placeholders only, and ensure `.env`, data, checkpoints and runs are gitignored. Avoid reading or printing secret values when using Codex.

**Compute:** CPU + external LLM API sufficient for core app; optional H100 for offline checkpoint inference. Do not make live investigation depend on GPU availability. Python 3.11 recommended; keep upstream MOONATLAS/model execution in a **separate environment** from MAF/Streamlit to avoid conflicts.

**Completion gates:** (G1) real validated source, (G2) deterministic computed scientific result, (G3) at least two distinct selectable validated regions (or explicitly partial), (G4) communicating autonomous agents including a challenged and revised finding, (G5) repeatable verification bundle, (G6) measured matched-budget evaluation with honest results, (G7) a 2-minute live/replay demo + slides. One failed gate cannot be disguised by UI.

**Hackathon deadline:** 2026-11-01 13:00 America/New_York for final slides/end of hacking per official site. Presentation starts 14:30. 2-minute delivery is our chosen target, not a published competition-mandated time cap.

**Judging fit:** Problem significance 20%; scientific/technical impact 25%; decentralized agency 20%; collective capability 25%; execution/validation 10%; up to +10 collaboration points if another team materially reuses something.

## 9. Ownership, known risks, and scope control

- Source/geometry truth: MOONATLAS input contract and pinned source files; IceClaw never changes it.
- Numerical truth: deterministic Python science tools.
- Reasoning: LLM-generated proposals which do not override measurements.
- Experiment truth: blinded evaluator only.
- Presentation truth: saved outputs from real executed runs, no hardcoded favorable metrics.
- Main risk: one published smoke region with limited modalities is **not** enough for a real 4-specialist evaluation. Fetch test observations early; label support per region. If data access fails, preserve a smaller truthful demo and explicit limitations.
- No fine-tuning, bespoke scientific model training, arbitrary Moon coverage or full research paper before core validation.

**Build order and conflict-prevention policy:** See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). This spec is the single immutable acceptance contract until a documented, approved version change.
