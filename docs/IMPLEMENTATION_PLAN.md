# IceClaw — Engineering Implementation Plan

**Revision:** 3.1 · 2026-10-31  
**Status:** P0–P8 implemented and committed (360 offline tests pass, 1 opt-in live skip). One genuine live four-specialist investigation is recorded, verified offline and published as a public-redacted bundle. P7 was executed once under a pre-registered protocol with a negative, exploratory result (communication did not improve detection, completeness or efficiency on six synthetic cases from one tile; see docs/BENCHMARK_P7.md). Remaining manual check: one live investigation started from the Streamlit UI.

**Product and scientific authority:** [PROJECT_SPEC.md](PROJECT_SPEC.md).  
**Real-data feasibility:** [SOMBENCH_FEASIBILITY.md](SOMBENCH_FEASIBILITY.md).  
**Existing validated upstream artifact:** [UPSTREAM_MOONATLAS.md](UPSTREAM_MOONATLAS.md).

This is a **software engineering specification**, independent of the editor, terminal, or development assistant. It defines implementation order, module ownership, interfaces, tests, scientific gates, and acceptable degradation modes. Do not add features outside its milestones.

## 1. What already works — DO NOT REBUILD

| Milestone | Status | Implemented repository files | Scientific reality |
| --- | --- | --- | --- |
| **P0 — Core package and contracts** | DONE (committed) | `pyproject.toml`, `config.py`, `contracts.py`, `cli.py`, foundational tests | Pydantic models, CLI placeholders, redacted settings |
| **P1 — Genuine MOONATLAS artifact** | DONE (committed) | `data/moonatlas_adapter.py`, adapter tests and upstream validation documentation | Pinned upstream commit `945fb5196679cf6927354c53be624d745083f9e9`; published sector `ice-s-0001-0002`; raw numeric prediction; validated geolocation, statistics, checksums |
| **P2 — Catalog and science tools** | DONE (committed) | `data/catalog.py`, `data/validation.py`, `science/`, `services.py`, catalog/science tests | 10 deterministic tools, `ToolMeasurement`, reproducible tool IDs, `list-regions`; 94 tests reported passed at P2 |
| **P3 — Evaluation isolation** | DONE (committed) | `evaluation/{fixtures,faults,baselines,score}.py`, `docs/EVALUATION_PROTOCOL.md` | Sealed, reproducible prediction-only suite; evaluator truth isolated; preliminary baseline |
| **P4 — Real original observation pairing** | DONE (committed) | `data/sombench_adapter.py`, observation catalog, multimodal evaluation | One prediction+TMAX/LPSR/SLOPE fully paired; 11 observation-only tiles; 160 tests reported passed at P4 |
| **P5 — Genuine agent runtime** | DONE (committed) | `agents/`, `swarm/`, `docs/AGENT_RUNTIME.md` | One authentic four-specialist run, 22 model calls / 15 scientific tools / 24 events; cross-agent follow-up with no warranted challenge |
| **P6 — Service, bundles, verifier** | DONE (committed) | `services.py`, `reports/{export,package,verify}.py`, `docs/INVESTIGATION_BUNDLE.md` | Live P6 run (19 calls, 74 991 tokens) exported, verified offline (23 checks) and published public-redacted (`iceclaw-bundle/2`, 26 checks); token reservations; replay |
| **P7 — Comparative benchmark** | DONE (this release) | `benchmark.py`, `evaluation/paired.py`, `agents/benchmark_arms.py`, `benchmarks/`, `docs/BENCHMARK_P7.md`, `docs/p7_results/` | v1 suite + one-case pilot; pre-registered v2 suite (6 cases) executed once: rules F1 0.857, generalist 0.667, isolated 0.857, communicating 0.667 (1 incomplete case); 213 calls, 815 736 tokens; no challenge/revision; negative, exploratory |
| **P8 — Streamlit workbench** | DONE (committed) | `app.py`, `ui/`, `docs/WORKBENCH.md` | Real layers, recorded replay, verify, public download; live path tested offline; live UI click not yet exercised |

**P2 input limitations, confirmed by the implementation:** The smoke sector is **partial**: 256×256 decoded model predictions at **238.56 m/pixel**, plus a *tile scalar* PSR fraction and *single-anchor* slope, temperature and shadow metadata. **It has no pixelwise thermal/shadow/slope rasters**. Current thermal and geological tools correctly return `UNAVAILABLE_MODALITY`. Do not treat one anchor as 65,536 observations.

The measured 825 values outside [0,1] are preserved as raw model regression values. Prediction discontinuities/gradients are **model-output screening metrics**, not actual ice discoveries.

**Current Git policy:** P0–P8 are committed. Keep the user's existing `.env` untouched. `data/` and `runs/` (sources, sealed evaluation material, run logs) stay outside Git; only scored public benchmark summaries, configs, registrations and frozen-output hashes are tracked.

## 2. Final, fixed architecture

| Boundary | Choice | Rule |
| --- | --- | --- |
| Main application | Python 3.12 (P0 used 3.12.3) | Preserve existing `src/iceclaw/` code and contracts |
| Data preparation | Pinned MOONATLAS Science + SomBench / NASA–IBM | Separate `.upstream/` virtual environment; checkpoint inference is optional and offline |
| Scientific numerics | Existing NumPy tools + Rasterio / pyproj when required by original GeoTIFFs | Deterministic and unit-tested, no LLM-made numeric claims |
| Agent framework | Microsoft Agent Framework | Real four-role tools and asynchronous evidence exchange |
| Reasoning provider | OpenAI using `agent-framework-openai` | Explicit verified model ID via `ICECLAW_LLM_MODEL`, API key via local runtime only |
| Event persistence | SQLite `sqlite3` + JSONL export | Append-only records, parent-event references |
| Application services | Existing `iceclaw.services` | CLI and UI call **the same functions** |
| Interface | One Streamlit page + Plotly | Select validated region, run, inspect, verify, export |
| Benchmark evaluator | Separate `iceclaw.evaluation` | Hidden targets and fault labels must never reach agents |

**Not part of MVP:** FastAPI, React/Next.js, Redis, a database server, multiple hosted services, fine-tuning, a 3D Moon or arbitrary file uploads.

**Official agent integration:** Microsoft documents `pip install agent-framework-openai` and `from agent_framework import Agent; from agent_framework.openai import OpenAIChatClient` (Responses API). Verify the exact installed package and minimal tool-call behavior rather than assuming an old API. Source: https://learn.microsoft.com/en-us/agent-framework/agents/providers/openai.

## 3. Import boundaries and contracts

```text
CLI / Streamlit app
        │
        ▼
iceclaw.services ────────> iceclaw.reports (verification and export)
        │                           ▲
        ▼                           │
iceclaw.swarm ────────> iceclaw.science (deterministic tools)
        │                           ▲
        ▼                           │
iceclaw.agents              iceclaw.data/catalog
                                    ▲
                         validated upstream artifacts

iceclaw.evaluation (hidden answers, fault key, scoring)
            │
            └──── reads frozen run outputs ONLY; never imported by agents/swarm
```

### Existing types (reuse, do not replace)

- `RegionCatalogEntry`: **private** full record containing model prediction, original source, scientific grid, modality availability and evaluator-only `reference_path`.
- `AgentRegionContext`: **explicit allowlist**, containing no reference path, truth labels, fault descriptions, arbitrary filesystem paths or credentials. Already has `RegionCatalogEntry.to_agent_context(...)`.
- `ToolMeasurement`: deterministic calculation, tool ID, parameters, source IDs/hashes, declared units, numeric result or unavailable reason, tolerance and reproducible tool-run ID. **Preserve the existing P2 record format**.
- `EvidenceEvent`: event ID, time, agent, run, region, event kind, parent IDs, evidence/tool IDs and optional supported metric/limitations; `challenge` / `revision` require parents.
- `InputManifest` and `InvestigationBundle`: declared source and model revisions, provenance and saved run files.

### Important additional rules

1. **Evaluator firewall:** The `PRO` expert target (and synthetic-fault truth) is stored under an evaluator-owned root, never under agent-visible input root. Agents must not receive `RegionCatalogEntry` or general-purpose file-reading tools; they receive `AgentRegionContext` and approved science-tool names/arguments only. No evaluator imports from agent/science modules.
2. **Existing one-unit-per-measurement schema:** P2 science tools sometimes return dictionaries containing fields with different units. Keep their suffix-based field documentation for compatibility, but any downstream claim, plot, scoring or report must carry/derive the correct unit **per displayed field**. A single `ToolMeasurement.unit` must not be attached indiscriminately to all dictionary entries. Prefer separate scalar `EvidenceEvent` measurements when ambiguity exists; only migrate the contract if necessary with all tests/docs updated.
3. **No false coverage:** A tile containing only observations does **not** have a NASA–IBM prediction. Record separately `observation_ready` and `prediction_ready` in catalog diagnostics/service data. Only offer complete prospectivity investigations when both exist and are spatially registered. Missing modality gives `UNAVAILABLE_MODALITY`, never a filled array.
4. **Raw numerical integrity:** Preserve values outside [0,1], masks, nodata, original TIF dtype/band count, scientific source checksum, native lunar polar CRS, pixel-center coordinate and affine grid. Do not read preview WebP for science. Do not silently resample/reproject to force alignment.
5. **Source attribution:** Distinguish upstream published prediction, IceClaw fresh prediction (only if actually executed), physical observation, derived science statistic and LLM hypothesis.

## 4. Ordered implementation phases

### P3 — Sealed evaluation boundary and preliminary deterministic benchmark

**Prerequisites:** committed P2.  
**Owns:** `src/iceclaw/evaluation/{__init__,fixtures,faults,baselines,score}.py`, tests, optionally `docs/EVALUATION_PROTOCOL.md`. No agent runtime, downloads, UI or LLM calls.

**Implement in this order:**

1. Implement physically separate **agent-visible scenario** and **evaluator-only sealed answer** representations. No general-purpose `Path` / directory access in science/agent callable tools. Include a test that `PRO`, `reference_path`, fault labels, seed and evaluator file paths never appear in serialized `AgentRegionContext`, tool-call arguments or agent-visible JSON.
2. Establish **case-level** split/ID, input provenance, permitted observation modalities, tool budget, fault category, case label, independent data source and repeat/seed. Freeze a versioned case manifest and SHA256 *before any LLM benchmark run*.
3. Create **copies** of the authentic P1 prediction raster or its metadata. Produce: at least one clean control, a controlled prediction perturbation, and a metadata/geometry corruption. Document injection intent, magnitude, and whether a given fault is detectable via the available P2 tools. Do not insert thermal/PSR or geological faults yet: those require P4 data.
4. Build deterministic rule baseline with the existing P2 tools and case-level scoring for fault detection: TP/FP/FN/TN, precision, recall, F1, clean false-positive rate. Handle undefined denominators explicitly; don't silently report perfect scores.
5. Define the **secondary** reference-disagreement task, evaluation-only scoring interface, candidate-window ranking and top-K recovery; do not make reference scores available before the rankings are frozen. Do not claim expert prospectivity is physical truth.
6. Tests: no leakage, original input immutability, clean negative control unchanged, deterministic seeded faults, stable case IDs/SHA, expected baseline metrics, no correlated-pixel pseudo-replication.

**Acceptance:** `python -m pytest -q` and Ruff pass; a frozen *prediction-only* preliminary case suite plus baseline scoring can be reproduced offline; hidden truth cannot reach agent-facing types.

**Boundary:** P3 establishes the evaluator firewall. Full physical-modality benchmark cases are added/frozen in **P4**, before P5's actual agents see any benchmark data. Do not alter the frozen P3 cases silently; use an explicitly versioned new suite.

### P4 — Acquire real SomBench observations and pair them with verified predictions

**Prerequisites:** P2 and P3. **This is the data-acquisition work moved forward from old P6.**  
**Owns:** `src/iceclaw/data/{sombench_adapter,validation,catalog}.py`, `src/iceclaw/science/{thermal,terrain}.py` only as required, `scripts/prepare_sombench.py`, data tests and `docs/SOMBENCH_INGESTION.md`. Must not change `moonatlas_adapter.py` or P2 formulas unless a failing integration test establishes a real bug.

**Pinned public upstream inputs already discovered in P2:**

- Hugging Face dataset: `nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression`.
- Dataset revision: `78040503ceaa68107829f48114774112e8ee940d` (verify pinned hash still resolves). MOONATLAS commit: `945fb5196679cf6927354c53be624d745083f9e9`.
- `test_filtered.txt` declares 12 south-polar held-out patches; `patch_0001_0002_S_80S` corresponds to `ice-s-0001-0002`. Every downloaded patch must be verified independently against the split manifest.
- Eight model-input file types: `ASP_SIN_COS`, `CUR`, `DICE`, `LPSR`, `LPSR_DEN`, `LPSR_DIS`, `SLOPE`, `TMAX`; `ONLYCOS` is a nonselected alternative aspect file. `PRO` is **evaluation-only target**, not a model input.
- P2 estimates the south test inputs are a manageable download; validate real sizes/dtypes/band counts before claiming them. Original data licensing/provenance recorded; no raw scientific files in Git.

**Implementation and tests, in order:**

1. **Download first the original eight input GeoTIFFs for `patch_0001_0002_S_80S`** from the pinned dataset (via `huggingface_hub.hf_hub_download(..., repo_type="dataset", revision=...)` or the official MOONATLAS helper). Hash them and record exact relative path, dataset revision, source split and license. No checkpoint download in this step.
2. Inspect `rasterio` shapes (all bands, dimensions and dtype), CRS/projection, affine transforms, nodata, pixel masks, min/max and unit metadata. Resolve the documented `ASP_SIN_COS` two-band discrepancy, `ONLYCOS` name mismatch and unclear `CUR` / `DICE` / `LPSR_DEN` / `LPSR_DIS` units from actual data and upstream source where possible. If a unit cannot be established, label unknown and disable physical interpretations relying on it.
3. **Pair exactly with the P1 prediction:** derive the mapping from the pinned upstream patch-to-sector function, verify the source split and tile ID, validate lunar CRS, pixel-center locations, **array orientation, affine transforms and footprint** against the MOONATLAS prediction. No silent resampling; if native grids differ, record the mismatch and implement a scientifically justified, explicitly tested reprojection stage only if necessary. Unpaired inputs cannot enable physical inference.
4. Extend `RegionArrays.layers` and `RegionCatalogEntry.modalities` to include only verified **pixelwise** `TMAX` (K), `LPSR` (mask/fraction with documented threshold), `SLOPE` (degrees) and other layers with correctly established units. Tile scalar `psrFraction` and single-anchor values remain metadata, not pixelwise arrays. Use stable raw SHA in P2's `ToolMeasurement`.
5. Execute the existing `psr_temperature` and `terrain_slope` tools against **real source arrays** on the paired `ice-s-0001-0002` sample; verify with independent NumPy computations and explicit units/masks. P2's prediction-only tools and known measurements must remain unchanged.
6. Obtain more authentic held-out south-polar inputs (ideally all 12). **Observation-only tiles are not fully supported investigation choices**: unless a verified NASA–IBM predicted raster also exists for that tile, mark them `observation_ready=True, prediction_ready=False`. Acquire additional predictions through documented pinned MOONATLAS/NASA batch inference **only if** checkpoint dependencies and hardware allow it; record checkpoint SHA against actual downloaded file if run. RTX 5070 Ti / 12GB feasibility is not yet tested; a full model run is optional.
7. Extend the sealed P3 experiment protocol with a **new frozen, versioned multimodal suite** containing realistic physical-layer corruption and clean controls, before any P5 live-agent benchmarking. Store `PRO` targets only under evaluator control; do not expose raw test split PRO files to the agents.
8. Tests: exact split/source identity; real band count/dtype/nodata/CRS/masks; alignment/row/col geography; P2 regression; realistic source-unit mismatches; no label leakage; mutually exclusive ready/missing statuses; unexpected layer names cause typed errors.

**Acceptance:** **Mandatory:** at least one **same-tile** NASA prediction + authentic pixelwise TMAX/LPSR/SLOPE observations + real numerical thermal/terrain results with source SHA and verified alignment, and P3 firewall still passes. **Target:** two or more distinct verified predicted-and-observed south-polar regions for user selection. If only the first can be paired, label coverage `PARTIAL_ONE_REGION`; do not fabricate model output or label observation-only regions as research-ready.

**Non-goals:** model retraining, invented "ice probabilities", full set of global lunar coordinates, silently using PRO as a source input.

### P5 — Genuine Microsoft Agent Framework specialist collective

**Implementation status:** DONE (committed). The live run is evidence of authentic cross-agent follow-up, not a live challenge→revision or an improvement over a baseline. Preserve the captured authentic event log and model usage. No reason to force a scientifically unwarranted challenge.

**Prerequisites:** P3 **and P4 mandatory same-tile data gate**.  
**Owns:** `src/iceclaw/{agents,swarm}/`, agent integration tests. No UI, no benchmark scoring.

1. Install compatible `agent-framework-openai` dependency and pin resolved versions. Verify one MAF `Agent(client=OpenAIChatClient(model=...))` can invoke an approved local deterministic scientific tool. Model comes from runtime config; never guess a model ID. One model/provider for all roles at first.
2. Implement separate **Thermal**, **Geology**, **Prediction Auditor**, **Skeptic/Experiment Planner** agents. Each receives only `AgentRegionContext` plus explicitly registered source-valid tools; the thermal and terrain specialists now have real data to inspect.
3. Run agents concurrently. Persist append-only `EvidenceEvent` to SQLite; asynchronously deliver genuinely generated messages. A reacting agent may decide to run a new test, issue a challenge, or revise a claim, with linked parent IDs. There is no central LLM ordering the scientific steps.
4. Implement bounded `RunBudget` (events, model calls, tools, time, hops), event dedup, exception handling, cancellation and retry behavior. A skeptical agent can report no warranted contradiction.
5. Offline tests with stubbed model replies check actual MAF function-tool routing, event linkage, distinct agent IDs, budget ceilings, secrecy of evaluation root and deterministic source/tool records. Mark live API tests separately; a mock does **not** prove live API operation.
6. Execute one genuine OpenAI-backed run on the now multimodal paired tile, with real cited numeric measurements and saved original interactions. Distinguish previous hypotheses from later evidence-based revisions; never fabricate a debate if agents agree.

**Acceptance:** a real source-backed live four-specialist investigation, at least one observed cross-agent evidence-triggered follow-up where scientifically warranted, and a machine-verifiable event history. If no challenge occurs naturally, it remains an honest unresolved/no-conflict result, not a contrived success. All fast tests and Ruff pass. **Reported P5 evidence:** authentic Skeptic follow-up based on Prediction Auditor's peak, 22 model calls, ~91k tokens; no observed live challenge or revision. Treat the live trace as evidence-sharing, not measured superiority. The recorded SQLite/JSONL run remains the source of truth.

### P6 — Shared investigation service, saved report and offline verifier

**Prerequisites:** P5.  
**Owns:** `src/iceclaw/services.py`, `src/iceclaw/reports/{export,verify}.py`, CLI implementations and tests. Do not duplicate scientific calculations.

1. Create `InvestigationService`: validated region input, `RunBudget`, live/replay mode and unique run ID; use the same service from CLI and UI.
2. Write exactly `runs/<id>/{input_manifest.json,events.jsonl,evidence.json,report.md,rankings.csv,verification.json}`, optional plots and a **minimal permitted source-input snapshot** for independent recomputation. Include actual source checksums, model checkpoint source mode, original pixelwise inputs, source SHA, code revision, model version, tool params/units and uncertainty. The prior P5 SQLite tool/event history must be exported **without reconstructing or inventing activity**. Preserve P5's recorded live run as an authentic replay fixture; it may be incomplete until P6 exports the full contract.
   **Portability gate:** A bundle advertised as *offline portable* must contain the exact non-evaluator source arrays needed by its deterministic tools, with original source SHA and license attribution, or explicitly identify and verify a separately supplied content-addressed source package. Otherwise label verification as *requires local source data* instead of falsely claiming that an isolated downloaded ZIP can recompute its numbers. Never bundle `PRO`, fault labels, `.env` or absolute credential-bearing paths.
3. Scientific report sections: question, available inputs, observed numeric measurements, model-prediction inconsistencies, alternative explanations, outstanding uncertainties, and **research follow-up priority** (NOT physical ice probability). For mixed-unit dictionaries preserve correct per-field unit labels or produce separate scalar evidence.
4. Offline `verify --run ID` recomputes real deterministic tool measurements from exact source files within declared per-tool tolerances. Detect tampered sources, inconsistent grid/units, missing event parents and mismatch between cited measurement and recorded result.
5. Authentic replay reads the saved events only, always displays `recorded`, and makes no new agent calls.
6. **Token/cost hardening before benchmark:** P5's first live run consumed ~91,049 tokens across 22 model calls and 24 events, with growing Skeptic context. Implement a deterministic bounded evidence digest for new prompts (recent relevant event IDs + cited compact tool values, retaining full history in SQLite), max prompt tokens/calls per investigation and model-usage accounting from actual provider responses. No fabricated token price: cost remains `unpriced` until an independently verified model rate is configured. Test that bounded context does not drop required source IDs/parent references. Preserve the prior full P5 trace unchanged.

**Acceptance:** live generated bundle passes offline verifier; tampered source/metadata or fabricated event parent fails and returns nonzero exit. CLI `investigate`/`verify` both call `InvestigationService`. A second independent Python process (no LLM key, no GPU) can verify using the declared source package. Prompt growth and per-run token usage are bounded and recorded.

### P7 — Comparative scientific benchmark on frozen cases

**Prerequisites:** P3, P4, P5 and P6.  
**Owns:** a thin **composition runner outside `iceclaw.evaluation`** (e.g., `src/iceclaw/benchmark.py`) plus `iceclaw.evaluation.score`, benchmark tests and actual result artifacts. **Do not break P3's AST/import firewall:** evaluator code cannot import `agents` or `swarm`; the composition runner can invoke an agent via `services` and pass only a public `AgentScenario`, then submit frozen outputs to the evaluator's scoring functions.

1. **Resolve scenario eligibility before any four-arm comparison.** The P3 prediction-only cases carry no pixelwise TMAX/LPSR/SLOPE, and 11 of P4's 12 multimodal cases have observations but no verified NASA prediction. These are useful for *component/tool robustness*, not a full-four-specialist end-to-end benchmark. Create and freeze a NEW, clearly versioned **paired-tile case suite** using copies of the one verified predicted+observed tile, with clean controls and independently declared faults. The production runtime currently accepts only paired catalog regions; provide a dedicated, strictly allowlisted **scenario execution adapter** that supplies a sealed case's `AgentRegionContext` and `RegionArrays` without bypassing the source/input firewall. Never hand an agent a catalog record, evaluator reference, hidden labels or arbitrary file paths. Ensure a new isolated run, conversation and tool state for every opaque case ID, with no cross-case context memory.
2. Run four systems on exactly the **same eligible case suite** and declared total budgets: deterministic rule baseline, one agent with all authorized tools, four specialists **without** inter-agent messages, and four communicating specialists. P3 prediction-only and P4 observation-only suite scores may also be reported as **separate, limited ablations**; do not conflate these with a full four-specialist comparison.
3. Keep identical test cases, total model/tool budget, source version, scientific tool set, reference access restrictions and measured runtime. Do **not** allocate each of four agents the same full budget granted to a single agent. Before committing to a full suite, run **one paid case per LLM arm** as a cost/latency feasibility pilot, then lock the affordable number of cases and budgets. Current swarm consumption of ~91k tokens / 22 calls per live investigation would imply ~1.09M tokens and 264 model calls across 12 comparable full-swarm cases alone if unchanged; do not assume an unverified dollar price. Prefer fewer honest matched cases to an unbounded or unequal-budget comparison.
4. Freeze each system's detection flags/prioritized region list and original run logs **before** evaluator-only access to fault truth or `PRO` reference. For synthetic faults compute case-level precision, recall, F1, clean false-positive rate. For untouched source cases compute residual-disagreement top-K recovery, explicitly stating this is not physical ice ground truth. `ObsBaselineConfigV2` was defined after identifying the v1 clean-noise false positive; a new suite/config must be locked in advance, and its result must **not** be advertised as independent confirmation of a threshold chosen from prior failed controls.
5. Export CSV/JSON evidence for all case-level counts, costs and caveats. Use repeated cases/seeds if feasible; small geographic samples preclude broad statistical claims. Report a lack of benefit or regression honestly. One paired source tile and derived fault copies do **not** establish performance generalization to the Moon; do not infer statistical independence from the number of synthetic cases.

**Acceptance:** at least one real measured matched-budget comparison with reproducible scoring code and all individual case outcomes. Never use invented benchmark scores, artificial model replies or mock-agent outputs in the submitted comparison.

### P8 — Minimal Streamlit workbench and final demo

**Prerequisites:** P6; incorporate actual P7 results if completed. **Scheduling note:** to protect the judged live demo, build the smallest functional P8 Streamlit page **immediately after P6** and BEFORE launching a large paid P7 benchmark if time is tight. Later add benchmark results to the existing page without changing the source/service path.  
**Owns:** `app.py` and UI tests/presentation docs.

1. Single page: validated region selector, status/coverage and modality badges; Plotly heatmap from georeferenced raw model raster; start live investigation; authentic events and report; Verify/Recompute button; download saved bundle.
2. Show a source-supported map in lunar stereographic coordinates, not an Earth basemap or unsupported complete Moon. Only offer active investigation on **fully paired** region records; allow inspection-only view for other catalog entries with explicit readiness.
3. Show saved benchmark results and their case-level validation only when real data files exist. Replay is explicitly marked `Recorded investigation`; never present it as a live run.
4. UI calls the existing service and verifier only; test multiple actual region IDs if P4 produced them. Reject unsupported selections clearly.
5. Presentation uses one real scientific investigation, one authentic collaboration trace, one verifier action and one genuinely measured comparison. Up to five slides; factual caveats and output source IDs visible.

**Acceptance:** `python -m streamlit run app.py` opens in WSL; selected real region triggers actual run, shows provenance-backed measurements and verified downloadable bundle. Two-region demonstration is a target; explicitly declare `PARTIAL_ONE_REGION` if unavailable.

## 5. Exact commands and readiness matrix

Use the existing project virtual environment for the IceClaw app; preserve the separate upstream environment for MOONATLAS.

```bash
cd ~/IceClaw
source .venv/bin/activate
python -m pytest -q
ruff check .
ruff format --check .
python -m iceclaw.cli list-regions --json

# P6+, after successful P5 live agent run:
python -m iceclaw.cli investigate --region ice-s-0001-0002
python -m iceclaw.cli verify --run RUN_ID

# P7+, only with a frozen manifest containing no answer keys:
python -m iceclaw.cli benchmark --config PATH_TO_CONFIG

# P8+:
python -m streamlit run app.py
```

**Upstream source validator**, from its own virtual environment:

```bash
cd ~/IceClaw/.upstream/moonatlas-science
source .venv/bin/activate
python -m pytest -q
moonatlas-science validate --root examples/science-smoke --scope published
```

Do not mistake `--scope published` for a fresh NASA checkpoint inference test.

| Gate | Required evidence |
| --- | --- |
| **G0 P0–P4 (DONE committed)** | P0/P1 real source, 94 P2 tests reported, sealed P3 suite, paired P4 TMAX/LPSR/SLOPE, 160 P4 tests reported |
| **G0b P5–P8 (DONE committed)** | 360 tests pass, 1 opt-in live skip; four genuine MAF specialists; verified P6 bundles; pre-registered P7 v2 executed once (negative, exploratory); Streamlit smoke test passed (live UI click pending) |
| **G1 P3** | Prediction-only sealed benchmark case suite; privacy/firewall regression tests |
| **G2 P4** | Authentic, spatially aligned TMAX/LPSR/SLOPE TIFFs from matched tile; test on pixelwise data; second region target |
| **G3 P5** | Real Microsoft Agent Framework event trace and valid agent tool executions |
| **G4 P6** | Complete bundle, offline recomputation, source tamper detection |
| **G5 P7** | Fair comparison with frozen scores/case outputs, no leaked labels |
| **G6 P8** | Functional single-page Streamlit + real outputs and optional authentic replay |

### Phase acceptance report

For each phase record: exact changed modules, commands actually executed and test counts, dependency/commit SHA, source IDs/hashes and new evidence artifacts, no-secrets check, `DONE/PARTIAL/BLOCKED` and precise remaining limitation. A passing mock test does not establish a successful live API run; a published fixture does not establish a new model-inference run.

## 6. Time-critical scope and scientific non-negotiables

**Final submission deadline:** Sunday Nov. 1, 2026 at 1 PM Eastern per competition site; allocate work to actual measured science and live evidence, not UI cosmetics.

**Non-skippable engineering dependencies:** **P0 → P1 → P2 → P3 → P4 → P5 → P6**, then **P8 minimal live/replay UI** and **P7 evaluation** can proceed in either order or parallel. Finish a working P8 demonstration before committing to a costly P7 suite if time is short. P7 can attach measured results to P8 afterward. Model checkpoint reruns and extra predicted tiles remain optional, with honest one-paired-region coverage if unfinished.

**If P4 cannot pair one real prediction with real observation arrays**, the full four-specialist scientific claim is **BLOCKED**. Do not replace thermal/geology with invented arrays. A prediction-only audit may still be implemented, but the report and demo must explicitly say the thermal/geological hypotheses lack pixelwise testing.

**If P4 finds only one paired tile**, the app is partially reusable and must state the coverage limit. If P7 is incomplete, only display actually measured methods/metrics, never extrapolate a win.

## 7. Maintained sources

- SomBench source feasibility and pinned revision: [SOMBENCH_FEASIBILITY.md](SOMBENCH_FEASIBILITY.md).
- NASA–IBM task: https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model
- Pinned MOONATLAS Science: https://github.com/RRG1312/moonatlas-science
- MOONATLAS data contract: https://github.com/RRG1312/moonatlas-science/blob/main/docs/provenance/data-contract.md
- Microsoft Agent Framework provider: https://learn.microsoft.com/en-us/agent-framework/agents/providers/openai
- MAF workflows/examples: https://github.com/microsoft/agent-framework/tree/main/python/samples/03-workflows
- Streamlit: https://docs.streamlit.io/
- ScienceClaw: https://scienceclawhack.ai/#competition

**Engineering phases complete.** Remaining manual check: one live investigation started from the Streamlit UI on the demo machine (paid; needs the local OpenAI configuration). Any further benchmark requires a new versioned suite and a pre-registration written before the suite is generated.
