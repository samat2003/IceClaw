# IceClaw — Engineering Implementation Plan

**Revision:** 2.0 · 2026-10-31  
**Status:** NOT IMPLEMENTED. This document specifies what to build and how to prove each milestone works.  
**Authoritative product/science contract:** [PROJECT_SPEC.md](PROJECT_SPEC.md)  
**Repository:** https://github.com/samat2003/IceClaw

## 0. Immutable decisions and rules

### Product

A reusable research workbench for **auditing lunar ice-prospectivity predictions**. The user selects a verified covered south-polar data tile and runs a new investigation. Four independent scientific specialists use deterministic tools and exchange evidence. Outputs are an auditable report, actual source measurements, a prioritization of follow-up research, and reproducible benchmark evidence.

**Never assert discovery of actual ice.** The prediction and expert reference are prospectivity indices, **not** direct ice detections, ice probabilities, or independent ground truth.

### Final technology choices

| Layer | Technology | Process boundary |
| --- | --- | --- |
| Application/runtime | Python 3.11 or 3.12 (one version per environment) | One local application |
| Scientific numerics | NumPy, Pandas, SciPy, Rasterio, pyproj | Pure/tested tools |
| Scientific model | NASA–IBM Lunar Foundation Model | Offline batch inference only |
| Data normalization/provenance | MOONATLAS Science | Separate pinned Python environment, export read-only validated artifacts |
| Specialist agents | **Microsoft Agent Framework** | Python application environment |
| Agent LLM | OpenAI via `agent-framework-openai` / `OpenAIChatClient` | Key from runtime environment, model passed explicitly |
| Concurrency/events | `asyncio` + append-only SQLite | No central LLM supervisor |
| Interface | Streamlit + Plotly | Same Python services as CLI |
| Tests | pytest, pytest-asyncio where required, Ruff | No remote API in default tests |
| Deployment | WSL/local initially; optional single-VM Streamlit | No mandatory GPU or microservices |

Do **not** add FastAPI, Next.js/React, Redis, Celery, LangGraph, a vector database, model fine-tuning or arbitrary file uploads to the MVP. Do not place heavy PyTorch/TerraTorch dependencies in the Streamlit/agent environment.

### One source of truth

- Science and product claims, typed data contracts, input/output rules: `docs/PROJECT_SPEC.md`.
- This document: phase ownership, module boundaries, test gates and runtime commands.
- `docs/SCIENTIFIC_PROTOCOL.md` and `docs/DEMO_2MIN.md`: earlier background drafts; these must not override the two canonical documents.
- If code requires a contract change, update `PROJECT_SPEC.md` with the rationale and a schema migration/test *before* making the code change.
- Implementation proceeds one phase at a time. Never skip a failed gate and silently call the feature finished.

### Scientific data firewall (non-negotiable)

The internal `RegionCatalogEntry` may contain a reference-prospectivity file path. **Never pass that model directly to agents.** Derive an explicit `AgentRegionContext` containing only approved observations, NASA–IBM prediction, visible provenance, available tool names, and limits. Reference targets and hidden injected-fault manifests reside in evaluator-only storage/services. No agent tool may open arbitrary files or call evaluator functions. Tests must assert these forbidden fields and data are absent from prompts, tool arguments and event logs.

### Working-directory assumptions

- WSL repository at `~/IceClaw`. Existing `.env` is user-owned: **never open, rewrite, delete or commit it**.
- GitHub branch `main`; local changes must be reviewed before any `git pull` or push.
- OpenAI runtime key: `OPENAI_API_KEY`; model config: `ICECLAW_LLM_MODEL` (a model ID the user has independently verified); provider: `ICECLAW_LLM_PROVIDER=openai`.
- Coding-tool authentication is **not an IceClaw runtime dependency**.
- Default and CI tests must work without API keys or internet. Integration runs requiring the NASA checkpoint or real LLM are marked separately.

## 1. Component architecture (import direction)

```text
Streamlit app.py --------> iceclaw.services ----------------> reports
CLI iceclaw.cli ---------> iceclaw.services ---------> swarm/agents -----> science
                                |                         |                   |
                                +----> catalog/data -------+-------------------+
                                                           
evaluation (sealed reference/fault truth) -----> archived/frozen run outputs
                    [never imported by agents/science tools]
```

**Allowed dependencies:**

- `contracts` is foundational; all modules may import it.
- `config` reads settings; does not expose secret values in logs, reports or exports.
- `data` reads/validates upstream artifacts and exposes covered samples, with clear provenance.
- `science` uses data + numeric utilities only; no LLM, Streamlit, or evaluation labels.
- `swarm` manages event delivery, budgets and SQLite; `agents` uses MAF and only approved scientific tool functions.
- `reports` creates/validates saved bundles from real inputs, science tool outputs and logged events.
- `evaluation` owns hidden truth and frozen-run scoring; it may import public science tools but is not imported by `agents`, `swarm`, or `data.public`.
- `services` is the **only application-facing investigation API**; both CLI and UI call it. Do not reimplement science or event logic in `app.py`.

Proposed committed repository:

```text
IceClaw/
  README.md
  AGENTS.md                     # short universal repository guidance
  CLAUDE.md                     # short reference to AGENTS.md, not a duplicate spec
  .gitignore
  .env.example                  # placeholders only
  pyproject.toml
  app.py                        # created at Phase 8
  src/iceclaw/
    __init__.py
    config.py
    contracts.py                # RegionCatalogEntry, AgentRegionContext, EvidenceEvent, InvestigationBundle
    cli.py
    services.py
    data/
      catalog.py
      moonatlas_adapter.py
      validation.py
    science/
      raster_metrics.py
      thermal.py
      terrain.py
      model_audit.py
    swarm/
      store.py                  # SQLite append-only log and subscriptions
      runtime.py                # async execution, limits, cancellation
    agents/
      provider.py               # MAF OpenAI client integration
      specialists.py            # four configured agents
      tools.py                  # approved deterministic tool adapters
    reports/
      export.py
      verify.py
    evaluation/
      fixtures.py
      faults.py
      baselines.py
      runner.py
      score.py
  tests/                        # contracts, readers, tools, event store, firewall, replay, benchmarks
  scripts/                      # optional pinned upstream preparation wrappers
  docs/
    PROJECT_SPEC.md
    IMPLEMENTATION_PLAN.md
    SCIENTIFIC_PROTOCOL.md
    DEMO_2MIN.md
  .upstream/                     # ignored MOONATLAS checkout
  data/                          # ignored raw/normalized scientific files
  runs/                          # ignored per-investigation outputs
```

Use these names unless there is a documented compelling technical conflict. The modules can be small: avoid creating unused abstractions or placeholder microservices.

## 2. Frozen data contracts (implement exactly once)

**Source of authority:** `PROJECT_SPEC.md` §5. Use Pydantic 2 or typed dataclasses with explicit validation. At minimum:

**`RegionCatalogEntry` (private full record):** region ID, pole, original source tile/split, original dataset and processing revision, source hashes/manifests, native lunar CRS/affine transform/footprint, width/height/pixel resolution, available modalities and unit/nodata metadata, raw prediction path/checkpoint, evaluator-only reference path, readiness and failure reasons.

**`AgentRegionContext` (public restricted projection):** ID, covered footprint, raw *prediction access via approved tool*, instrument layers allowed for that case, source/provenance IDs, advertised available scientific tools and run budget. Must exclude reference raster paths, labels, hidden fault descriptions, global evaluator root path, any secret. Construct via an explicit allowlist rather than serializing the full record with a blacklist.

**`EvidenceEvent` (durable append-only):** event ID, run ID, region ID, time, origin agent, event kind, parent IDs, structured claim/test, cited source/tool run IDs, numeric value and unit if present, status, limitations and errors. Event kinds are `claim`, `measurement`, `challenge`, `test_request`, `test_result`, `revision`, `completion`, `failure`. A challenge points to an earlier claim and a revision points to a qualifying piece of new evidence; missing evidence must be flagged.

**`InvestigationBundle` (under `runs/<id>`):** `input_manifest.json`, `events.jsonl`, `evidence.json`, `report.md`, `rankings.csv`, `verification.json`, optional `plots/`. Bundle stores executed tool IDs/parameters, file checksums, data split/revision, model/provider ID (never credentials), code SHA, science availability/limitations. Saved LLM responses are audited, but exact LLM wording is not expected to reproduce.

**`ToolMeasurement` (introduced here):** typed science response with `tool_id`, parameters, region/footprint, source file IDs and SHA256, raw numeric result or unavailable reason, units, valid-pixel count, scale/resolution, reproducibility tolerance and deterministic implementation version. `EvidenceEvent` refers to `ToolMeasurement` by unique `tool_run_id`.

**One storage rule:** raw model output is numeric and remains unmodified, even outside [0,1]. Display PNG/WebP is never used for numerical analyses. The MOONATLAS example may contain only *single-anchor* thermal/terrain metadata; do not pretend that is a full spatial field.

## 3. Environment and package setup

**Do this only when Phase 0 creates `pyproject.toml`.** The repo currently contains planning documents and no executable Python package.

```bash
cd ~/IceClaw
python3 --version                     # must be 3.11 or 3.12
python3 -m venv .venv                 # only if .venv does not exist
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

The project depends on installed `agent-framework-openai` for direct OpenAI inference. Current documented Python imports are `from agent_framework import Agent` and `from agent_framework.openai import OpenAIChatClient`, with `OpenAIChatClient(model=...) `. **Verify a minimal import/execution probe against the exact installed version before depending on any class or workflow API**; some older examples use renamed clients. Keep OpenAI usage behind `agents/provider.py`.

The application should load local `.env` via `python-dotenv` during startup **without printing its values**. Do not create/overwrite the user's existing `.env`. A missing key should not prevent `list-regions`, numeric tools, saved-run verification, or viewing a completed replay; only the live investigation reports `MISSING_API_KEY`. Use `.env.example` as a template, never as real credential storage.

**Separate scientific environment** for MOONATLAS/NASA (only Phase 1+) lives at `.upstream/moonatlas-science/.venv`:
```bash
git clone https://github.com/RRG1312/moonatlas-science.git .upstream/moonatlas-science
cd .upstream/moonatlas-science
git rev-parse HEAD                       # record exact upstream SHA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
moonatlas-science validate --root examples/science-smoke --scope published
```
Then return to `~/IceClaw` and activate the application `.venv` to run its tests. Do not run `moonatlas-science smoke` or download large checkpoints until the fixture reading works and needed disk/GPU capacity is checked. For full inference use only the upstream documented setup and task configuration, verified at the pinned commit.

## 4. Exact implementation milestones and acceptance gates

### P0 — Packaging, configuration, contracts, CLI shell

**Depends on:** none. **Owns:** package skeleton, `pyproject.toml`, `config.py`, `contracts.py`, `cli.py`, basic tests.

1. Create `src/iceclaw` installable packaging, Python version guards and tested minimal dependencies; separate `dev` extras. Python 3.11 recommended.
2. Implement typed contracts from §2, with exhaustive validation of required scientific provenance, valid event kinds, positive raster dimensions and absence of reference fields in `AgentRegionContext`.
3. Centralize settings with `ICECLAW_DATA_ROOT`, `ICECLAW_RUNS_ROOT`, `ICECLAW_LLM_PROVIDER`, `ICECLAW_LLM_MODEL` and `OPENAI_API_KEY`. No key material in log output.
4. Add CLI subcommands `list-regions`, `investigate --region`, `verify --run`, `benchmark --config`. Until implemented, commands fail with a clear `NOT_IMPLEMENTED`, nonzero exit status. Do not return fake example results.
5. Add tests and a `pytest -q` runner; add Ruff if compatible.
6. **Gate:** fresh local install works, `python -m pytest -q` passes, `python -m iceclaw.cli --help` lists commands, malformed contracts raise errors. No live API call.

### P1 — Verify one authentic MOONATLAS scientific artifact

**Depends on:** P0. **Owns:** upstream acquisition notes and `data/moonatlas_adapter.py`, upstream integration tests.

1. Install and run MOONATLAS's upstream example validator in its separate environment as in §3; save pinned revision and command output in local run metadata.
2. Inspect the actual `examples/science-smoke` directory: `manifest.json`, `sources.json`, `ice-sectors.json`, `ice-mosaics.json`, `processing-manifest.json`, and referenced `overlays/ice/*` numeric files.
3. Parse real sector `ice-s-0001-0002`; report exactly what the upstream says about pole, footprint, coordinate system, source split, source checkpoint, grid dimensions, raw prediction format and available modalities.
4. Decode source **raw numeric** prediction through the documented encoding/scale, not the display heatmap. Verify a few decoded pixel positions and source metadata; test missing file and incorrect checksum behavior. Confirm published sample's `atAnchor` is not a pixelwise physical modality.
5. **Gate:** one real source sector loads and numerical summaries can be recomputed; MOONATLAS published-sample validator passes. Mark `source_mode=upstream_published_example`; never call it fresh IceClaw model inference.

### P2 — Deterministic science library and region catalog

**Depends on:** P1. **Owns:** `data/catalog.py`, `data/validation.py`, `science/*`, tests.

1. Materialize a read-only validated catalog; include exact source/split/checkpoints, valid CRS/affine grid, raster shape, true footprint, units, nodata handling and an explicit list of available modalities.
2. Build pure functions for valid-pixel summary, native-grid statistics, pixel-distance/edge checks and predictive discontinuity candidates. Thermal/PSR/terrain checks run only when complete relevant input arrays are present.
3. Return `ToolMeasurement` and an explicit `UNAVAILABLE_MODALITY` / `UNVERIFIED_SOURCE` / `OUT_OF_COVERAGE` failure when appropriate; no guessed values.
4. **Gate:** `python -m iceclaw.cli list-regions` lists real sector(s); a deterministic tool reports one actual raw-array numerical measurement traceable to the original source checksum. Tests verify array math on small synthetic fixtures and source integration using the authentic sector.

### P3 — Frozen scientific evaluation contract / no-leakage controls

**Depends on:** P2. **Owns:** `evaluation/fixtures.py`, `evaluation/faults.py`, `evaluation/baselines.py`, tests.

1. Prepare small case-level fault scenarios derived by *copying* genuine data: altered metadata/units, spatial shift, missing modality, corrupt prediction, and clean controls. Store true fault labels and seeds in evaluator-only files, never within agent-readable source entries.
2. Define deterministic anomaly/QC baseline and metrics: precision, recall, F1, clean-case false-positive rate; ensure entire case IDs, not individual pixels, are the counting units.
3. Define secondary model/reference-disagreement benchmark with **hidden expert target** until prioritization is frozen. Specify Top-K score and split/time budgets in a precommitted manifest.
4. Verify reference/map leakage impossible through `AgentRegionContext`, tool API, imports and prompts. Unit-test sealed boundary and fixed seeded corruption.
5. **Gate:** frozen case manifest/hash, source IDs, clean controls and deterministic baseline run. No AI has been tested against the sealed answers yet.

### P4 — Microsoft Agent Framework runtime and event store

**Depends on:** P2 and P3. **Owns:** `agents/*`, `swarm/*`, tests.

1. Confirm direct OpenAI provider package `agent-framework-openai` and current `OpenAIChatClient` API; first implement a tiny MAF agent that invokes one real deterministic IceClaw tool.
2. Define four independently operating roles: thermal scientist, geology scientist, prediction auditor and skeptic/planner. Their tools are limited to actually available modalities; no promise of a thermal calculation if the input has only one anchor metadata value.
3. Implement concurrency, subscriber delivery and append-only SQLite event records. At least one agent can request a follow-up test after seeing another agent's finding, and each request/revision references parent evidence IDs.
4. Scheduler may manage work delivery, retries, max hops, token/tool/event budget, deduplication, timeout and cancellation; **not** prescribe the scientific findings. Use MAF as the true agent runtime, not four simulated role strings.
5. Offline tests use a fake model provider and deterministic numeric fixtures. Live tests use the configured OpenAI key and capture provider/model ID, actual API responses, actual MAF tool calls, and actual event timestamps. Do not make the code print `.env`.
6. Do not force a disagreement: if no contrary evidence exists, agents may conclude insufficient data, unresolved or no detected issue.
7. **Gate:** a genuine live run produces a real source-backed scientific tool result plus independently recorded events, and an evidence-triggered cross-agent follow-up where warranted. An absent API key yields a clear error for live mode, without breaking offline science.

### P5 — Investigation service, verifiable bundle and replay

**Depends on:** P4. **Owns:** `services.py`, `reports/export.py`, `reports/verify.py`, CLI implementations, tests.

1. Implement a shared `InvestigationService` entry point used by UI and CLI. Accept verified region ID, mode (`live` or `replay`), and bounded options; create a unique run ID.
2. Save all authentic activity and numeric provenance to `runs/<run_id>/` under the exact bundle contract (§2). Generate Markdown conclusion with alternatives, uncertainty, unresolved questions, missing modalities and recommended tests.
3. Implement `python -m iceclaw.cli investigate --region ID` and `python -m iceclaw.cli verify --run ID`. Verification runs without a model API: checks files/hashes, source/CRS, event references, and recomputes at least one saved deterministic measurement within declared tolerance.
4. Replay only saved authentic events. Do not fabricate agent interactions or edited scores for a presentation.
5. **Gate:** offline verifier passes an authentic saved bundle; modified source raster, wrong hash or missing event parent fails verification; CLI returns nonzero exit code on failure.

### P6 — Multi-region real-data catalog

**Depends on:** P1, P2 and P5. **Owns:** data preparation scripts and catalog integration tests.

1. Retrieve additional held-out **actual** south-polar SomBench samples, inspecting official split manifests and source modality tensors. Prepare NASA–IBM outputs using a pinned MOONATLAS run when feasible; reuse the published outputs where correctly attributable.
2. Validate samples independently and append covered regions to catalog. Keep raw input artifacts/checkpoints outside Git. Record exactly whether each region is a published upstream output or a locally reexecuted checkpoint; log input-layer availability.
3. Test selecting two distinct real regions yields different source files, raster values and evidence. Do not infer Cabeus/Shackleton coverage from the region name.
4. **Gate target:** at least two covered verified south-polar regions. If data access/compute blocks this, mark `PARTIAL_ONE_REGION` and continue to P7–P8 with truthful limits; never synthesize a second source region.

### P7 — Matched-budget experiment and reproducible scoring

**Depends on:** P3, P4, P5. Can run alongside P6 after P5. **Owns:** `evaluation/runner.py`, `evaluation/score.py`, experimental artifacts, tests.

1. From one frozen, sealed case list run: (A) deterministic baseline; (B) one LLM scientist with all allowed tools; (C) isolated four specialists with communication disabled; (D) full communicating IceClaw swarm.
2. Fix information, cases, total API-call/token/tool budgets, code/data revisions, repeat count/seeds (where supported), and allowed tool access. Preserve actual budget consumption; do not silently give the swarm four times the inference.
3. Freeze each arm's predictions/rankings and run files before evaluator-only hidden fault labels/reference are revealed. Score seeded fault detection and clean controls; separately score untouched reference-disagreement prioritization. Report insufficient sample size, uncertainty and bias.
4. Export real CSV/JSON results and charts from saved metrics; include elapsed time, cost/tokens and unsupported scientific claims.
5. **Gate:** at least one honest measured comparison with raw per-case outputs and evaluator command that reproduces reported numbers. If not enough runs, state results as an exploratory demonstration; never print invented superiority.

### P8 — Minimal Streamlit UI, final integration and presentation

**Depends on:** P2, P5 and scored experiments from P7 when available. **Owns:** `app.py`, UI integration tests, docs/demo updates.

1. One Streamlit page only: dropdown from verified catalog, available modalities/source info, a `Start investigation` action, 2D lunar-georeferenced prediction heatmap, authentic timeline, evidence/limitations, `Verify` button, download saved report/bundle, real benchmark summary.
2. `app.py` calls `InvestigationService` / same functions as CLI; never duplicates scientific math, loads hidden reference into the agent context, or invents a successful run.
3. Support authentic replay without a current API key, clearly labeled `Recorded run`. A region with missing science inputs displays relevant tool as unavailable.
4. **Gate:** `python -m streamlit run app.py` opens; real user selection triggers a genuine new investigation in live mode; saved-run verify works; source file, values and run ID change when a second validated region is selected.
5. Prepare at most five concise slides and a two-minute explanation using **only executed** results, with a pre-recorded authentic replay fallback and a command judges can run to verify a finding.

## 5. Command contract and test matrix

**These are commands to implement during the phases above, not a claim they currently exist.**

```bash
# Main application venv after P0
python -m pytest -q
python -m iceclaw.cli --help

# P2+
python -m iceclaw.cli list-regions

# P5+ (requires valid API configuration for live mode)
python -m iceclaw.cli investigate --region ice-s-0001-0002
python -m iceclaw.cli verify --run RUN_ID

# P7+ (path generated by P3 / checked in only when devoid of hidden labels)
python -m iceclaw.cli benchmark --config PATH_TO_APPROVED_CONFIG

# P8+
python -m streamlit run app.py
```

| Test suite | Data/API dependency | Must pass |
| --- | --- | --- |
| Contract/schema tests | None | P0 onward |
| Raw raster + provenance adapter | Published real MOONATLAS smoke fixture | P1 onward |
| Numerical methods / missing modalities | Small deterministic fixtures + authentic source | P2 onward |
| No-leakage benchmark guards | Synthetic sealed cases | P3 onward |
| Async event store / fake provider | None | P4 onward |
| Full live MAF provider integration | OpenAI API key + source data | Explicit opt-in P4 onward |
| Bundle verify / tamper test | Authentic archived tool outputs | P5 onward |
| Multi-region selection | Second verified held-out sector | P6 if available |
| Scoring reproducibility | Frozen scored case files | P7 onward |
| Streamlit UI smoke | Catalog + authentic saved bundle | P8 |

### Completion report required after **every phase**

- Files added and changed; exact `git status`/diff summary, commit hash if committed.
- Exact commands executed and actual pass/fail counts.
- Source data and science provenance involved, plus environment/package versions.
- Sample schema-valid artifact IDs/paths and whether derived from published or reexecuted model output.
- Remaining blockers and explicit `DONE`, `PARTIAL`, or `BLOCKED` status.
- Confirm secret values, model weights and raw datasets were not committed.

**No premature green checks:** running a mock provider doesn't prove an actual model API call; a published smoke sample doesn't prove fresh NASA inference; a unit-test synthetic array doesn't prove a scientific lunar result.

## 6. Integration release gates

- **G0 Setup:** reproducible app install; no committed secrets.
- **G1 Scientific source:** authentic NASA-derived numeric raster; MOONATLAS validation and provenance.
- **G2 Numeric truth:** at least one reproducible actual tool measurement.
- **G3 Collective intelligence:** Microsoft Agent Framework specialist activity, independent event reactions and test requests; authentic event log.
- **G4 Judge verification:** offline deterministic recomputation with tamper detection.
- **G5 Reusability:** selectable validated real regions; two if obtained, otherwise labeled partial.
- **G6 Evidence:** one matched-budget benchmark actually executed; no fake metrics or label leakage.
- **G7 Demo:** Streamlit opens and displays real, verified results, export and replay.

**Failure modes:** bad CRS, wrong raw-raster decoder, hidden reference leakage, invented scientific certainty, fabricated agent activity, model/secret exposure or an unmatched-budget comparison require a visible failing status and correction. Do not hide them behind a screenshot.

## 7. Schedule and scope cuts

**External deadline:** 2026-11-01 13:00 ET for final slides according to the competition page. Reserve sufficient time for scientific runs, presentation and backups.

Priority if constrained: P0 → P1 → P2 → P4 → P5 → P7 → P8. Complete P3's label-firewall/frozen benchmark contract before P4 benchmark runs. P6's extra sectors are important for reuse but may be labeled partial if unavailable. Everything else is lower priority than an honestly verifiable scientific result.

**Deferred:** custom 3D Moon, arbitrary dataset upload, fine-tuning, Nebius H100 dependency, multi-cloud services, stand-alone API and a journal paper.

## 8. Official implementation references

- NASA–IBM model: https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model
- MOONATLAS install/validation: https://github.com/RRG1312/moonatlas-science
- MOONATLAS data contract: https://github.com/RRG1312/moonatlas-science/blob/main/docs/provenance/data-contract.md
- Microsoft Agent Framework: https://github.com/microsoft/agent-framework
- Current direct OpenAI Python provider: https://learn.microsoft.com/en-us/agent-framework/agents/providers/openai
- Microsoft workflow/parallel samples: https://github.com/microsoft/agent-framework/tree/main/python/samples/03-workflows
- Streamlit: https://docs.streamlit.io/
- Competition: https://scienceclawhack.ai/#competition

**First executable task:** P0 only. The next milestone may proceed only when the preceding required gates have actual passing tests.
