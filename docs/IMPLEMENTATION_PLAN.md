# IceClaw — Codex Implementation Plan

**Version:** 1.0, 2026-10-31  
**Status:** executable roadmap, **not an implementation-complete report**  
**Read first:** [PROJECT_SPEC.md](PROJECT_SPEC.md), which is authoritative for semantics and interfaces.  
**GitHub:** https://github.com/samat2003/IceClaw

## 0. How the two documents stay synchronized

- **PROJECT_SPEC.md** owns product promises, science definitions, event/data schema, required behaviors, file layout and judge acceptance gates.
- **This file** owns installation, execution order, test commands, phase-specific ownership and exit criteria.
- If an older README, GitHub issue or docs/SCIENTIFIC_PROTOCOL.md contradicts these, prefer the two canonical documents and document the discrepancy in a commit; do not silently mix two architectures.
- **Single application architecture:** Python 3.11 + Microsoft Agent Framework + scientific tools + Streamlit + SQLite. **No FastAPI, React, Redis or LangGraph in MVP.**
- **Separate upstream execution:** MOONATLAS and NASA inference run through a separately pinned Python environment/subprocess; IceClaw consumes the exported validated artifacts. Never import a second independent version of the upstream stack into the main app process.
- **One backend service contract:** CLI and UI call identical Python services. Never reimplement measurements in Streamlit.
- **Upstream version and artifact contracts pinned before use.** Do not invent installation flags, model import paths, or provider class APIs; inspect the installed package and official samples and write a 10-line runtime probe first.
- **No 'placeholder success':** if a gate fails, report BLOCKED or PARTIAL and invoke the specified fallback; don't insert fabricated measurements, fictional events, or hardcoded benchmark results.
- Every completed phase must leave a green small test suite, a documented command and a Git commit. Later phases may extend contracts only through explicitly reviewed, backwards-compatible schema changes.

## 1. Where to run Codex and the app

**Preferred:** Your own machine with Git, Python **3.11**, VS Code and Codex CLI. Codex works in the *local clone* and can install dependencies and run tests. Push commits to `samat2003/IceClaw`. Use a Google Cloud VM only if the computer cannot handle scientific dependencies or you need a publicly accessible demo. GPU/Nebius is optional for offline NASA checkpoint inference, not the agent loop.

The **Codex coding-agent sign-in** is separate from runtime credentials: sign in to Codex with your ChatGPT account (no science API key needed just to code). The **four IceClaw scientists** use a model API key configured only in the app runtime. Do not paste any API key into prompts, commits or screenshots.

### macOS/Linux/WSL: one-time prerequisites

Install Git, Python 3.11, and a terminal. Install Codex (choose one):
~~~bash
npm install -g @openai/codex
# OR standalone installer:
# curl -fsSL https://chatgpt.com/codex/install.sh | sh
~~~

Clone and launch:
~~~bash
git clone https://github.com/samat2003/IceClaw.git
cd IceClaw
git status
python3.11 --version
codex
~~~

For Windows native, use the official Codex PowerShell installer, Git and Python 3.11; activate the Python environment with `.venv\Scripts\Activate.ps1`. WSL2 is preferable if native geospatial dependencies cause problems.

**Inside Codex:** sign in with ChatGPT if prompted; run `/status` to inspect configuration and `/permissions` to use an approval mode you understand. Start in this repo directory. The root `AGENTS.md` file is created in phase 0 and will tell Codex to read BOTH canonical documents. Use `/review` and inspect diffs between phases.

**First Codex instruction (planning + Phase 0 only):**
> Read docs/PROJECT_SPEC.md and docs/IMPLEMENTATION_PLAN.md completely. Work only on Phase 0. Explain any unresolved dependency compatibility and do not invent scientific results, change the canonical schemas, or read secret files. Create the minimal repository scaffold and tests. Run them, show exact results, then stop for review.

### Runtime secrets — different from Codex credentials

During Phase 0 create an untracked `.env` mechanism (or use shell environment/secret manager) and commit only `.env.example` listing variable names, never values:
~~~dotenv
ICECLAW_LLM_PROVIDER=openai
ICECLAW_LLM_MODEL=replace-with-verified-model-id
OPENAI_API_KEY=
# Alternatives only if their provider adapter is implemented:
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
ICECLAW_DATA_ROOT=./data/processed
ICECLAW_RUNS_ROOT=./runs
~~~

Choose and verify **one** provider first. For OpenAI, use the OpenAI Platform API key awarded for the hackathon, if available. Never confuse it with Codex's ChatGPT sign-in. The app must never print secrets, and tests must use a fake provider without external API calls. In CI, use repository secrets only if a separately approved integration test requires real credentials; default CI never does.

Ensure `.env`, `.venv*`, `.upstream/`, `data/`, `runs/`, model checkpoints, caches and local log files are ignored by Git. Generated plots/reports must be reviewed for sensitive prompts before publishing.

## 2. Fixed technical dependencies

The implementation should create a `pyproject.toml` for IceClaw with compatible, tested, **pinned** versions after verifying package import and minimal operations:

- Runtime: Python 3.11; NumPy, Pandas, SciPy, Pillow, Rasterio, pyproj, Pydantic, Streamlit, Plotly.
- Agents: **Microsoft Agent Framework** (choose the package and provider integration from the official current Python samples, for example `agent-framework` / `agent-framework-core`), one model SDK/provider, and standard-library `asyncio` + `sqlite3`.
- Science: MOONATLAS is **not an import dependency** of the UI app; read its versioned outputs. Source scientific package installation lives in `.upstream/moonatlas-science/.venv`.
- Tests: pytest, pytest-asyncio (as needed), Ruff; local fixtures deliberately small and deterministic.
- Optional only: huggingface-hub / PyTorch / TerraTorch inside the **upstream model** environment, never required for the Streamlit process.

Install the IceClaw app only **after** Phase 0 creates `pyproject.toml`:
~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
~~~
For reproducibility record exact resolved dependency versions (lock/constraints file) once the package imports and core smoke tests pass, and pin the upstream Git commit SHA, model checkpoint revision, and MOONATLAS data-contract schema version. No guessed versions in plan text.

## 3. Integration topology and canonical file ownership

~~~text
IceClaw/
  app.py                          # Streamlit ONLY (Phase 7)
  pyproject.toml                  # Phase 0, pinned after probes
  AGENTS.md                       # Phase 0, Codex contract referring to two docs
  .env.example / .gitignore      # Phase 0
  src/iceclaw/
    contracts.py                  # Phase 0; canonical typed Region / Event / Bundle models
    cli.py                        # Phase 0 shell; commands extended at gates
    data/                         # Phases 1-2; MOONATLAS read-only adapter/catalog
    science/                      # Phase 3; deterministic numerical checks
    evaluation/                   # Phase 4 harness and Phase 8 experiments
    agents/ and swarm/            # Phase 5; MAF, events, SQLite, budgets
    reports/                      # Phase 6; evidence + exports + verification
  tests/                          # Added incrementally WITH each module
  docs/PROJECT_SPEC.md            # Canonical product + science
  docs/IMPLEMENTATION_PLAN.md     # This roadmap
  docs/DEMO_2MIN.md               # Earlier provisional script; update in Phase 8
  artifacts/                      # Small attributable output/examples if license permits
  data/, runs/, .upstream/        # Gitignored local data, investigations and upstream checkout
~~~

Phase boundaries and imports:
- `data` defines validated catalog/read-only raster loading; **does not import** agent framework or UI.
- `science` imports `contracts` and data readers only; **does not import** LLM or UI.
- `evaluation` imports deterministic science and run/evidence records; **never exports hidden answers into agent-visible functions**.
- `agents` calls exposed science tool functions; `swarm` persists events; neither directly loads expert reference.
- `reports` verifies science + immutable events and exports a bundle.
- `app.py` calls shared library functions/CLI services only. It never computes scientific metrics in widget callbacks.
- Repository scientific truth is in original raw source artifacts + validation. Visualization rasters are not numerical evidence.

## 4. Phase-by-phase plan (one Codex task and checkpoint at a time)

### Phase 0 — Foundation and contract tests (target 60–90 min)

**Create:** `pyproject.toml`, `.gitignore`, `.env.example`, `AGENTS.md`, typed `src/iceclaw/contracts.py`, CLI stub and small tests. Point AGENTS.md to the two canonical docs; enforce scientific integrity and secrets policy. Make placeholders raise `NotImplementedError` rather than print success.

**Test:** fresh Python 3.11 venv; `python -m pytest -q`; `python -m iceclaw.cli --help`. Every required contract serializes/deserializes and rejects missing required scientific metadata.

**Exit:** clean install + tests + help works; commit. **Do not build agents, dashboard or make any API call in Phase 0.**

### Phase 1 — Upstream verified fixture and data-contract probe (target 1–2 h)

**Prerequisite:** Phase 0.

**Create:** `src/iceclaw/data/moonatlas_adapter.py` and scientific provenance notes. Separate upstream checkout:
~~~bash
mkdir -p .upstream
git clone https://github.com/RRG1312/moonatlas-science.git .upstream/moonatlas-science
cd .upstream/moonatlas-science
git rev-parse HEAD
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
moonatlas-science validate --root examples/science-smoke --scope published
python -m moonatlas_science.contracts
~~~
Return to repo root and main `.venv` before running IceClaw tests.

**Inspect actual files**, not just README claims: `ice-sectors.json`, `ice-mosaics.json`, `sources.json`, processing manifest, raw/encoded prediction raster and available source modality files. Record published-example source/split/revision, actual projection and geometry. Read contract v2. Differentiate one-point 'atAnchor' values from whole-raster measurements.

**Tests:** decode raw *numeric* raster and verify a few exact positions plus shape/CRS/mask; recompute metadata summary against published values within a declared tolerance. Missing source outputs are a blocking error, not replaced with demonstration data.

**Exit:** MOONATLAS validation passes AND one genuine region loads with provenance and numeric evidence. Explicitly label it **upstream published fixture**, not freshly rerun NASA inference.

### Phase 2 — Two-region catalog and original observations (target 2–3 h; highest data risk)

**Prerequisite:** Phase 1. **Owns:** `data/catalog.py`, `data/validation.py`, download/preparation scripts and tests.

Use actual official held-out SomBench split/sample files and MOONATLAS acquisition commands to obtain multiple south-polar regions with provenance. Need **original input arrays** to support actual thermal/geology analyses; prediction/reference fixture alone is not enough. For model inference, run the pinned official MOONATLAS environment and its documented smoke/infer commands after inspecting available CLI flags. The upstream describes checkpoint assets totaling ~6.3 GB; a GPU can accelerate the batch but is NOT required for UI/agents.

Create a batch-validated catalog with `RegionCatalogEntry` records (data/coverage/readiness/modality availability). Enforce unique IDs and exact footprint; verify no invented synthetic tiles or unsupported geographic assumptions (the initial sector must not silently become 'Cabeus'). Preserve all source hashes and checkpoint IDs.

**Test:** `python -m iceclaw.cli list-regions` lists verified regions and readiness, and science loaders distinguish at least two real source samples. Disable tools that lack input modalities.

**Exit target:** two or more real south-polar regions, each with numeric evidence and source provenance. **Fallback (explicit PARTIAL):** one real fixture only, UI can still investigate it but must not claim multi-region generality. Time-box data acquisition; do not block later deterministic tool development indefinitely.

### Phase 3 — Deterministic scientific tools (target 1.5–2 h)

**Prerequisite:** validated data contracts. **Owns:** `science/` and tests.

Implement separate pure tools returning measurements with source IDs/units/coverage: raster descriptive stats, no-data counts, temperature/PSR consistency (only if pixelwise inputs exist), terrain measurements (only where source data exist), prediction discontinuity/edge tests, and any reproducible comparison utilities. Do not infer ice detections from high indices. Make result schema compatible with `EvidenceEvent`.

**Tests:** known synthetic numeric arrays for formula correctness; a genuine NASA/MOONATLAS array integration test for each available modality; missing data returns a typed 'unavailable', not fabricated values.

**Exit:** selected region yields an independently recomputable actual numeric measurement and source-attached scientific finding, entirely offline.

### Phase 4 — Blind evaluation harness BEFORE model-agent tuning (target 1–1.5 h)

**Prerequisite:** Phase 3. **Owns:** `evaluation/cases.py`, `evaluation/sealed_manifest.py`, baselines, tests.

Define case-level synthetic perturbations on copies of authentic source arrays (e.g., subtle offset, cross-layer inconsistency, coordinate shift, missing layer) plus clean controls. Store hidden fault manifest in evaluator-only path; never let agent tools or prompts read it. Define target classifications, predeclared metrics, case/split IDs, seeds, budgets, and source independence.

Pre-register the real-disagreement experiment: predicted ranking without target access, freeze ranking, then score against held-out expert reference. **No tuning on test labels.**

**Tests:** seed repeatability, clean-control stability, hidden manifest not imported by agents, and deterministic rule baseline.

**Exit:** fixed test cases and baseline values; save the case manifest hash before any comparative agent runs.

### Phase 5 — Real Microsoft Agent Framework swarm (target 3–4 h)

**Prerequisite:** Phases 3–4. **Owns:** `agents/`, `swarm/`, MAF provider adapter and tests.

First run a **minimal official MAF example** with a verified installed package/provider: one agent calls one deterministic IceClaw tool; a concurrent two-agent example streams events. Only then add four specialists and the shared typed event bus backed by SQLite. Do not substitute generic `asyncio.gather` for the MAF agent implementation; standard asyncio is fine for scheduling/delivery.

Implement: autonomous per-agent task selection, event subscription, challenge references, follow-up test request/execution, independently generated revision, max hops, dedup, bounded tokens/calls, timeout, failure isolation. Avoid a centralized LLM manager that dictates the entire trace. Use a recorded fake provider in tests, but authentic frontier API calls for actual demo runs.

**Test:** trace includes a real region-specific scientific tool call and (when scientifically warranted) agent-to-agent challenge + evidence-linked revision. Concurrent activity and message delivery are measurable. Case where no contradiction exists must *not* manufacture one.

**Exit:** one authentic complete live swarm investigation with genuine model API output, actual tools and event history; persistent SQLite recovery works.

### Phase 6 — Verification, CLI and downloadable report (target 1.5–2 h)

**Prerequisite:** Phase 5. **Owns:** `reports/`, `cli.py` integration and tests.

Implement `list-regions`, `investigate --region`, `verify --run`, report exporter, JSONL event persistence and a complete `InvestigationBundle` including verifiable source/checkpoint/input and code revisions. Report comprises claims, measurements, contradictory evidence, uncertainty, missing modalities and recommended next tests. Rankings mean **priority for further investigation**, never physically calibrated ice probability.

**Test:** freeze a real run, verify file hashes and recompute at least one scientific result without the LLM. A tampered raw input or missing manifest must fail. Replay reads exactly the saved run—no synthetic chat conversation.

**Exit:** reviewer can download a run and verify its numeric evidence using the CLI offline.

### Phase 7 — Streamlit workbench (target 1.5–2 h)

**Prerequisite:** Phases 2 and 6. **Owns:** `app.py` only, lightweight UI tests.

Build ONE page with region dropdown populated from catalog, available modalities/coverage, **Investigate** button, scientific 2D heatmap with real coordinates, timeline of genuine events, source-linked findings, **Verify** action, saved-run replay, and bundle download. The app calls library methods used by CLI. Keep layouts standard; avoid maps that imply global scientific coverage.

~~~bash
python -m streamlit run app.py
~~~

**Test:** choose different supported region(s), see different region/source IDs and measurements. Unsupported coordinate or absent data must show informative error, no fake success. UI must work offline for saved verified bundles.

**Exit:** two-region interactivity if Phase 2 passed, or explicitly labeled one-region partial demo if data coverage prevented it.

### Phase 8 — Execute benchmarks, finalize evidence and 2-minute presentation (target 2–3 h)

**Prerequisite:** Phases 4–7.

Run four arms on the **same** small fixed evaluation cases under recorded comparable total LLM/tool-call budgets: rule baseline, one agent with all tools, four isolated specialists, and communicating IceClaw. Run clean negative controls. Freeze outputs, reveal hidden fault/reference truth to evaluator only, then compute metrics and plots from actual saved results. Track latency, API cost, unsupported-claim count and seed/sample IDs. Report failure/no improvement honestly; do not hardcode a win.

Generate actual scientific result file(s), verification log and 5-slide presentation:
- 0:00–0:20 lunar problem and physical uncertainty
- 0:20–0:40 NASA–IBM + MOONATLAS + Microsoft Agent Framework architecture
- 0:40–1:15 select region, show original numeric evidence, challenge/revision and verification
- 1:15–1:40 show measured baseline/single/isolated/swarm results and caveat
- 1:40–2:00 relevance, reusable outputs, GitHub + verification command

Rehearse a two-minute talk; pre-record an *authentic* run as the clearly labeled offline replay fallback. Submit slides by **Sunday 2026-11-01 13:00 America/New_York**.

**Exit:** end-to-end smoke command, UI demo, honest benchmark artifacts, slides and stable repository commit. Do not claim broad scientific discovery with few patches.

### Phase 9 — Optional only after all core gates are green

Potential cross-team reuse: document a small exportable evidence/verification module or minimal API if another team needs it. Scientific additional images, GPU fine-tuning, custom 3D visualization and deployment polish are lower priority. Collaboration bonus requires **actual meaningful reuse by other teams**, not just an open-source link.

## 5. Common command suite (contract, not pre-existing implementations)

Codex must implement these commands in earlier phases before this suite can be run end-to-end:
~~~bash
python -m pytest -q
python -m iceclaw.cli list-regions
python -m iceclaw.cli investigate --region ice-s-0001-0002
python -m iceclaw.cli verify --run <run-id>
python -m iceclaw.cli benchmark --config artifacts/benchmark-config.json
python -m streamlit run app.py
~~~
`benchmark-config.json` is generated/approved in Phase 4; don't run against fake filenames before then. CI runs fast deterministic tests with a fake provider and no large model downloads. Full NASA/MOONATLAS inference tests must be explicitly marked as optional external integration tests.

## 6. Phase-handoff checklist to avoid conflicts

At the end of every Codex phase, inspect and record:
1. Files created/modified, exact git diff, commit hash and any version/schema migrations.
2. Exact tested commands + their stdout/pass/fail summary; never claim green without running them.
3. New input and output artifact paths and a sample schema-valid JSON.
4. Upstream SHA, data sample ID, checkpoint revision and operating mode (published example vs fresh inference).
5. Which later phases are newly unblocked and which remain BLOCKED/PARTIAL.
6. No API secrets, whole datasets, model weights or huge generated results in git.

Use **one active implementation phase per Codex task/commit**. When a later phase wants to change a prior module, it must add/modify tests and preserve the explicit contract; do not duplicate data loading, science metrics, event schemas or reporting logic across stages.

**Stop conditions:** bad source provenance; reference-label leakage; misregistered lunar CRS; missing required source data; simulated agent events presented as real; unsupported scientific certainty; benchmark that compares unequal budgets. Surface these as failures and fix before demo.

## 7. Time and fallback priorities

The hard stop is **2026-11-01 at 13:00 ET** (final slide deadline). Budget time for actual experiment runs and slide rehearsal rather than polishing the UI.

**Priority order:** verified input > one deterministic scientific calculation > working independent MAF agents and evidence exchange > verifiable report > one fair benchmark > Streamlit wrapper > multi-region expansion > full rerun of all NASA checkpoints.

If data availability is limited, say exactly which modalities/regions are covered and disable corresponding specialist tools. If model API access fails, offer validated deterministic results and **clearly labeled authentic saved-run replay**, but do not claim a fresh live swarm ran. If the 4-arm benchmark cannot be completed, publish the actually measured arms and list unfinished comparisons. Never show invented metrics.

## 8. Source documentation

- ScienceClaw rules, judging, 13:00 Sunday deadline: https://scienceclawhack.ai/#competition
- Codex local setup/AGENTS.md: https://github.com/openai/codex and https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex
- NASA–IBM scientific checkpoint: https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model
- MOONATLAS quickstart and sample validation: https://github.com/RRG1312/moonatlas-science
- MOONATLAS v2 data contract: https://github.com/RRG1312/moonatlas-science/blob/main/docs/provenance/data-contract.md
- Microsoft Agent Framework official Python concurrent/workflow samples: https://github.com/microsoft/agent-framework/tree/main/python/samples/03-workflows
- Streamlit install: https://docs.streamlit.io/get-started/installation

**Start now with Phase 0, then Phase 1. Do not ask Codex to implement all phases in one unreviewed turn.**
