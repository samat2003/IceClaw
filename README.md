# IceClaw

**Decentralized AI planetary scientists for auditing lunar polar ice-prospectivity predictions**  
ScienceClaw 2026 · [Project source](https://github.com/samat2003/IceClaw)

**Project status (2026-10-31):** P0–P4 are committed: tested Python core, authentic MOONATLAS prediction, deterministic science, sealed evaluation, and real SomBench temperature/shadow/slope paired with one prediction tile. P5 was **reported completed locally but has not been committed yet**: 215 passing tests (one default opt-in skip), plus one genuine four-specialist Microsoft Agent Framework run. Its Skeptic performed a live cross-agent follow-up; no live challenge/revision occurred. P6 verification/report, P7 matched-budget comparative results and P8 Streamlit demo remain pending. These reported tests were not independently rerun during this README update.

## What it will do

A researcher selects a *verified, covered* lunar south-polar tile. Four independent scientific specialists inspect authentic NASA-derived observations and published NASA–IBM ice-prospectivity predictions, propose/test hypotheses, challenge one another and publish a transparent evidence log. IceClaw exports a verifiable investigation and prioritizes further research.

**Ice prospectivity is not a physical measurement of ice or a calibrated ice probability.** This research will audit predictions, **not claim new ice discovery**.

## Canonical project documents

1. **[Project specification](docs/PROJECT_SPEC.md)** — scientific objective, technology stack, data contracts, user experience, evidence standard, safety and evaluation semantics.
2. **[Engineering implementation plan (revision 3.1)](docs/IMPLEMENTATION_PLAN.md)** — P0–P4 committed, P5 reported complete locally; P6 portable verification and context-budget safeguards, P8 minimal working Streamlit demo, P7 only eligible paired-scenario and matched-budget experimental comparisons. **This is tool-independent software work, not instructions for a particular coding assistant.**
3. [Repository implementation guidance](AGENTS.md) — short durable project rules. [CLAUDE.md](CLAUDE.md) references these same rules without duplicating or changing the engineering plan.

The older [scientific protocol](docs/SCIENTIFIC_PROTOCOL.md) and [two-minute presentation draft](docs/DEMO_2MIN.md) are supplementary historic notes. In a conflict, prefer the project specification and current engineering plan.

## Upstream open-source foundations

- [NASA–IBM Lunar Foundation Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) — specialist lunar scientific inference.
- [MOONATLAS Science](https://github.com/RRG1312/moonatlas-science) — pinned scientific acquisition/inference, raster normalization, georeferencing, provenance, reproducibility and validation. It is an **independent** project, not operated or endorsed by NASA/IBM.
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) — independently acting specialist agents, scientific tool calls and concurrent workflows.

The original IceClaw code is the selectable-region research workbench, independently tested numerical tools, asynchronous specialist interactions, evidence verification and controlled collective-vs-single-agent evaluation.

## Intended software architecture

- **Python 3.11/3.12**, NumPy, Rasterio, SciPy, pyproj
- **Microsoft Agent Framework**, direct OpenAI model integration via its current supported Python OpenAI provider
- **SQLite** append-only research events
- **Streamlit + Plotly**, one-page interface; no separate API server for the hackathon
- **MOONATLAS/NASA inference** in a separate pinned scientific environment; the application consumes validated output artifacts
- **Live agent configuration:** `OPENAI_API_KEY`, `ICECLAW_LLM_MODEL`, and `ICECLAW_LLM_PROVIDER=openai` from local environment; NEVER committed

The planned UI allows a user to choose a region with **both** verified original scientific observations and a spatially paired model prediction, run a new investigation, inspect source-backed claims and alternatives, verify a deterministic calculation, and download a research bundle. Observation-only records may be inspected but are not complete prospectivity investigations. No assertion of global lunar coverage.

## Local WSL start

Use the current clone:

```bash
cd ~/IceClaw
git status --short
# Pull latest committed documents after safeguarding any colliding local untracked files.
git pull --ff-only origin main
```

**Note:** This repository now tracks `.gitignore`, `CLAUDE.md`, `AGENTS.md`, and `.env.example`. If you already created one of these locally but it is *untracked*, Git may refuse to pull because it would overwrite it. **Move the local copy to a backup outside the repository first; never delete or overwrite `.env`.** After pulling, compare the backup with the tracked version. See the next step below.

Open `docs/IMPLEMENTATION_PLAN.md`. **P0–P4 are committed and P5 is reported complete locally (commit pending). Next: review/commit P5, then start P6.** Run existing unit tests before changes. Do not conflate the P3/P4 engineering baselines with an actual LLM collective-vs-single-agent experiment. The verifiable downloadable report and Streamlit demo have not been implemented yet.

Local `.env` may already hold an OpenAI key. Leave it untouched. `.env.example` contains only placeholder variable names, and `.gitignore` excludes `.env`, data, virtual environments, local research runs, model weights and local settings.

## First scientific reproducibility gate

The published MOONATLAS sample includes a south-polar sample identified as `ice-s-0001-0002`. It is a **published example, not a fresh IceClaw inference**. P1 has validated this real upstream example. The relevant reproduction commands in the upstream MOONATLAS checkout are:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
moonatlas-science validate --root examples/science-smoke --scope published
```

Those commands must be run **within the upstream MOONATLAS scientific environment**, not in the application root. P2 identified **no pixelwise thermal/PSR/slope arrays** in the smoke fixture. **P4** will ingest the pinned original SomBench observation GeoTIFFs for the exact P1 patch and verify their alignment before full thermal/geology analyses or four-specialist scientific claims. Original targets (`PRO`) remain evaluator-only.

## Verification and evaluation goals

The full tool will produce source-referenced numerical measurements, event logs of authentic agent actions, a reproducible scientific bundle, benchmark artifacts and a fast offline verification command. It will compare (1) deterministic rules, (2) single AI scientist, (3) isolated specialists, and (4) communicating specialists on held-out, budget-controlled cases. Expert reference maps and synthetic fault labels are evaluator-only; the metric is **scientific auditing**, not detection of real water deposits.

**Hackathon timeline:** final slides due Nov. 1, 2026 at 1:00 PM ET per [official competition](https://scienceclawhack.ai/#competition). Two-minute talk is our presentation target, not necessarily a rule.
