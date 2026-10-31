# IceClaw

**Decentralized AI planetary scientists for auditing lunar polar ice-prospectivity predictions**  
ScienceClaw 2026 · [Project source](https://github.com/samat2003/IceClaw)

**Project status (2026-10-31):** P0–P2 implemented and committed: Python core/contracts, validated real MOONATLAS numeric source, repeatable scientific tools, and region catalog. The reported local suite has **94 passing tests**; this README update did not rerun them. **P3–P8 remain pending.** No fresh NASA inference, live scientific swarm, app interface or measured benchmark is claimed.

## What it will do

A researcher selects a *verified, covered* lunar south-polar tile. Four independent scientific specialists inspect authentic NASA-derived observations and published NASA–IBM ice-prospectivity predictions, propose/test hypotheses, challenge one another and publish a transparent evidence log. IceClaw exports a verifiable investigation and prioritizes further research.

**Ice prospectivity is not a physical measurement of ice or a calibrated ice probability.** This research will audit predictions, **not claim new ice discovery**.

## Canonical project documents

1. **[Project specification](docs/PROJECT_SPEC.md)** — scientific objective, technology stack, data contracts, user experience, evidence standard, safety and evaluation semantics.
2. **[Engineering implementation plan (revision 3.0)](docs/IMPLEMENTATION_PLAN.md)** — P0–P2 completed; P3 sealed evaluator; **P4 real SomBench thermal/PSR/slope data pairing before P5 agents**; P6 verifiable reports; P7 measured benchmark; P8 minimal Streamlit UI. **This is tool-independent software work, not instructions for a particular coding assistant.**
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

Open `docs/IMPLEMENTATION_PLAN.md`. **P0, P1 and P2 have been implemented and committed; the next phase is P3.** Do not redo the initial package or adapter. Run existing unit tests before changes. **The benchmark, agents, verification and Streamlit UI do not yet exist**; check their phase gates rather than calling them prematurely.

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
