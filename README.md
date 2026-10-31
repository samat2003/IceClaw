# IceClaw

**Decentralized AI planetary scientists for auditing lunar polar ice-prospectivity predictions**
ScienceClaw 2026 · [Project source](https://github.com/samat2003/IceClaw)

**Project status (2026-10-31):** all engineering phases **P0–P8 are implemented**. 360 offline
tests pass (one opt-in live test skipped); Ruff is clean. One genuine live four-specialist
investigation was recorded, exported, verified offline and published as a public-redacted bundle.
The P7 comparison was executed once under a pre-registered protocol and its result is **negative
and exploratory** (below). The live-investigation button of the Streamlit app is wired to the real
runtime and covered by offline tests, but **a live click from the UI has not yet been exercised**
(final manual check).

**Ice prospectivity is not confirmed lunar ice.** The NASA–IBM output is a regression model's
ice-prospectivity *index*: not a detection, not an ice abundance and not a calibrated probability.
IceClaw audits that prediction; it does not claim to discover ice.

## What IceClaw does

A researcher selects a verified lunar south-polar region. Four Microsoft Agent Framework specialists
(Thermal, Geology, Prediction Auditor, Skeptic) — running on the configured OpenAI model
(`gpt-6-luna`) — independently run deterministic scientific tools on the published NASA–IBM
prediction and the genuine SomBench temperature (TMAX, K), permanent-shadow (LPSR, fraction) and
slope (SLOPE, deg) layers, publish claims that cite their measurements, and may request tests from
or challenge one another. Every action goes to an append-only SQLite evidence log.

IceClaw then writes a **reproducible investigation bundle**: the events, every measurement, a
report, an investigation-attention list and the exact source rasters. Anyone can re-verify it
offline — the verifier recomputes every measurement from the embedded rasters, re-derives the
report, checks hashes, units and references, and confirms that no agent statement claims detected
ice. This turns a model output into an auditable set of checked measurements, documented
alternative explanations and open research questions in about a minute, instead of a manual
cross-checking session.

## Coverage: one paired region, eleven observation-only tiles

* **`ice-s-0001-0002`** is the only region with a verified NASA–IBM prediction (published
  MOONATLAS example, not a fresh IceClaw inference) **paired** with the same tile's SomBench
  TMAX/LPSR/SLOPE. It is the only region that can be investigated (`PARTIAL_ONE_REGION`).
* **11 further south test tiles** have verified observations but **no prediction**. The app lists
  them as *UNAVAILABLE* for investigation; no prediction is ever invented.

## Install, test and launch (WSL / Linux, Python 3.11–3.12)

```bash
cd ~/IceClaw
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q          # offline; no API key needed
ruff check . && ruff format --check .
python -m streamlit run app.py
```

Open **http://localhost:8501** (the server runs headless; it does not open a browser itself).

Live investigations need `OPENAI_API_KEY`, `ICECLAW_LLM_MODEL=gpt-6-luna` and
`ICECLAW_LLM_PROVIDER=openai` in your environment or a local `.env` (see `.env.example`, which holds
empty placeholders only). Never commit `.env`. Recorded replay, verification and downloads need no
key.

## Recorded vs live investigations

| | Recorded (default) | Live |
| --- | --- | --- |
| Source | A saved **public-redacted** bundle in `runs/` (default `inv-20261031T201513Z-p6-public`) | A new run of the four specialists |
| Model calls | **None** — replayed from saved events, labelled `RECORDED INVESTIGATION` | Real, paid OpenAI calls (budget: ≤ 28 model calls, 150k input tokens) |
| Starts | Automatically on page load | Only after ticking a cost confirmation and clicking **Investigate**; one run at a time; refreshes never start another |
| Output | Shown as saved | Verified private bundle → published public-redacted bundle, which the page then loads |

The page shows the region's real layers (Plotly heatmaps in south-polar stereographic km), the
four agents' activity and any real cross-agent follow-up, the recorded measurements with units,
AI hypotheses and open questions, the investigation-attention ranking (explicitly **not** ice
prospectivity), and **Verify findings** / **Download public bundle**. Details:
[docs/WORKBENCH.md](docs/WORKBENCH.md).

## Verify and export an investigation (command line)

```bash
python -m iceclaw.cli verify --run inv-20261031T201513Z-p6-public   # offline; no key, no GPU
python -m iceclaw.cli replay --run inv-20261031T201513Z-p6-public   # events only; 0 model calls
python -m iceclaw.cli investigate --region ice-s-0001-0002           # LIVE, paid
python -m iceclaw.cli publish --run <private-bundle>                 # public-redacted copy; no model call
```

Bundles, the verifier's 23–26 checks and the public-redacted profile are described in
[docs/INVESTIGATION_BUNDLE.md](docs/INVESTIGATION_BUNDLE.md). Only public-redacted bundles are
offered for download: they contain no provider-opaque reasoning data and no internal run database.

## A fresh clone is NOT demo-ready by itself

`data/` and `runs/` are ignored by Git (source data, private evaluation material and run logs are
never committed). In a fresh clone the app starts, but with no saved bundle and no local catalog it
shows "No regions are available". To prepare a demo machine safely:

1. **Recorded demo (no key, no downloads):** copy the public-redacted bundle directory
   `inv-20261031T201513Z-p6-public` (≈ 1.1 MB; it embeds the byte-exact SomBench GeoTIFFs,
   CC BY 4.0, and the NASA–IBM prediction raster, Apache-2.0, with attribution) from a trusted copy
   into `runs/`, then confirm it before launching:
   `python -m iceclaw.cli verify --run inv-20261031T201513Z-p6-public` → `RESULT: VERIFIED (26 passed …)`.
   Copy **only** the `-public` bundle — never the private bundle (it contains an internal run
   database and opaque provider data) and never `.env`.
2. **Region catalog and live runs:** install the pinned MOONATLAS checkout (commit
   `945fb5196679cf6927354c53be624d745083f9e9`, see [docs/UPSTREAM_MOONATLAS.md](docs/UPSTREAM_MOONATLAS.md))
   into `.upstream/moonatlas-science`, then download the public SomBench inputs at the pinned
   revision: `python scripts/prepare_sombench.py --patches-from-split test --pole south --layers inputs`
   (checksums verified; see [docs/SOMBENCH_INGESTION.md](docs/SOMBENCH_INGESTION.md)).

The P7 benchmark suites, sealed keys and raw benchmark logs also live under the ignored folders;
only their scored summaries, configs, registrations and frozen-output hashes are in
[docs/p7_results/](docs/p7_results/) and [benchmarks/](benchmarks/). External parties can audit the
scoring and the hashes but cannot re-score from raw logs without those private files.

## P7 result: communication did not help (exploratory)

Pre-registered v2 protocol ([docs/BENCHMARK_P7.md](docs/BENCHMARK_P7.md)): 6 frozen synthetic cases
(2 clean, 4 faults) built from the one paired tile; equal total budget per case for every LLM
approach (16 model calls, 16 tool calls, 64k input tokens) with a guaranteed minimum per specialist;
one pass; outputs frozen before labels were opened.

| Approach | Faults found | F1 | False alarms | Incomplete | Model calls | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Deterministic rules | 3/4 | 0.857 | 0 | — | 0 | 0 |
| One generalist agent | 2/4 | 0.667 | 0 | 0 | 39 | 170 586 |
| Four isolated specialists | 3/4 | 0.857 | 0 | 0 | 87 | 293 612 |
| Four communicating specialists | 2/4 | 0.667 | 0 | 1 case (2 verdicts) | 87 | 351 538 |

The LLM comparison used 213 model calls and 815 736 tokens. No challenge, revision, test request
or verdict change occurred; communication did **not** improve detection, completeness or
efficiency. Six correlated synthetic cases from one real tile, one pass: no statistical
generalization, and synthetic faults are not lunar anomalies.

## Documents

1. [Project specification](docs/PROJECT_SPEC.md) — scientific objective, contracts, evidence standard.
2. [Engineering implementation plan (rev. 3.1)](docs/IMPLEMENTATION_PLAN.md) — phases P0–P8, gates; tool-independent.
3. [AGENTS.md](AGENTS.md) — durable repository rules ([CLAUDE.md](CLAUDE.md) points to them).
4. [Workbench](docs/WORKBENCH.md), [bundles and verification](docs/INVESTIGATION_BUNDLE.md),
   [agent runtime](docs/AGENT_RUNTIME.md), [evaluation protocol](docs/EVALUATION_PROTOCOL.md),
   [P7 benchmark](docs/BENCHMARK_P7.md).

`docs/SCIENTIFIC_PROTOCOL.md` and `docs/DEMO_2MIN.md` are historic notes; prefer the spec and plan.

## Foundations

- [NASA–IBM Lunar Foundation Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model) — the ice-prospectivity model (published output used; no fresh checkpoint inference).
- [MOONATLAS Science](https://github.com/RRG1312/moonatlas-science) — independent project providing the georeferenced published prediction (not operated or endorsed by NASA/IBM).
- [SomBench Ice-Prospectivity-Regression](https://huggingface.co/datasets/nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression) — original TMAX/LPSR/SLOPE observations (CC BY 4.0).
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) — the scientific agents' runtime (OpenAI provider).

Stack: Python 3.11/3.12, NumPy, Rasterio, Pillow, Pydantic, SQLite, Microsoft Agent Framework,
Streamlit + Plotly. MOONATLAS runs in its own pinned environment; IceClaw consumes its validated
artifacts.

**Hackathon timeline:** final slides due Nov. 1, 2026, 1:00 PM ET per the [competition site](https://scienceclawhack.ai/#competition).
