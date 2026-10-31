# IceClaw

**ScienceClaw 2026 — MIT Media Lab**

**Status: architecture and scientific protocol only. No scientific results or running demo are claimed yet.**

IceClaw is a research-oriented, decentralized collective of AI planetary scientists that audits predictions of lunar polar ice prospectivity, tests alternative explanations with real NASA observations, and prioritizes follow-up investigations.

## Research question

**Does sharing evidence among autonomous specialist agents improve the detection and explanation of unreliable lunar ice-prospectivity predictions, compared with a single agent or isolated specialists at the same tool/compute budget?**

Ice prospectivity is an *expert-derived index*, **not** measured lunar ice, a calibrated ice probability, or direct proof of a deposit. We must never label a high score as a discovered deposit.

## Three upstream foundations

1. [NASA–IBM Lunar Foundation Model](https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model): pretrained scientific model and ice-prospectivity checkpoint. [Model/dataset on Hugging Face](https://huggingface.co/nasa-ibm-ai4science).
2. [MOONATLAS Science](https://github.com/RRG1312/moonatlas-science): pinned acquisition, running published checkpoints, geolocation, normalization, validation, provenance and sample fixture. Independent project, not endorsed by NASA/IBM.
3. [Microsoft Agent Framework](https://github.com/microsoft/agent-framework): independent tool-using scientific agents, parallel workflows, handoffs, streaming, tracing.

IceClaw owns the scientific investigation protocol, independently acting agents, evidence events, evaluation, API and web research console. No upstream names/logos imply endorsement.

## Minimal usable product

- **Input:** one verifiable held-out *south-polar* SomBench tile (8 source modalities, 256×256 at 240 m/px); NASA–IBM prediction; MOONATLAS geolocation and provenance.
- **Validation:** MOONATLAS science-smoke validator on its *published real example*; full inference on an official held-out sample after checkpoints are available.
- **Processing:** verify shape, coordinate reference system, missingness, raster provenance and scientific data split.
- **Agents:** thermal scientist, geology scientist, model auditor, skeptic/experiment selector. They exchange structured findings asynchronously; no central model invents all next steps.
- **Output:** interactive lunar map (raw model output and reference clearly separated), evidence log, disagreement or anomaly investigations, ranked *follow-up priorities*, and reproducible evaluation plots.
- **Benchmark:** deterministic rule baseline, one agent with all tools, isolated agent team, communicating swarm; fixed information and call budgets. Expert reference maps stay hidden from agents until evaluation.
- **Delivery:** working scientific analysis + usable UI + source code + 2-minute presentation with an actual measured result, even if improvement over baseline is absent.

Do not claim data covers Cabeus or Shackleton until an exact tile footprint has been verified. The first example should use its actual coordinates.

## Repository structure (target)

```text
iceclaw/
├── app/                 # Streamlit UI: map, agent events, benchmark
├── iceclaw/
│   ├── data/            # MOONATLAS adapter; integrity/provenance validation
│   ├── science/         # deterministic numerical and geospatial analysis
│   ├── agents/          # Microsoft Agent Framework agents/tools
│   ├── events/          # event schema and evidence store
│   └── evaluation/      # no-leakage baselines and ablations
├── tests/               # scientific invariants + event protocol tests
├── docs/
│   ├── SCIENTIFIC_PROTOCOL.md
│   └── DEMO_2MIN.md
└── artifacts/           # generated (gitignored unless small and licensed)
```

Start with Python 3.11/3.12, NumPy/Rasterio/pyproj, Microsoft Agent Framework, SQLite and Streamlit/Plotly. Favor minimal working infrastructure over Redis/Postgres, fine-tuning or GPU orchestration.

## First reproducibility gate

From an independently cloned MOONATLAS Science checkout (Python 3.11/3.12):

```bash
pip install -e ".[dev]"
pytest
moonatlas-science validate --root examples/science-smoke --scope published
```

These validate an upstream *published fixture*; they are **not a new IceClaw inference run**.

For actual inference, follow that project's `python scripts/setup_environment.py` and `moonatlas-science smoke`. It documents ~6.3 GB of model/task checkpoints, GPU recommended but CPU possible.

## Hackathon constraints

Official [ScienceClaw competition](https://scienceclawhack.ai/#competition): 25% scientific/technical impact, 25% collective capability, 20% problem significance, 20% decentralized agency, 10% execution/validation; +10 possible collaboration bonus. Slides due Nov 1, 2026, 1:00 PM ET.

Read [scientific protocol](docs/SCIENTIFIC_PROTOCOL.md) and [presentation story](docs/DEMO_2MIN.md) before implementation.
