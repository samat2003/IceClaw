> **ARCHIVED — superseded pre-implementation draft (kept for history).** It describes plans and placeholders, not results; some envisioned behaviour (for example agents challenging and revising each other) did not occur in the measured runs. For the current state see the [README](../../README.md), [PROJECT_SPEC](../PROJECT_SPEC.md) and [BENCHMARK_P7](../BENCHMARK_P7.md).

# IceClaw: 2-minute judge presentation

**Draft; insert actual benchmark measurements after the evaluation runs. Never claim unrun experiments or unexplained detections.**

## Narrative and slide pacing

| Elapsed | Visual | Key statement |
|---|---|---|
| 0:00–0:20 | Slide 1: lunar south-pole image | The scientific problem: lunar ice prospectivity is uncertain; no single signal proves ice. |
| 0:20–0:40 | Slide 2: 3 upstream components + original IceClaw layer | NASA model produces predictions; MOONATLAS validates georeferenced source data; IceClaw independently audits findings through a decentralized swarm. |
| 0:40–1:15 | Slide 3: live/replayed web investigation | Click actual held-out region; show real lunar coordinates and layered model/reference/measurement data; two agents challenge each other; show sourced follow-up recommendation. |
| 1:15–1:40 | Slide 4: evaluation bars/table with REAL values | Under the same data and call budget compare heuristic, single, isolated, swarm. State the measured result honestly (including failures). |
| 1:40–2:00 | Slide 5: conclusion and reuse | Validated research artifact, provenance, open-source agent/evidence interface and future investigation utility. |

## Rehearsal script (~240 words; fill brackets only with measured facts)

'Where should scientists investigate for water ice on the Moon? The challenge is not just predicting a promising region. It is knowing which predictions are trustworthy and what evidence is still missing.

NASA and IBM released a lunar foundation model trained on enormous collections of observations. MOONATLAS makes its output georeferenced and reproducible. But an ice-prospectivity score is not direct evidence of ice.

We built IceClaw on top of these open-source foundations: a decentralized collective of AI planetary scientists. Thermal, geological and prediction specialists work independently. They publish evidence to a shared research log, challenge each other's assumptions, and choose follow-up tests rather than waiting for a central planner.

Here is a real held-out south-polar tile. The map is tied to its source and coordinates. The [agent] raised [finding]. Another agent examined [measurement or alternative explanation], so the collective revised [its hypothesis or research priority]. Every displayed numerical claim links back to the processed scientific data.

We also tested whether collaboration actually helps. Under equal budgets, our [number] experiments show [measured comparison and uncertainty]. We report failures as well as successes, and never call a prediction a confirmed ice deposit.

IceClaw turns existing lunar AI predictions into an auditable, autonomous scientific investigation. Its event and evidence interface can be reused by other research teams. The goal is not to replace planetary scientists, but to help them ask better questions of the data.'

## Demo fail-safe

Have a fully recorded, provenance-backed deterministic replay of an actual completed investigation. If API/model access fails onstage, replay the exact saved event log and open the same source artifacts; label replay clearly.

## Don't claim

- We found a new ice deposit.
- NASA/IBM officially endorse IceClaw.
- A prospectivity score is an ice probability.
- The swarm beat a baseline before measurements exist.
- A smoke fixture is a fresh checkpoint run.

## Required screenshot/artifacts before slides are final

- exact source record and split + source hash;
- working map + selected region;
- independent scientist events with traceable tool results;
- score comparison with methods, budget and seed;
- repository URL and reproducible command line.
