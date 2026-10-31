# Specialist collective on Microsoft Agent Framework (P5)

Everything below describes code and one recorded live run. The scientific outputs are
screening statistics on a prospectivity index; nothing here detects, confirms or quantifies ice.

## 1. Stack (verified by import/execution probes, 2026-10-31)

| Package | Version | Used for |
| --- | --- | --- |
| `agent-framework-core` | 1.24.0 (pinned) | `Agent`, `@tool`, function-invocation loop, `BaseChatClient` |
| `agent-framework-openai` | 1.18.0 (pinned) | `OpenAIChatClient` (OpenAI Responses API) |
| `openai` | 3.42.0 (resolved) | transport |

API facts confirmed before building: `Agent(client, instructions, name, tools, default_options)`;
`@tool` builds the JSON schema from the signature; `FunctionInvocationConfiguration`
(`max_iterations`, `max_function_calls`) bounds each turn's tool loop; **sync tool functions run in
worker threads** (our tools are therefore `async`, see §4); MAF reads a `.env` file only if a path
is passed explicitly, and IceClaw never passes one. Model ID and key come only from IceClaw
settings (`ICECLAW_LLM_MODEL`, `OPENAI_API_KEY`); a missing value gives `MISSING_API_KEY` /
`MISSING_LLM_MODEL` before any client is built. Calls use `store=False`.

## 2. Architecture

```
swarm/store.py      append-only SQLite: events, event_parents, tool_runs, llm_calls (+ JSONL export)
swarm/runtime.py    validated actions, event bus, budgets, hop limit, quiescence, supervisor record
agents/provider.py  InstrumentedOpenAIChatClient (real) / ScriptedChatClient (fake): meter + record
                    every model round-trip below MAF's tool loop
agents/tools.py     8-9 MAF function tools per agent (thin closures over the runtime)
agents/specialists.py  four roles, permissions, prompts, Agent construction
```

* **Four independent specialists**, one real MAF `Agent` each: Thermal Scientist, Geology
  Scientist, Prediction Auditor, Skeptic/Investigation Planner. One provider/model for all roles.
* **No central LLM, no scripted order.** Thermal, Geology and Auditor start concurrently; the
  Skeptic has no data of its own and takes no turn until another agent publishes a claim.
* **Role permissions** (enforced in the runtime, not by prompt): Thermal — `pixelwise_temperature`,
  `pixelwise_shadow_mask`, `thermal_psr_consistency` + generic stats; Geology — `pixelwise_terrain`,
  `terrain_slope_summary` + generic; Auditor — `nasa_ibm_prediction_raw`, all prediction tools
  (`boundary_check`, `peak_location`, `unit_interval_check`, ...); Skeptic — every layer/tool plus
  `recompute_measurement`. Tile scalars and single-anchor metadata are not exposed as arrays.
* **Actions** (the only way to act): `run_science_tool`, `publish_claim`, `publish_challenge`,
  `request_test`, `publish_revision`, `conclude`, `list_events`, `get_event`
  (`recompute_measurement` for the Skeptic).
* **Authenticity:** every number in the log comes from a deterministic IceClaw tool executed by the
  runtime and stored as a `ToolMeasurement` (source file ids + SHA256, unit, parameters, valid
  pixels). Agents cannot write numbers into events. Claims citing numbers must cite measurement
  event ids (`EVIDENCE_REQUIRED` otherwise; `hypothesis` is the label for untested ideas).
* **References:** a challenge targets another agent's claim/revision/result and cites parents; a
  revision must be of the agent's own claim and cite ≥ 1 measurement/test_result published
  **after** that claim (`NEW_EVIDENCE_REQUIRED`); a test result links to the request it answers.
  Hop depth = 1 + max(parent hops); the store re-verifies parents, hops and tool-run links.
* **Delivery:** every event goes to every other agent's inbox. An agent is woken (new turn) only
  by: a claim/revision (Skeptic only), a challenge of its event, a test request addressed to it,
  or a result of its request. Everyone sees the full digest at their next turn.
* **Budgets** (`RunBudget` + round cap): model calls (hard, counted per real round-trip, raised
  before any spend), tool calls (refused with `TOOL_BUDGET_EXHAUSTED`; identical calls are cached
  and free), events (`EVENT_BUDGET_EXHAUSTED`; the supervisor's closing event is exempt), hops
  (`HOP_LIMIT`), wall time (cancels all tasks), rounds per agent, per-turn MAF caps.
* **Termination** is deterministic: nobody running and nobody woken → `QUIESCENT`; otherwise
  `BUDGET_EXHAUSTED` or `TIMEOUT`. A neutral `supervisor` completion event records why. A failed
  turn is recorded as a `failure` event and retried once.
* **Region gate:** `load_paired_region` accepts only a verified `PAIRED` region. The 11
  observation-only tiles raise `PREDICTION_UNAVAILABLE`; prediction-only raises
  `REGION_NOT_INVESTIGABLE`.

## 3. Secrecy / firewall (tested)

Agents receive an `AgentRegionContext`, the prompt digest and the tool results; no path, file
handle, `RegionCatalogEntry`, `PRO`, reference raster, sealed manifest, fault label or evaluator
root. Tool schemas contain no path/file/dir parameters; `agents/` and `swarm/` import nothing from
`iceclaw.evaluation` or `os/pathlib/open`; prompts, stored model requests and events contain none
of `PRO`, `reference`, `sealed`, `evaluation`, `fault`, `salt`, `/home/` or key-like strings. The
API key never enters messages or stored records.

## 4. Concurrency note (a real bug found and fixed)

MAF executes synchronous tool functions in worker threads. The first offline run produced
spurious `SystemError`s because the runtime's counters and SQLite connection were being used from
several threads. All tools are now `async` wrappers, so every action runs atomically on the event
loop thread; a regression test asserts this.

## 5. Tests

| Suite | Count | What it proves |
| --- | --- | --- |
| `tests/test_swarm.py` | 30 | store immutability (every table, incl. raw SQL), validation, role permissions, parameter bounds, dedup, ceilings, hops, delivery |
| `tests/test_agents.py` | 23 | real MAF agents + scripted fake model: routing, concurrency, cross-agent challenge→revision, request→result, no-conflict runs, LLM/tool/event/hop/round/timeout ceilings, retry, tool schemas, secrecy, real-region numbers |
| `tests/test_live_agents.py` | 1 (opt-in) | real OpenAI round-trip with a real tool call |

A scripted fake **does not prove live operation**; the live test and run below do.

## 6. Baseline false-positive limitation and the v2 configuration

`p4-multimodal-v1` (frozen) with baseline config **v1** scored TP 7, FP 1, FN 1, TN 3. The one false
positive is `clean_quantization_noise`: v1's `shadow_tol = 1e-6` equals the control's own noise
(σ = 1e-6), so rule `shadow_range` flagged a clean tile. That is a mis-specified tolerance, not a
physical finding. Also `temperature_spatial_roll` was pre-registered as undetectable but was
detected.

* v1 (`ObsBaselineConfig`) and every frozen v1 result are **unchanged** (tested).
* **v2** (`ObsBaselineConfigV2`, `OBS_BASELINE_CONFIGS["v2"]`) changes exactly one value,
  `shadow_tol` 1e-6 → 1e-4 (≈100× the noise and float32 round-off, far below a semantic mask
  corruption), plus `version="2"`. It still flags a mask scaled ×1.5 (tested).
* v2 was **not tuned or scored on the sealed v1 labels**. A comparative benchmark (P7) must use a
  **new suite frozen after this config**; reporting v2 on `p4-multimodal-v1` would be post-hoc.

## 7. The recorded live run

Command: `python scripts/p5_live_investigation.py --live` (run after a passing dry run and the
opt-in live test). Region: `ice-s-0001-0002` (verified PAIRED, nothing else was investigated).
Outputs (git-ignored): `runs/inv-20261031T192018Z-p5live/` (`events.sqlite`, `events.jsonl`,
`tool_runs.json`, `llm_calls.jsonl`, `run_summary.json`).

| Item | Value |
| --- | --- |
| Provider / model | `openai` / `gpt-6-luna` (from the user's IceClaw settings; not guessed) |
| Limits | 28 model calls · 16 tool calls · 40 events · hop ≤ 5 · 420 s · 3 rounds per agent |
| Outcome | `QUIESCENT` in **30.5 s**; offline `store.verify()` problems: none |
| Model calls | **22**, all successful (thermal 6, geology 5, auditor 4, skeptic 7) |
| Tokens | 86 411 in / 4 638 out / 91 049 total |
| Cost | **not computed** (no verified price list supplied; pass `--price-in/--price-out`) |
| Tool calls | **15** (auditor 8, thermal 3, geology 2, skeptic 2); 0 refusals recorded as failures |
| Events | **24**: 15 measurement, 4 claim, 5 completion (incl. 1 supervisor); 0 failure, 0 challenge, 0 test_request, 0 test_result, 0 revision |
| Sources cited | prediction raster (`dbedcc288a59…`), `TMAX` (`054b20b2f49b…`), `LPSR` (`76e4036a7f5b…`), `SLOPE` (`b858a1281ca9…`) |
| Provider/runtime failures | none |

What the agents did (from the saved log): Thermal measured PSR vs non-PSR temperature
(139.215 K vs 246.989 K, Δ −107.774 K; 6.54 % of PSR pixels below its own 110 K screening
threshold), temperature quantiles and gradients; Geology measured slope statistics (mean 8.734°,
p95 20.416°, 15.2 % above its own 15° threshold) and gradient discontinuities; the Auditor ran
eight prediction tools (825 values outside [0,1], peak at row 194 col 33, no missing data);
all values equal the independently verified P2/P4 numbers. Each published a claim citing its
measurement events, with alternatives and limitations (e.g. "not a validated physical constant",
"does not establish ice").

**Cross-agent behaviour, stated plainly:** the Skeptic, woken by the specialists' claims, used the
Auditor's published peak location (evt-0011) to run **its own** windowed thermal and prediction
measurements around that peak (evt-0018/0019), published a claim whose parents are the Auditor's
and its own events, and concluded that **no contradiction was warranted**. There was **no
challenge, no test request and no revision**: no natural conflict arose, so none was invented.
This satisfies "evidence-triggered cross-agent follow-up" only in the weak sense of one agent
acting on another's published evidence; the challenge→revision path is demonstrated by the offline
scripted tests, **not** by this live run.

Caveats: one run, one region, one model, non-deterministic wording; the temperature/shadow layers
and the prediction share upstream modeling assumptions, so agreement is not independent
validation. The live integration test made at most 4 additional model calls with an in-memory
store (not retained).

## 8. Known limitations

* No report/verification bundle (P6), no benchmark (P7), no UI (P8).
* Cost is reported as tokens only; the prompt grows with the digest (skeptic ≈ 35k input tokens).
* Wake rules are deliberately conservative (claims wake only the Skeptic); specialists see other
  specialists' claims at their next turn and react when challenged, asked or answered.
* Single paired region: `PARTIAL_ONE_REGION`.
