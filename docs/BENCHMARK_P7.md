# P7 — Does communication between four AI scientists help? (paired-tile benchmark)

**Research question.** Under equal *total* model-call, token and tool-call budgets, does
communication between four independent AI scientists improve the detection and investigation of
synthetic faults in authentic lunar scientific data?

**Status (final):** v1 = frozen 12-case suite + one-case live feasibility pilot (sections 1–7).
**v2 = pre-registered 6-case evaluation, executed once (sections 8–13).** Result: communication did
**not** improve detection, completeness or efficiency — deterministic rules and isolated
specialists F1 0.857 (3/4 faults), generalist and communicating specialists F1 0.667 (2/4), no
false alarms, one incomplete communicating case; 213 model calls, 815 736 tokens. Exploratory only:
six synthetic cases from one real tile, one pass, no statistical generalization.

## 1. Frozen suite `p7-paired-v1` (`src/iceclaw/evaluation/paired.py`)

* Source: copies of the ONE verified predicted+observed tile `ice-s-0001-0002` — the published
  NASA–IBM prediction (`upstream_published_example`) plus the genuine SomBench TMAX (K),
  LPSR (fraction) and SLOPE (deg). The expert target `PRO` is not used and never enters either root.
* 12 cases = 4 clean controls + 8 synthetic faults (pre-registered in code before any run):

| Fault | Intent | Baseline expected to detect |
| --- | --- | --- |
| `clean_identical`, `clean_identical_repeat` | unmodified copies | (clean) |
| `clean_float32_roundtrip` | benign float32 round-trip | (clean) |
| `clean_subthreshold_noise` | 1e-6 index, 1e-3 K, 1e-4 deg noise | (clean) |
| `prediction_percent_scale` | index ×100 | yes |
| `prediction_offset_small` | index +0.03 | no |
| `prediction_misregistered` | prediction shifted 15–24 px vs observations | no (needs cross-layer comparison) |
| `prediction_nan_block` | covered pixels NaN, mask not updated | yes |
| `temperature_unit_celsius` | TMAX in °C | yes |
| `shadow_mask_inverted` | LPSR → 1 − LPSR | yes |
| `temperature_misregistered` | TMAX shifted 25–34 px vs other layers | yes (PSR contrast collapse) |
| `slope_unit_radians` | SLOPE in radians | yes |

* Firewall as P3/P4: agent-visible `data/scenarios/p7-paired-v1/<case-NN>/` (4 layers + meta);
  evaluator-only `data/evaluation/p7-paired-v1/` (fault ids, parameters, per-case seeds,
  expectations, **private random suite seed**, salt); public manifest with a salted commitment.
  Frozen manifest SHA256 `28035f1e773d430975af6c3e531e5543ed764d218b3016c154150b90f0378796`.
* P3 `p3-prediction-only-v1` and P4 `p4-multimodal-v1` code and suites are unchanged.
* All cases share one tile's pixels: they are **not independent samples**; synthetic faults are
  **not genuine lunar anomalies**.

## 2. Four arms, one task, one total budget

| Arm | Agents | Communication |
| --- | --- | --- |
| `deterministic_rules` | none — `PairedBaselineConfig` (paired-v1) rules over the P2 tools | — |
| `single_generalist` | 1 MAF agent, all layers and tools | — |
| `isolated_specialists` | Thermal, Geology, Prediction Auditor, Skeptic (P5 permissions) | **off**: no delivery, cannot read/cite each other, no `request_test`/`publish_challenge` |
| `communicating_specialists` | same four | **on**: P5 event bus (claims wake the Skeptic; requests, challenges, revisions) |

* Same task text, same science-tool implementations, same assessment tool for all LLM arms.
* **Budget per case and arm (TOTAL, never per agent):** 16 model calls, 16 tool calls,
  48 events, hop ≤ 5, 360 s, 64 000 provider-reported input tokens (pre-call reservation).
  A test proves four agents together cannot exceed a total of 3 calls when the cap is 3.
* Per-turn caps differ only so every arm *can* reach the same total: the single agent may spend
  all 16 calls within one turn; specialists have 6 round trips per turn × up to 3 turns each.
* Model: OpenAI `gpt-6-luna` through Microsoft Agent Framework (from the user's environment;
  the runner refuses to start if the configured model differs from the benchmark config).
* Fresh `SwarmRuntime`, SQLite event store and MAF clients (`store=False`) for **every case and
  arm**; no context is carried between cases (tested).

## 3. Pre-registered machine-readable decision

`submit_assessment(verdict ∈ {fault_suspected, no_fault_found}, suspected_layers, cites ≥ 1
measurement, rationale)` writes an `ASSESSMENT {json}` completion event. **Case flagged iff at
least one agent's latest verdict is `fault_suspected`**; no assessment → not flagged and reported
as `decision_missing`. An agent that ends a turn without an assessment is woken once more (same
rule in every arm, bounded by its round limit). The deterministic arm flags iff any rule fires.

Note: the "any agent" rule favours recall over false-positive rate for the four-agent arms
relative to the single agent; this is a property of the declared rule, reported, not tuned.

## 4. Freeze, then score (`src/iceclaw/benchmark.py`)

Per case: original `events.jsonl`, `llm_calls.jsonl`, `tool_runs.json`, `run_summary.json`,
`decision.json`. `freeze` records the decisions (`FrozenDetections`, optionally over a DECLARED
subset) and the SHA256 of every output file. `score` opens the sealed key only afterwards and
refuses if any output changed after freezing. Metrics: case-level TP/FP/FN/TN, precision, recall,
F1, clean false-positive rate (undefined = `null`), plus model calls, tokens, tool calls,
elapsed time. The pixel is never the unit; no confidence intervals.

```bash
python -m iceclaw.cli benchmark --config benchmarks/p7_pilot.json --action build-suite --eval-root data/evaluation
python -m iceclaw.cli benchmark --config benchmarks/p7_pilot.json --action estimate
python -m iceclaw.cli benchmark --config benchmarks/p7_pilot.json --action run --confirm-paid
python -m iceclaw.cli benchmark --config benchmarks/p7_pilot.json --action freeze
python -m iceclaw.cli benchmark --config benchmarks/p7_pilot.json --action score --eval-root data/evaluation
```

## 5. Results (measured; exported to `docs/p7_results/`)

### 5a. Live feasibility pilot — ONE case (`case-01`), all four arms

`case-01` was declared before the suite was built (first case in manifest order). After freezing,
scoring revealed it is the clean control `clean_float32_roundtrip`.

| Arm | Verdict(s) | Outcome | Model calls | Input / output / total tokens | Tool calls | Elapsed | Ended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deterministic_rules | no rule fired | TN | 0 | 0 | 10 | 0.008 s | — |
| single_generalist | no_fault_found | TN | 6 | 22 097 / 1 562 / 23 659 | 16 | 20.7 s | QUIESCENT |
| isolated_specialists | 3 × no_fault_found (Thermal filed none) | TN | 16 | 48 553 / 3 296 / 51 849 | 16 | 11.9 s | BUDGET_EXHAUSTED |
| communicating_specialists | 4 × no_fault_found | TN | 15 | 57 032 / 2 882 / 59 914 | 16 | 14.8 s | BUDGET_EXHAUSTED |

Pilot total: **37 live model calls, 127 682 input + 7 740 output = 135 422 tokens** (provider-
reported; unpriced), all within the declared caps (≤ 16 calls and ≤ 64 000 input tokens per arm).

**What the pilot shows — and does not.** All arms correctly left a clean control unflagged.
Precision, recall and F1 are undefined (no fault case). This is a runner/cost/latency check: it is
**not** evidence that communication helps, hurts or makes no difference.

Observed behaviour (from the original logs, not inferred):

* Every LLM arm used all 16 tool calls. The single generalist batched tools and needed only 6 model
  calls (2.2–2.5× fewer tokens than the four-agent arms).
* **Isolated arm: the shared tool budget was drained by three concurrent agents before Thermal ran
  any measurement**; Thermal then exhausted the shared model-call budget and filed no assessment
  (3 of 4 agents decided). Equal *total* budgets plus uncoordinated concurrency starve agents.
* Communicating arm: all four agents assessed; the Skeptic's assessment cited Thermal's and the
  Prediction Auditor's measurements (3 cross-agent references). **No test request, challenge or
  revision occurred.** The Skeptic hit the shared input-token cap.

### 5b. Deterministic baseline on all 12 cases (no model calls)

TP 6, FP 0, FN 2, TN 4 → precision 1.0, recall 0.75, F1 0.857, clean FPR 0.0. Misses are exactly the
two faults pre-registered as undetectable (`prediction_offset_small`, `prediction_misregistered`);
no mismatch with the pre-registration. Same author wrote rules and expectations: a plumbing check.

### 5c. Budget for a larger experiment (report before running)

| Scope | Model calls | Tokens |
| --- | --- | --- |
| Measured per case, all three LLM arms | 37 | 135 422 (127 682 in) |
| Projected: 12 cases × 3 LLM arms (one repeat) | ≈ 444 | ≈ 1.63 M (≈ 1.53 M in) |
| Hard cap: 12 cases × 3 LLM arms | 576 | 2.30 M input (+ soft output) |
| Each additional repeat of all 12 cases | ≈ 444 (cap 576) | ≈ 1.63 M (cap 2.30 M input) |

One pilot case is a weak basis for projection: fault cases may trigger more turns (up to the caps).
Sequential wall time ≈ 47 s per case for all three LLM arms (≈ 10 min per 12-case pass).

## 6. Limitations

* One genuine paired tile; 12 synthetic variants of it share pixels — no generalization, no
  confidence intervals, case is the unit. Synthetic faults are not lunar anomalies.
* The "any agent flags" rule structurally favours recall (and risks false positives) for the
  four-agent arms; it was declared in advance and is applied identically to both four-agent arms.
* The task text lists example fault types (identical for every LLM arm); the deterministic rules
  were written by the same author who designed the faults.
* The frozen-detections hash binds flags, subset and suite; per-arm identity and output integrity
  are bound by the per-arm frozen record (SHA256 of every original output file).
* **The developer has now seen all `p7-paired-v1` labels** (pilot + baseline scoring). Agents
  never had access, but any change to the LLM arms made from here on is label-informed.

## 7. Readiness for a larger experiment

Runner, isolation, budgets, freezing and scoring work end-to-end live. Before a full paid run,
decide and freeze (new config version, ideally a **new suite `p7-paired-v2` with a fresh private
seed**, so design changes are not label-informed):

1. **Tool-budget allocation in four-agent arms.** The pilot shows starvation under a single shared
   pool. A pre-declared equal split (e.g. 4 tool calls per agent, total still 16) keeps totals equal
   while letting every specialist measure; or keep the shared pool and report starvation as part of
   the result. Either choice must be fixed before the run.
2. **Input-token cap** (64 000): binding for both four-agent arms in the pilot; keep (equal-budget
   principle) or raise for all arms equally.
3. **Repeats:** model sampling varies run to run; ≥ 3 repeats per case would cost ≈ 4.9 M tokens.
4. Expected spend for the full 12-case pass: ≈ 444 calls / ≈ 1.6 M tokens (caps 576 / 2.3 M
   input). Not launched: requires explicit approval.

---

# P7 protocol v2 (pre-registered, then executed once — results in section 13)

v1 (`p7-paired-v1`, the one-case pilot and all v1 scores) is preserved unchanged; its configs
still hash to their original output folders (`663766aa59f1`, `1a6047ac4711`) and every frozen v1
output re-verifies.

## 8. Fairness review of the v1 implementation

| # | Issue found in v1 | Effect | v2 handling |
| --- | --- | --- | --- |
| 1 | One shared tool/model/token pool, first come first served | Pilot: Thermal never measured (isolated arm) | Per-agent **reserve** inside the same totals |
| 2 | A team with a missing member still counted as a complete "not flagged" decision (`decision_missing` only if *nobody* assessed) | Silence was indistinguishable from "no fault found" | Explicit `incomplete` individual verdict and `INCOMPLETE` team decision |
| 3 | Communicating Skeptic starts idle and only wakes on claims | Could never act if no claim appeared or budget ran out | Reserve held until it assesses + one guaranteed **final turn** |
| 4 | Event budget could be exhausted before an agent's assessment | Assessment refused | One reserved **assessment event slot** per agent |
| 5 | "Any agent flags" rule | Favours recall (and false positives) of four-agent arms | Kept and pre-registered; complete-case secondary metrics added |
| 6 | Per-turn caps differ between arms | Needed so each arm *can* reach the same total | Kept; documented |
| 7 | Isolated agents' cache hits | Already charged in v1 (no free cross-agent benefit) | Unchanged |
| 8 | Developer saw all v1 labels | Design changes are label-informed | Fresh suite `p7-paired-v2`, new private seed, rules registered first |
| 9 | Task text lists example fault types; baseline author designed the faults | Equal across LLM arms; baseline is a plumbing check | Disclosed |

## 9. v2 rules (registered before the suite existed)

Files: `benchmarks/p7_v2.json` (config SHA256 `ab80de50562ba928eecfb0c0ba40fa493e069703d8b1f3307a6e4a214c10ecf3`),
`benchmarks/p7_v2_preregistration.json` (registered 2026-10-31 22:33:45 UTC, `suite_generated: false`),
then `benchmarks/p7_v2_suite_frozen.json` (suite generated 13 s later).

* **Budget per case and LLM arm (TOTAL, identical for all arms):** 16 model calls, 16 tool calls,
  48 events, hop ≤ 5, 360 s, 64 000 input tokens — the same totals as v1.
* **Reserve (isolated and communicating arms only):** each of the four specialists is guaranteed
  3 model calls, 3 tool calls, 12 000 input tokens and 1 assessment event slot
  (reserved 12/16 calls, 12/16 tools, 48 000/64 000 tokens, 4/48 events; the rest is a common
  pool). Another agent can never spend an agent's unused reserve; it returns to the common pool when
  that agent files its first assessment or exits. Every agent that has not assessed gets one final
  turn before the run may end. The single generalist owns the whole budget (no reserve).
  Implemented in `SwarmRuntime` (`AgentReserve`; admission checks for model calls, input tokens,
  tool calls and events) and tested offline, including greedy scripted agents that starve Thermal
  under v1 rules but not under v2.
* **Verdicts:** individual = `fault_suspected` / `no_fault_found` (latest assessment, ≥ 1 cited
  measurement) / `incomplete` (none filed). Team = FLAGGED if any `fault_suspected`; NOT_FLAGGED if
  all `no_fault_found`; otherwise INCOMPLETE. Primary metrics count INCOMPLETE as not flagged;
  incomplete cases and incomplete individual verdicts are reported; secondary metrics use complete
  cases only. Baseline: FLAGGED iff a rule fires.
* **Cases (6):** `clean_identical`, `clean_float32_roundtrip`, `temperature_unit_celsius`,
  `shadow_mask_inverted`, `temperature_misregistered`, `prediction_misregistered`
  (1 single-layer unit error + 3 cross-layer faults). One repeat; no re-runs or exclusions;
  descriptive analysis only.
* **Isolation:** fresh runtime, event store and MAF clients per case and arm (unchanged).
* **Reporting:** detection metrics plus provider-reported input/output/total tokens, model calls,
  tool calls and elapsed seconds per case and arm.

## 10. v2 suite

`p7-paired-v2`, suite version 2, 6 opaque cases, manifest SHA256
`6db2c8d654da09913939700675e82ae616c48327dadb9dddd63462e206edd786`, sealed commitment
`8694d008…`; fresh random seed and salt in `data/evaluation/p7-paired-v2/` only (not opened). Same
firewall and the same copies of the one genuine tile as v1; `PRO` not used.

## 11. Expected budget for the v2 run (not launched)

| | Model calls | Tokens |
| --- | --- | --- |
| Hard cap (6 cases × 3 LLM arms) | 288 | 1 152 000 input (output soft) |
| Projection at v1 pilot rates | ≈ 222 | ≈ 0.81 M total |
| Projection if both four-agent arms reach the input cap (reserves make this likely) | ≈ 230–288 | ≈ 0.95–1.0 M total |
| Worst case incl. output (~6 % of input in the pilot) | 288 | ≈ 1.22 M total |

Elapsed: ≈ 5 min sequential at pilot speed (≈ 47 s per case for all three LLM arms); hard worst
case 18 runs × 360 s ≈ 108 min. Cost is unpriced (no verified rate).

## 12. Readiness

Ready to execute on separate authorization with
`python -m iceclaw.cli benchmark --config benchmarks/p7_v2.json --action run --confirm-paid`, then
`--action freeze`, then `--action score --eval-root data/evaluation`.

Residual limitations: the reserve sizes (3 calls / 3 tools / 12k tokens) are based on pilot usage
and have not been validated live under v2 — a reserve guarantees an *opportunity*, not a useful
measurement; 6 correlated cases from one tile support no statistical claim; cross-layer faults may
be missed by every arm, which would be reported as such.

## 13. v2 results (executed once, 2026-10-31; exported to `docs/p7_results/v2_*`)

Pre-execution checks passed: 360 offline tests, Ruff, config SHA256 = registered, manifest SHA256 =
frozen record, salted commitment verified, registration (22:33:45 UTC) before suite (22:33:58 UTC),
agent-visible case objects contain no labels, seeds, tile ids or `PRO`. Run: one pass, 6 cases ×
4 approaches, exit 0; all outputs frozen (with SHA256 of every log) before the sealed key was opened.

### Per case (outcome / team decision)

| Case | Fault (revealed after freeze) | Rules | Single | Isolated 4 | Communicating 4 |
| --- | --- | --- | --- | --- | --- |
| case-01 | clean_identical | TN | TN | TN | TN (INCOMPLETE: Skeptic none) |
| case-02 | prediction_misregistered | FN | FN | FN | FN |
| case-03 | shadow_mask_inverted | TP | TP | TP (Thermal) | TP (Thermal; Skeptic none) |
| case-04 | temperature_misregistered | TP | FN | **TP (Skeptic)** | FN |
| case-05 | temperature_unit_celsius | TP | TP | TP (Thermal) | TP (Thermal + Skeptic) |
| case-06 | clean_float32_roundtrip | TN | TN | TN | TN |

### Aggregates (primary metrics; INCOMPLETE counts as not flagged)

| Approach | TP | FP | FN | TN | Precision | Recall | F1 | FPR | Incomplete cases / verdicts | Model calls | Input / output / total tokens | Tool calls | Elapsed (sum) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Deterministic rules | 3 | 0 | 1 | 2 | 1.00 | 0.75 | 0.857 | 0.00 | 0 / 0 | 0 | 0 | 60 | 0.02 s |
| Single generalist | 2 | 0 | 2 | 2 | 1.00 | 0.50 | 0.667 | 0.00 | 0 / 0 | 39 | 160 234 / 10 352 / 170 586 | 93 | 143.2 s |
| Isolated specialists | 3 | 0 | 1 | 2 | 1.00 | 0.75 | 0.857 | 0.00 | 0 / 0 | 87 | 275 548 / 18 064 / 293 612 | 96 | 79.9 s |
| Communicating specialists | 2 | 0 | 2 | 2 | 1.00 | 0.50 | 0.667 | 0.00 | 1 / 2 | 87 | 333 355 / 18 183 / 351 538 | 87 | 111.4 s |

Secondary (complete cases only): communicating 5 cases, TP 2 FP 0 FN 2 TN 1 (recall 0.50); the
other approaches had no incomplete cases. Total paid usage: **213 model calls, 769 137 input +
46 599 output = 815 736 tokens** (provider-reported, unpriced), within the registered caps
(288 calls; 64 000 input tokens per case and arm — no overshoot) and the registered projection.

### Did communication help? (exploratory)

* **Detection:** no. Communicating specialists caught 2 of 4 faults, isolated specialists 3 of 4.
  The only difference is case-04 (temperature layer shifted against the others): in both four-agent
  runs the Skeptic measured the same weak PSR/non-PSR temperature contrast; working alone it called
  this a fault, while in the communicating run it and Thermal judged it unremarkable. One case is
  not interpretable as an effect.
* **Completeness:** no. Communicating runs left 2 individual verdicts missing (the Skeptic in
  case-01 and case-03: its next call was refused by the pre-registered input-token admission after
  its 12 000-token reserve was used and the common pool was spent; the prompts of a communicating
  agent carry others' events). Isolated runs had none. The v2 reserve fixed v1's tool starvation but
  was too small for the communicating Skeptic.
* **Efficiency:** no. Communicating specialists used the most tokens (58.6k per case vs 48.9k
  isolated, 28.4k single) with the same 87 model calls as isolated, and took longer than isolated.
* **False alarms:** none in any approach (2 clean controls only).
* **What "communication" amounted to:** 1–3 claims per case and up to 8 cross-agent citations;
  **zero test requests, challenges or revisions, and no agent ever changed its verdict.** (Isolated
  runs show cross-agent references only through identical deterministic measurements, which are
  charged and then made visible to the requester — not messages.)
* The single generalist was cheapest (≈ 0.58× the isolated team's tokens) but missed case-04.
  Deterministic rules matched the best LLM approach at zero model cost, missing only the
  prediction misregistration — which every approach missed.

### Caveats

Six synthetic cases from ONE genuine lunar tile, one pass, no repeats: correlated cases, no
confidence intervals, no generalization. Differences of one case between approaches are within
what model sampling alone could produce. The faults are synthetic, not lunar anomalies. The
benchmark task text named example fault types for all LLM approaches. Note on logs: the console
shows asyncio "Event loop is closed" warnings from HTTP clients of earlier cases being garbage-
collected after their case completed; every case's outputs were written before that, and all
outputs re-verify against their frozen hashes.
