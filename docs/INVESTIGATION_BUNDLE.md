# Investigation bundles, offline verification and replay (P6)

Everything here is implemented and tested; numbers come from recorded runs. The science is
screening statistics on a prospectivity index: no bundle, report or ranking detects, confirms or
quantifies lunar ice, and the research-priority ranking is **not** a probability of ice.

## 1. Commands (CLI and a future UI call the same `InvestigationService`)

```bash
python -m iceclaw.cli investigate --region ice-s-0001-0002   # live agents -> exported + verified bundle
python -m iceclaw.cli verify --run RUN_ID_OR_PATH            # offline: no API key, no GPU, no catalog
python -m iceclaw.cli verify --run RUN --source-package PATH # external content-addressed source package
python -m iceclaw.cli verify --run RUN --local-source        # rebuild the package from local data
python -m iceclaw.cli verify --run RUN --write               # refresh verification.json
python -m iceclaw.cli export --run RECORDED_RUN              # bundle a recorded run (no model call)
python -m iceclaw.cli replay --run RUN                       # display saved events (no model call)
```

`investigate` uses the OpenAI key and model ID from the IceClaw runtime settings (local `.env`,
never printed) through Microsoft Agent Framework; only the verified `PAIRED` region is accepted
(observation-only tiles raise `PREDICTION_UNAVAILABLE`). Exit codes: `verify` 0 verified / 1
failed / 2 run not found; `investigate` 0 ok / 1 region refused / 2 live mode unavailable
(e.g. `MISSING_API_KEY`) / 3 bundle failed verification.

## 2. Bundle layout (`runs/<id>/`)

| File | Content |
| --- | --- |
| `input_manifest.json` | the P0 `InputManifest` fields (region, source/checkpoint hashes, provider/model, code revision, visible modalities, budget, tools executed, limitations) + `bundle_format`, `derivation_version`, `mode`, source-package identity, provider-reported model usage per call, prompt policy, run outcome, `integrity` hashes |
| `events.jsonl` | the authentic ordered event log, exported exactly as recorded (test: identical to the SQLite history) |
| `evidence.json` | every deterministic measurement (full `ToolMeasurement` record + per-field units + evidence class), every statement with its epistemic label, cited measurements and numeric check, the cross-agent exchange |
| `report.md` | readable report (sections below) |
| `rankings.csv` | research follow-up priority, see §5 |
| `verification.json` | the verifier's result at export time |
| `llm_calls.jsonl` | the retained model transcript: request/response per call + provider-reported token usage (includes the provider's opaque `protected_data` reasoning blobs) |
| `source/` | the portable source package (§3) |
| `internal/` | live runs only: the full SQLite history (not needed for verification) |

`mode` is `live`, `recorded-export` or `scripted-test`; the report says which, in words.

## 3. Source package and portability

`source/` holds the exact bytes of the **non-evaluator** arrays the tools read: the NASA-IBM
prediction raster (re-encoded losslessly by MOONATLAS) and the SomBench `TMAX`, `LPSR`, `SLOPE`
GeoTIFFs (~0.7 MB), plus grid, projection, footprint, decoding parameters, units, per-file SHA256
and the attribution/modification notice (SomBench CC BY 4.0; model output of the Apache-2.0 NASA-IBM
checkpoint, published by MOONATLAS). Its identity is `package_sha256` (hash of the canonical
manifest), recorded in the input manifest. **Excluded by policy and tested:** the expert target
layer and any reference raster; the upstream `ice-sectors.json` (it carries statistics of the
evaluator-only reference; only its hash is recorded, with a note); fault labels; layers with
unestablished units; `.env`; absolute paths.

The agents run on arrays loaded **from this package** (not from the catalog), so the verifier
recomputes from exactly the data they saw. A test proves package-decoded arrays equal the catalog's
bit-for-bit. Verification modes (printed): *embedded source package (offline portable)*; *external
content-addressed source package* (`--source-package`, must hash to the declared identity); *requires
local source data* (nothing embedded: stated as such, never silently claimed portable).

## 4. What `verify` proves (23 named checks; any failed error check => exit 1)

structure and manifest schema; derivation version; per-file SHA256 integrity; events valid with
unique ids; every parent exists **and precedes** its child; challenge targets another agent's event,
revision revises the same agent's claim with evidence **newer** than that claim, results answer a
request; measurement events linked to records; source-package integrity and identity; declared
hashes; grid / resolution / footprint / unit / source-hash consistency; **recomputation of every
recorded measurement from the source arrays within its declared tolerance**; event text equals the
displayed measurement; `evidence.json`, `rankings.csv` and `report.md` **re-derived** from the
events and measurements and required to equal the saved files; token totals and budgets; no `PRO`
token, key-like string, absolute path, `.env` or sealed material; the disclaimer is present.
Warnings (do not fail): quoted numbers not found in cited evidence; affirmative ice wording (negated
wording is accepted); unpriced cost.

Three attacks were demonstrated on a copy of the final live bundle (clean environment, no key):

| Attack | Result |
| --- | --- |
| flip one byte of the embedded `TMAX` GeoTIFF | `source_package_integrity` FAIL, exit 1 |
| fabricate an event with a non-existent parent and recompute the file hash | `event_parents`, `reference_rules`, `evidence_rederived`, `report_rederived` FAIL, exit 1 |
| edit a recorded measurement value and recompute **every** hash | `recompute_measurements`, `event_text_matches_measurement`, `evidence_rederived`, `report_rederived` FAIL, exit 1 |

**Limits, stated:** this proves the numbers follow from the source data and the bundle is
internally consistent. It cannot prove a language model wrote the agent text (anyone can write a
self-consistent log). Hash anchoring is only as strong as where the manifest is kept.
Derivation is versioned (`iceclaw-derive/1`); a verifier with a different derivation version
refuses the bundle with a clear message instead of reporting spurious tamper failures.

## 5. Report content and labelling

Sections: question; how to read it; available inputs; observed numeric measurements (4a on lunar
observation data, 4b on NASA-IBM model output); model-prediction observations (deterministic one-line
readings, no interpretation); what the AI specialists concluded; cross-agent exchange; alternative
explanations; outstanding uncertainties; research follow-up priority; provenance.
Four labels, never mixed: **NASA-IBM model prediction**, **lunar observation data**, **deterministic
calculation** (on model output or on observations, decided by the source file, not by an agent), and
**AI hypothesis**. Mixed-unit measurement dictionaries are expanded **per field** with the correct
unit (`K`, `deg`, `pixels`, `dimensionless`, `<unit>/m`, `pixel index (relative to window origin)`);
the single-unit record is preserved verbatim in the evidence record. A `test_request`'s JSON is shown
as readable text and its parameters are not number-checked (they are inputs, not claims).

`rankings.csv` orders the pixels and windows that the recorded measurements **named** (raw peak,
gradient-discontinuity candidates, examined windows) by how many distinct agents' evidence covers or
names them. It measures how much independent scrutiny a location received, not physical support.

## 6. Replay

`replay` reads `events.jsonl` (or a run's SQLite history) only, always prints `RECORDED RUN`, builds
no model client (tested by making client construction raise) and reports `0 model calls made`.
The P5 live run is preserved byte-for-byte (its files' hashes are identical before and after P6) and
exports to a verifiable bundle without any model call.

## 7. Token hardening (bounded evidence digest) and measured results

Design (`agents/digest.py`): each turn shows a deterministic digest capped at 5 000 chars (prompt
capped at 9 000): events addressed to the agent first (requests to it, challenges of its events,
results of its requests), then claims/revisions/results, then measurements; every line keeps the
event id, **all parent ids** (to 12 then a count), status and short source-file ids, only the free
text is clipped; omitted events are listed by id range; `get_event` returns any event in full; the
full history stays in SQLite. Also: hard model-call, input-token and total-token ceilings
(`max_input_tokens` default 150 000; see "Token-budget enforcement" below for exactly what is and is
not guaranteed), per-agent token accounting, cost recorded as **`unpriced`** (no verified model rate
exists).

Measured on the real model (`openai/gpt-6-luna`, 4 specialists, limit 28 calls):

| Run | Calls | Input / output / total tokens | Ended | Notes |
| --- | --- | --- | --- | --- |
| P5 (pre-P6, unbounded digest) | 22 | 86 411 / 4 638 / 91 049 | quiescent | Skeptic input 35 255 over 7 calls |
| P6 run A (cap 70 000) | 20 | 72 524 / 3 646 / 76 170 | cap reached | cap set too low: it blocked any further Skeptic turn |
| P6 run B (cap 70 000) | 19 | 70 789 / 3 761 / 74 550 | cap reached | |
| P6 run C (cap 100 000) | 27 | 108 316 / 4 697 / 113 013 | cap reached | richest: Skeptic asked Thermal for a windowed test, Thermal answered |
| **P6 final (cap 150 000)** | **19** | **70 593 / 4 398 / 74 991** | **quiescent** | bundle `inv-20261031T201513Z-p6` |

Honest reading: the digest reduced growth *across* turns (the Skeptic's second turn started at
2 804 input tokens in run B versus 3 794 in P5; Skeptic input 26 129 over 6 calls versus 35 255) and
comparable runs used ~17 % fewer tokens (74-76k vs 91k). Run-to-run variance is larger than that:
the richest run used 113k. The dominant cost is not the digest: every model call re-sends the
instructions and tool schemas (~2.35k tokens on the first call) plus the in-turn conversation, so
cost scales with the number of calls. An instruction to batch independent tool calls changed little
(−2 %). Further levers, **not** implemented: prompt caching, shorter tool schemas, fewer specialist
turns. Defaults were set from these observations (cap 150 000 guards against runaway growth; normal
runs use 71-113k input tokens).

### Token-budget enforcement (P6 correction)

**Finding.** The original check compared only *completed* provider-reported tokens with the cap, so
it ignored (a) the size of the call being admitted and (b) calls already in flight. Re-reading the
recorded per-call timings (read-only):

* Runs A and B (cap 70 000) were **not** concurrency overshoots: the last call started under the cap
  with **0** calls in flight and then added its own input (6 423 and 3 175 tokens).
* Run C (cap 100 000) overshot by 8 316 tokens: **concurrency** (calls 26 and 27 were in flight
  together) plus each call's own size.

**Fix (`SwarmRuntime.llm_before_call` / `llm_release` / `llm_record`).** Each call must reserve before
it is made: it is admitted only if `completed input tokens + in-flight reservations + this call's
estimated input <= max_input_tokens`; otherwise `BudgetExhausted("INPUT_TOKENS")` is raised before
any spend. The estimate is the characters actually sent (messages, instructions, tool schemas) times
a tokens-per-character ratio calibrated from this run's own completed calls (highest observed ratio
x 1.25; initial 0.5). Reservations are released on success, error, cancellation and timeout (tested).
Because the estimate is not a provider tokenizer, the **input cap is enforced against an estimate**;
the manifest records `input_cap_overshoot_tokens` (measured, normally 0) and `peak_inflight_calls`.

**What stays soft.** `max_total_tokens` is a **soft** cap: output length is not capped, so a fixed
1 500-token output allowance is reserved per call and the realised total can still exceed the cap.
`limits.enforcement` in the manifest says so. The call-count cap remains exact. The recorded runs'
numbers above are unchanged (they predate this fix).

Regression tests: `tests/test_token_reservations.py` (legacy own-size and concurrent overshoot
arithmetic; own-size refusal; in-flight blocking and release; concurrent callers stay under the cap;
calibration; soft total cap; end-to-end capped scripted run with overlapping simulated latency).

## 8. Honest findings from the live runs

* Five live runs in total (P5 plus four in P6), one region, one model: **no challenge and no
  revision occurred in any of them.** One run (C) produced a genuine **test request → test result**
  between the Skeptic and Thermal; the others ended with "no contradiction is warranted". The
  challenge→revision path is exercised only by scripted tests (a bundle built from a scripted run
  verifies and reports it).
* In the P6 live runs the agents' quoted numbers matched their cited measurements (0 warnings in
  runs A, B and final); run C's only numeric warning was a test-request false positive (parameters
  are inputs, now excluded). In the P5 export the one warning was a tokenizer false positive on
  `65,536` (fixed).
* **Verifier incident (kept visible):** the first run with the new verifier failed
  `no_forbidden_content` on a "key-like string". Investigation (without reading any credential):
  the match was 472 chars long, inside a 2 444-char `protected_data` field (the provider's opaque
  reasoning data), preceded by base64 characters: `sk-` occurred by chance mid-blob. The detector
  now requires a token boundary and has a regression test. I did **not** compare the transcript
  against the real configured key (that check was blocked as credential access and I did not route
  around it); the code paths that build prompts never include the key, and tests assert secrets never
  reach stored records.
* Three early live bundles were derived before the numeric-check/request-text/derivation-version
  fixes, so the final verifier refuses them; they are preserved (not deleted) under
  `runs/_superseded_p6_dev/` and superseded by `runs/inv-20261031T201513Z-p6`.

## 9. Limitations

* One paired region (`PARTIAL_ONE_REGION`); published prediction, not a fresh inference; one model.
* No cost figure (unpriced); token use varies run to run (71-113k input observed).
* Offline verification cannot prove authorship of LLM text; the transcript is retained for audit.
* The **private** `llm_calls.jsonl` (~0.35 MB) includes provider-opaque reasoning blobs; use the
  public-redacted bundle (section 10) for distribution.
* Derivation changes require a version bump; each bundle is verified under the derivation version
  its manifest declares, and unknown versions are refused, not reinterpreted.
* **Rankings are investigation attention, not validated ice prospectivity.** `rankings.csv` orders
  targets by how many distinct agents and evidence events covered them. It does not measure how
  likely ice is, has not been validated against independent data, and may rank an examined window
  above a measured feature. **Ties are common** (in the final live bundle every target was covered
  by one agent) and are broken **alphabetically by target id**, which carries no importance: there
  the raw maximum ranks 6th only because of its name. `iceclaw-derive/2` makes this explicit in the
  CSV (`note`, `tie_group_size`, `tie_break` columns) and the report (section 10). The preserved
  `/1` bundle keeps its original wording (it is verified under `iceclaw-derive/1`) and is superseded
  for reading purposes by this note.

## 10. Public-redacted bundles and the versioned contract

| | `iceclaw-bundle/1` + `iceclaw-derive/1` | `iceclaw-bundle/2` + `iceclaw-derive/2` |
| --- | --- | --- |
| Rankings | original columns and note | + `tie_group_size`, `tie_break`; attention/ties wording in CSV and report |
| Manifest | no profile fields | `transcript_profile`, `transcript_redactions`, `bundle_dir` |
| Verified by | the same verifier (format and derivation must match each other) | same |

Profiles: `full-private` is the authentic run (complete transcript, `internal/` SQLite history,
provider reasoning payloads untouched). `public-redacted` is for distribution.

```
python -m iceclaw.cli publish --run <bundle> [--out <id>]     # no model call; source only read
```

`publish` refuses an unverified or already-public source. It copies the embedded source package
byte-for-byte and re-derives events, evidence, rankings and the report from the verified bundle; only
the values of `protected_data` / `encrypted_content` in `llm_calls.jsonl` are replaced by
`[redacted: opaque provider reasoning data]`. Messages, tool calls, tool results, token usage and the
per-call table are kept, and `internal/` is not included. The manifest records the counts removed
and the sha256 and size of the private transcript it was made from, so the public bundle is tied to
the private one without containing it. The verifier adds `transcript_profile`,
`public_transcript_redacted` (no opaque keys and no long opaque string except the marker) and
`public_bundle_no_internal_history`; re-inserting a blob (even with rehashed files) fails.

Measured on the preserved live bundle `inv-20261031T201513Z-p6` (left unchanged, still verifies as
`/1`): `publish` produced `inv-20261031T201513Z-p6-public`, 26 opaque fields (125 656 chars)
removed, transcript 352 000 -> 227 436 bytes, same events, same 19 calls and 74 991 tokens,
26 checks passed offline.
