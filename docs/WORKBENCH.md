# IceClaw workbench: scientist-facing Streamlit interface

```bash
cd ~/IceClaw && source .venv/bin/activate
python -m streamlit run app.py          # http://localhost:8501
```

One page, Streamlit + Plotly only. It reuses `InvestigationService`, the catalog, the bundle
exporter/verifier and the agent runtime; it recomputes no science. Code: `app.py` (rendering) and
`src/iceclaw/ui/{views,runner,regions}.py` (Streamlit-free, unit-tested). Tests: `tests/test_ui.py`.

## What the page does

| Section | Behaviour |
| --- | --- |
| Sidebar | Region selector: 1 paired region (`ice-s-0001-0002`) and the 11 observation-only tiles, each labelled **UNAVAILABLE: observations only, no prediction**. Selecting one shows the catalog's reason and no controls. No prediction is ever invented. |
| 1 Region and data layers | Plotly heatmaps of the real prediction and TMAX, LPSR, SLOPE in lunar south-polar stereographic kilometres (equal axes, lat/lon on hover, no Earth basemap). Each layer shows its class (MODEL OUTPUT vs OBSERVATION-DERIVED), unit, source file and sha256. Optional overlay of investigation-attention targets (marked "not an ice location"). Arrays come from the saved bundle's hash-verified embedded source package, or from the local catalog. |
| 2 Investigate | **Recorded investigation** (default: `inv-20261031T201513Z-p6-public`, labelled `RECORDED INVESTIGATION`, no model calls) and **Live investigation** (explicit, paid; see below). |
| 3 Specialists | Thermal, Geology, Prediction Auditor, Skeptic cards (role, events, model calls, tokens) and the real event timeline. Cross-agent follow-up is highlighted only if events of kind test_request / test_result / challenge / revision exist; otherwise the page says none occurred. |
| 4 Findings | Headline recorded measurements, full measurement table (units from the recorded tool output), AI hypotheses (labelled, with cited evidence, alternatives, limitations), open questions and the agents' own proposed follow-up, the attention ranking (separate tab, with tie groups), sources, attribution and limitations. |
| 5 Research follow-up screening | Deterministic, post-processing screen of the bundle's source rasters: follow-up windows and model–observation review windows outlined on the prediction heatmap, evidence tables, criteria, sensitivity, provenance, optional `triage.json`/CSV downloads, and an optional, explicitly confirmed targeted review by the four specialists. Separate from the attention ranking. See [TRIAGE_REVIEW.md](TRIAGE_REVIEW.md). |
| 6 Verify and export | **Verify findings** runs the existing offline verifier (no key, no network) and lists every check with its real outcome; failure disables the download. **Download** gives the public-redacted `iceclaw-bundle/2` zip only. |

## Live investigations: safety properties

* Started only by the button callback, after ticking the cost/privacy confirmation. Plain reruns,
  widget changes and page refreshes never start a run (tested: reruns leave `starts_attempted` at 0/1).
* `LiveRunManager.start` is atomic and single-flight per runs folder (in-process lock plus a pid
  marker file for a second server); further starts are refused (`ALREADY_RUNNING`) until it ends.
* Configuration is checked first (`MISSING_API_KEY`, `MISSING_LLM_MODEL`, unsupported provider); a
  missing key blocks the run before any directory or client exists. Only presence flags and the
  model name are shown; error text is scrubbed for key-like strings.
* The run is `InvestigationService.investigate`: four Microsoft Agent Framework specialists using
  the OpenAI model from the user's environment (`gpt-6-luna` in the recorded run).
* Progress is read read-only from the run's own SQLite history: only real events appear.
* On success the verified private bundle is published to a public-redacted bundle (no model call)
  and that is what the page loads; the private bundle stays on disk and is never offered.
* The page loads settings with the same `.env` loader as the CLI and never prints values.

## Evidence labelling (always visible)

The prediction is an **ice-prospectivity index** (raw regression output, not clipped, not a
probability, not detected ice). Calculations are labelled deterministic and tied to evidence ids;
agent text is labelled an AI hypothesis. The attention ranking is shown separately with the
warning that it is an investigation-attention order with alphabetical ties, not physical ice
prospectivity. Agent text is rendered as escaped plain text, never as HTML or markdown links.

## Degradation

Missing local catalog: only saved investigations are listed (live disabled, with a notice).
Missing/corrupt bundle files: a specific message. Missing source rasters: an error with the data
location. Missing API configuration: live disabled with the reason; recorded replay unaffected.
Failed verification: red banner with failing checks, download disabled.

## Visual inspection (headless Chromium via Playwright, real data, no model calls)

`docs/screenshots/`: `p8_recorded_page.png` (whole page), `p8_prediction_overlay.png`,
`p8_layer_tmax.png`, `p8_verify_findings.png` (26 checks PASS), `p8_unavailable_tile.png`.
Emoji glyphs appear as empty boxes only because the headless browser has no emoji font; every
status also carries a text label (PASS/FAIL, RECORDED, UNAVAILABLE). The live-running panel was
exercised with the scripted provider in tests, not screenshotted, because that would need a paid run.
