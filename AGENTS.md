# IceClaw repository instructions

Read these two documents before changing code:

1. [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) — authoritative research goal, science rules, data contracts and acceptance requirements.
2. [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — authoritative implementation sequence, file ownership, dependency boundaries and test gates.

Work within **one numbered implementation phase at a time**. Verify that its prerequisites passed. At the end, report actual commands/test results, files changed, observed issues, data provenance and readiness for the next phase. Never mark an unexecuted test as passed.

Engineering invariants:

- Python 3.11/3.12; one IceClaw runtime, Microsoft Agent Framework for scientific agents, Streamlit UI, SQLite evidence store.
- NASA–IBM checkpoints and MOONATLAS inference run in a separately pinned environment; IceClaw reads validated data artifacts.
- CLI and UI call the same service functions. Deterministic calculations happen in scientific tools, not in an LLM response.
- Never mislabel a prospectivity index as detected lunar ice or an independently calibrated probability.
- Keep the evaluator's reference targets and injected-fault truth out of all agent-visible inputs, tools and prompts.
- Read source numeric rasters, not color-mapped previews. Record CRS, units, no-data masks, source revisions and file checksums.
- Do not open, print, modify, delete, copy into prompts or commit the user's `.env`, API keys, model weights or raw scientific datasets. A `.env.example` may contain **only empty placeholders**.
- Tests use fake model providers unless an integration run is explicitly requested. Never fabricate scientific findings, agent messages, benchmark results or tests.
- Avoid unrelated features, premature UI work and changing canonical contracts without matching documentation and regression tests.

Existing planning docs `docs/SCIENTIFIC_PROTOCOL.md` and `docs/DEMO_2MIN.md` are supplementary and may reflect older assumptions.

**Current implementation status:** P0–P4 committed; P5 reported complete locally but UNCOMMITTED after 215 passing tests and one genuine live four-specialist run. **Next: review/commit P5, then implement P6** (source-portable verification bundle and bounded event digest). After P6, prioritize the minimal working P8 Streamlit demonstration; P7 then measures carefully budgeted benchmarks on eligible paired scenarios. The source of truth is implementation plan v3.1. Do not redo committed phases or bypass the evaluation firewall.
