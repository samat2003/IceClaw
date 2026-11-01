"""IceClaw scientist workbench (P8): one Streamlit page.

    python -m streamlit run app.py

The page reuses the existing services and contracts and recomputes nothing:

* layers come from the hash-verified source package of a saved bundle (or the local catalog),
* investigations are the real ``InvestigationService`` (Microsoft Agent Framework specialists,
  OpenAI model from the user's environment),
* findings are the recorded measurements and agent statements,
* "Verify findings" calls the existing offline verifier,
* downloads are public-redacted ``iceclaw-bundle/2`` packages only.

Recorded replay never contacts a model. A live run starts only from an explicit button click.
"""

from __future__ import annotations

import html
import re
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from iceclaw.config import Settings
from iceclaw.services import DEFAULT_BUDGET, InvestigationService
from iceclaw.ui import regions as regions_mod
from iceclaw.ui import runner, views

DEMO_BUNDLE_ID = "inv-20261031T201513Z-p6-public"
POLL_SECONDS = 2.0

st.set_page_config(
    page_title="IceClaw · lunar ice-prospectivity audit workbench",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.ic-banner {border-left: 6px solid #8A4B00; background:#FFF4E0; color:#3B2300; padding:.7rem 1rem;
  border-radius:4px; margin:.25rem 0 1rem 0; font-size:.95rem;}
.ic-recorded {border-left: 6px solid #0B5FA5; background:#E8F1FA; color:#0A2E4D; padding:.6rem 1rem;
  border-radius:4px; margin:.25rem 0 .75rem 0;}
.ic-live {border-left: 6px solid #1B7A3A; background:#E7F5EC; color:#0F3B1D; padding:.6rem 1rem;
  border-radius:4px; margin:.25rem 0 .75rem 0;}
.ic-chip {display:inline-block; padding:1px 8px; border-radius:12px; border:2px solid;
  font-size:.82rem; font-weight:600; background:#FFFFFF; color:#1A1A1A; white-space:nowrap;}
.ic-kind {display:inline-block; padding:1px 7px; border-radius:4px; background:#EEF0F3;
  color:#1A1A1A; font-size:.78rem; font-family:monospace;}
.ic-follow {background:#FFF8E1; border:1px solid #E0B000;}
</style>
"""


# ------------------------------------------------------------------------------ helpers
def plain(text: str) -> str:
    """Agent text is untrusted model output: show it as text, never as markdown/HTML."""
    return re.sub(r"([\\`*_\[\]<>#|~$])", r"\\\1", str(text))


def chip(agent_id: str) -> str:
    color, sym = views.AGENT_STYLE.get(agent_id, ("#595959", "•"))
    return (
        f'<span class="ic-chip" style="border-color:{color}">{sym} '
        f"{html.escape(views.agent_title(agent_id))}</span>"
    )


def kind_badge(kind: str) -> str:
    return f'<span class="ic-kind">{html.escape(kind)}</span>'


def fmt_time(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso or "?"


@st.cache_resource
def get_settings() -> Settings:
    return runner.app_settings()


def roots_key(s: Settings) -> str:
    return "|".join([str(p) for p in s.moonatlas_roots] + [str(s.sombench_root), str(s.data_root)])


@st.cache_data(show_spinner="Reading the local region catalog…")
def cached_catalog(_settings: Settings, key: str) -> regions_mod.CatalogView:
    return regions_mod.load_catalog_view(_settings)


@st.cache_data(show_spinner="Georeferencing source layers…")
def layers_from_bundle(path: str, stamp: str):
    arrays = views.source_arrays(views.load_bundle_view(path))
    return views.build_layers(arrays)


@st.cache_data(show_spinner="Loading verified source layers from the local catalog…")
def layers_from_catalog(_settings: Settings, key: str, region_id: str):
    return views.build_layers(regions_mod.load_catalog_arrays(_settings, region_id))


@st.cache_data(show_spinner="Screening windows from the bundle's source package…")
def cached_triage(path: str, stamp: str) -> dict:
    from iceclaw.reports import triage_artifact as ta

    view = views.load_bundle_view(path)
    return ta.build_artifact(view.source_dir, view.run_id)


@st.cache_data(show_spinner=False)
def cached_zip(path: str, stamp: str) -> bytes:
    return views.public_bundle_zip(path)


def stamp_of(p: Path) -> str:
    f = p / "input_manifest.json"
    try:
        s = f.stat()
        return f"{s.st_mtime_ns}:{s.st_size}"
    except OSError:
        return "missing"


# -------------------------------------------------------------------------------- header
def header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("🧊 IceClaw — lunar ice-prospectivity audit workbench")
    st.markdown(
        '<div class="ic-banner"><strong>Research prioritisation only.</strong> IceClaw audits a '
        "model's <em>ice-prospectivity index</em> against lunar observation layers. It does not "
        "detect, confirm or discover water ice, and the index is not a probability of ice. Every "
        "number below is traceable to a deterministic tool run that the offline verifier "
        "recomputes.</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------ 1. region + layers
def pick_region(cat: regions_mod.CatalogView, bundles: list[views.BundleSummary]):
    rows = list(cat.rows)
    if not rows:  # no local catalog: only saved investigations are known
        for rid in sorted({b.region_id for b in bundles}):
            rows.append(
                regions_mod.RegionRow(
                    rid,
                    "SAVED_ONLY",
                    False,
                    "Saved investigation only: the local catalog is not available here.",
                )
            )
    if not rows:
        return None, rows
    ids = [r.region_id for r in rows]
    if st.session_state.get("region") not in ids:
        paired = [r.region_id for r in rows if r.investigable]
        saved = [r.region_id for r in rows if any(b.region_id == r.region_id for b in bundles)]
        st.session_state["region"] = (paired or saved or ids)[0]
    by_id = {r.region_id: r for r in rows}
    st.sidebar.selectbox(
        "Lunar region",
        ids,
        key="region",
        format_func=lambda i: (
            by_id[i].label
            if by_id[i].investigable
            else f"{i}  ·  UNAVAILABLE: "
            + (
                "observations only, no prediction"
                if by_id[i].status == regions_mod.OBSERVATION_ONLY
                else "saved investigation only"
            )
        ),
        help="Only regions with a verified NASA–IBM prediction paired with TMAX, LPSR and SLOPE "
        "can be investigated. Observation-only tiles have no prediction; none is invented.",
    )
    return by_id[st.session_state["region"]], rows


def unavailable_panel(row: regions_mod.RegionRow, n_obs_only: int) -> None:
    st.subheader("1 · Region")
    st.error(
        f"**{row.region_id}: unavailable for prediction investigations.** "
        + (
            "This tile has verified lunar observations but **no NASA–IBM prediction**, so there is "
            "nothing for the agents to audit. IceClaw does not create predictions."
            if row.status == regions_mod.OBSERVATION_ONLY
            else row.reason
        ),
        icon="🚫",
    )
    c = st.columns(3)
    c[0].metric("Status", row.status.replace("_", " ").title())
    c[1].metric("Original tile", row.tile or "?")
    c[2].metric("Grid", row.grid or "?")
    if row.reason:
        st.caption(f"Catalog reason: {row.reason}")
    st.caption(
        f"{n_obs_only} observation-only tile(s) are listed in the region selector, all unavailable. "
        "Choose the paired region to investigate."
    )


def region_section(row, view: views.BundleView | None, settings: Settings, cat) -> None:
    st.subheader("1 · Region and data layers")
    st.markdown(views.PROSPECTIVITY_NOTICE)
    layers = geom = None
    source_label = ""
    try:
        if view is not None and view.region_id == row.region_id:
            geom, layers = layers_from_bundle(str(view.path), stamp_of(view.path))
            source_label = (
                f"embedded, hash-verified source package of `{view.run_id}` "
                f"(content address `{view.manifest['source_package']['package_sha256'][:16]}…`)"
            )
        elif row.investigable:
            geom, layers = layers_from_catalog(settings, roots_key(settings), row.region_id)
            source_label = "local verified catalog (checksums re-verified when loaded)"
    except Exception as e:  # noqa: BLE001
        st.error(
            f"Source layers could not be loaded ({type(e).__name__}: {runner.scrub(str(e))}). "
            "Check the local data files or the saved bundle's `source/` folder.",
            icon="📁",
        )
    if not layers:
        st.info("No source layers available for this region in this environment.")
        return
    a, b = st.columns([3, 1.4], gap="large")
    with b:
        sel = st.radio(
            "Layer",
            [m for m in views.LAYER_ORDER if m in layers],
            format_func=lambda m: views.LAYER_STYLES[m].label,
            key="layer",
            help="Select which co-registered raster to display.",
        )
        lay = layers[sel]
        st.markdown(f"**{lay.style.kind_badge}** · unit: {lay.style.unit}  \n{lay.style.note}")
        overlay = False
        if view is not None and view.rankings:
            overlay = st.checkbox(
                "Overlay investigation-attention targets",
                key="overlay",
                help="Marks locations the agents' measurements named. This is where scrutiny "
                "landed, not where ice is.",
            )
            if overlay:
                st.caption("White × = attention targets only. Not ice locations.")
        st.caption(f"Source: {source_label}. File `{lay.file_id}` · sha256 `{lay.sha256[:16]}…`")
        st.caption(
            "Footprint (lat, lon) NW/NE/SE/SW: "
            + "; ".join(f"({p[0]:.2f}, {p[1]:.2f})" for p in geom.footprint)
            + f". Pixel {geom.resolution_m:g} m, {geom.shape[1]}×{geom.shape[0]} px."
        )
    with a:
        targets = views.rank_targets(geom, view.rankings) if overlay and view else None
        fig = views.make_heatmap_figure(geom, lay, targets=targets)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.caption(
            "Lunar south-polar stereographic coordinates (kilometres from the pole), equal axis "
            "scale, so shapes are not distorted; hover shows latitude/longitude and the value. "
            "There is no Earth basemap."
        )
    with st.expander("Text alternative: what the recorded tools say about this layer"):
        if view is None:
            st.write("Load an investigation to see recorded statistics for this layer.")
        else:
            rows = [
                r
                for r in views.measurement_rows(view.evidence)
                if lay.style.modality in view_layers(view, r["event"])
                and r["tool"] in ("quantiles", "missingness")
            ]
            if rows:
                st.dataframe(
                    [
                        {k: v for k, v in r.items() if k in ("event", "quantity", "value", "unit")}
                        for r in rows
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.write("No quantile table was recorded for this layer in this investigation.")


def view_layers(view: views.BundleView, event_id: str) -> list[str]:
    for m in view.evidence.get("measurements", []):
        if m["event_id"] == event_id:
            return list(m.get("layers", []))
    return []


# ------------------------------------------------------------------------- 2. investigate
def choose_bundle(row, bundles: list[views.BundleSummary], mgr):
    """Decide which saved public bundle is active BEFORE any section renders.

    Default: the saved demo investigation when present, else the newest. A live run finished by
    this app is adopted once. Returns (summaries for this region, loaded view or None, error).
    """
    mine = [b for b in bundles if b.region_id == row.region_id]
    if not mine:
        return mine, None, ""
    ids = [b.run_id for b in mine]
    snap = mgr.snapshot()
    if snap.state == runner.SUCCEEDED and snap.public_run_id in ids:
        if st.session_state.get("adopted") != snap.run_id:
            st.session_state["adopted"] = snap.run_id
            st.session_state["bundle"] = snap.public_run_id
    if st.session_state.get("bundle") not in ids:
        st.session_state["bundle"] = DEMO_BUNDLE_ID if DEMO_BUNDLE_ID in ids else ids[0]
    by_id = {b.run_id: b for b in mine}
    try:
        return mine, views.load_bundle_view(by_id[st.session_state["bundle"]].path), ""
    except views.BundleLoadError as e:
        return mine, None, str(e)


def recorded_card(mine: list[views.BundleSummary], view, error: str) -> None:
    st.markdown("#### Recorded investigation")
    if not mine:
        st.info(
            "No saved public investigation for this region. A live investigation can create one "
            "(explicit action, right)."
        )
        return
    by_id = {b.run_id: b for b in mine}
    st.selectbox(
        "Saved public investigation",
        list(by_id),
        key="bundle",
        format_func=lambda i: f"{i}  ·  {by_id[i].model}  ·  {fmt_time(by_id[i].created_utc)}",
        help="Public-redacted bundles found in the runs folder. Replay makes no model calls.",
    )
    if error:
        st.error(f"This saved investigation cannot be shown: {error}", icon="📁")
    elif view is not None:
        st.caption(
            f"Loaded **{view.run_id}**: {len(view.timeline)} events, "
            f"{view.usage.get('calls', '?')} recorded model calls. Replay is read-only and offline."
        )


def banner_for(view: views.BundleView, mgr) -> None:
    u = view.usage
    snap = mgr.snapshot()
    when = fmt_time(str(view.manifest.get("created_utc", "")))
    stats = (
        f"{u.get('calls', '?')} model calls · {u.get('total_tokens', 0):,} tokens (provider-reported) "
        f"· ended `{view.manifest.get('run', {}).get('ended_reason', '?')}`"
    )
    if snap.state == runner.SUCCEEDED and snap.public_run_id == view.run_id:
        st.markdown(
            f'<div class="ic-live"><strong>LIVE INVESTIGATION completed in this app session</strong> '
            f"({html.escape(str(view.manifest.get('llm_model')))} via Microsoft Agent Framework, {when}). "
            f"{html.escape(stats)}. Showing its public-redacted bundle.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="ic-recorded"><strong>{views.RECORDED_LABEL}</strong> — not live. '
            f"Replayed from saved events of an earlier run "
            f"({html.escape(str(view.manifest.get('llm_provider')))}/"
            f"{html.escape(str(view.manifest.get('llm_model')))}, {when}). "
            f"{html.escape(stats)}. <strong>Viewing it makes no model calls.</strong></div>",
            unsafe_allow_html=True,
        )


def live_card(row, mgr: runner.LiveRunManager, status: dict) -> None:
    snap = mgr.snapshot()
    running = snap.state == runner.RUNNING
    st.markdown("#### Live investigation")
    st.caption(
        f"Four Microsoft Agent Framework specialists (Thermal, Geology, Prediction Auditor, Skeptic) "
        f"using `{status.get('model') or 'no model configured'}` via {status.get('provider')}."
    )
    if not status["ok"]:
        st.warning(
            f"Live investigation is not available: **{status['code']}**. {status['message']} "
            "Recorded investigations still work.",
            icon="🔑",
        )
    b = DEFAULT_BUDGET
    st.caption(
        f"Budget per run: up to {b.max_llm_calls} model calls, {b.max_tool_calls} tool calls, "
        f"{b.max_events} events, {int(b.timeout_s)} s. The saved demo used 19 calls and 74,991 "
        "tokens. Cost is not priced here."
    )
    can = bool(status["ok"] and row.investigable and not running)
    confirm = st.checkbox(
        "I understand this makes paid OpenAI requests and may take about a minute.",
        key="confirm_live",
        disabled=not can,
    )

    def _go() -> None:
        # Runs once per click, before the rerun. A plain rerun/refresh never reaches here.
        res = mgr.start(row.region_id)
        st.session_state["start_result"] = (res.started, res.code, res.message)
        st.session_state["confirm_live"] = False

    st.button(
        "🧪 Investigate with 4 live agents",
        type="primary",
        width="stretch",
        disabled=not (can and confirm),
        on_click=_go,
        help="Starts one real investigation. Disabled until you confirm; refused while one runs.",
    )
    res = st.session_state.pop("start_result", None)
    if res and not res[0]:
        st.error(f"Live run not started: **{res[1]}** {runner.scrub(res[2])}", icon="⛔")
    if not row.investigable:
        st.caption("Unavailable: this region has no verified prediction paired with observations.")


def progress_panel(mgr: runner.LiveRunManager) -> None:
    """Real events from the running investigation; polls only while a run is in flight."""
    snap0 = mgr.snapshot()
    every = POLL_SECONDS if snap0.state == runner.RUNNING else None

    @st.fragment(run_every=every)
    def _panel() -> None:
        snap = mgr.snapshot()
        if snap.state == runner.IDLE:
            return
        prog = mgr.progress(snap.run_id)
        what = "Targeted review" if snap.kind == "review" else "Live investigation"
        if snap.state == runner.RUNNING:
            st.info(
                f"**{what} `{snap.run_id}` running.** {len(prog.events)} evidence events, "
                f"{prog.model_calls} model calls, {prog.total_tokens:,} tokens so far. Showing real "
                "events as they are written; nothing is simulated. Refreshing the page does not "
                "start another run.",
                icon="⏳",
            )
            render_timeline(prog.events, empty="Waiting for the first agent event…")
            return
        if snap0.state == runner.RUNNING:  # finished since this fragment was set up
            st.rerun(scope="app")
        if snap.state == runner.FAILED:
            st.error(
                f"**{what} `{snap.run_id}` did not complete cleanly:** "
                f"{snap.error_code}. {snap.error_message} "
                "Any partial private history stays on disk and is never offered for download.",
                icon="⛔",
            )
            if st.button("Dismiss", key="dismiss_job"):
                mgr.reset()
                st.rerun(scope="app")
        else:
            if snap.kind == "review":
                st.success(
                    f"Targeted review `{snap.run_id}` finished: **{snap.ended_reason}**. Its "
                    "results are listed under Research follow-up screening below.",
                    icon="✅",
                )
            else:
                st.success(
                    f"Live investigation `{snap.run_id}` finished ({snap.ended_reason}); the "
                    f"public bundle `{snap.public_run_id}` is loaded below.",
                    icon="✅",
                )

    _panel()


# ---------------------------------------------------------------------- 3. agents + timeline
def render_timeline(rows: list[views.TimelineRow], *, empty: str = "No events.") -> None:
    if not rows:
        st.caption(empty)
        return
    for r in rows:
        with st.container(border=r.followup):
            st.markdown(
                f"`+{r.t_offset_s:6.2f}s` `{r.event_id}` {chip(r.agent_id)} {kind_badge(r.kind)}"
                + (f" · tool `{r.tool_id}`" if r.tool_id else "")
                + (f" · parents {', '.join(f'`{p}`' for p in r.parents)}" if r.parents else ""),
                unsafe_allow_html=True,
            )
            text = r.text
            if len(text) > 320:
                st.markdown(plain(text[:320]) + "…")
                with st.expander("Full text"):
                    st.markdown(plain(text))
            else:
                st.markdown(plain(text))


def agents_section(view: views.BundleView) -> None:
    st.subheader("3 · What the four specialists did")
    by_agent = view.usage.get("by_agent", {})
    counts: dict[str, int] = {}
    for r in view.timeline:
        counts[r.agent_id] = counts.get(r.agent_id, 0) + 1
    cols = st.columns(4)
    for col, aid in zip(cols, views.AGENT_ORDER, strict=True):
        with col, st.container(border=True):
            st.markdown(chip(aid), unsafe_allow_html=True)
            st.caption(views.agent_remit(aid))
            u = by_agent.get(aid, {})
            st.markdown(
                f"**{counts.get(aid, 0)}** events · **{u.get('calls', 0)}** model calls  \n"
                f"{u.get('total', 0):,} tokens"
            )
    fu = views.followup_events(view)
    st.markdown("##### Cross-agent follow-up")
    if fu:
        st.success(
            f"{len(fu)} real cross-agent event(s) occurred (test requests, results, challenges or "
            "revisions). They are highlighted in the timeline below.",
            icon="🔁",
        )
    else:
        st.info(
            "No challenge, revision or test request occurred in this investigation, so none is "
            "shown. The Skeptic reviewed the others' evidence and recorded no warranted "
            "contradiction.",
            icon="ℹ️",
        )
    with st.expander(
        f"Event timeline ({len(view.timeline)} events, real order)", expanded=bool(fu)
    ):
        render_timeline(view.timeline)


# ---------------------------------------------------------------------------- 4. findings
def findings_section(view: views.BundleView) -> None:
    st.subheader("4 · Findings for review")
    st.markdown(views.PROSPECTIVITY_NOTICE)
    heads = views.headline_metrics(view.evidence)
    if heads:
        cols = st.columns(min(4, len(heads)))
        for i, h in enumerate(heads):
            with cols[i % len(cols)]:
                st.metric(
                    h["label"],
                    f"{views.format_value(h['value'], h['unit'])} {'' if h['unit'] == 'fraction' else h['unit']}".strip(),
                    help=f"{views.evidence_basis(h['category'])}. Recorded in {h['event']}.",
                )
    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Measurements",
            "AI hypotheses",
            "Open questions and follow-up",
            "Investigation attention",
            "Sources and limitations",
        ]
    )
    with t1:
        rows = views.measurement_rows(view.evidence)
        st.caption(
            "Deterministic tool outputs, each independently recomputed by the verifier. "
            "Thresholds such as 110 K or 15° are analyst-chosen screening parameters, not "
            "physical constants."
        )
        cats = [c for c in views.CATEGORY_ORDER if any(r["category"] == c for r in rows)]
        pick = st.multiselect("Domain", cats, default=cats, key="meas_cats")
        shown = [r for r in rows if r["category"] in pick]
        for r in shown:
            r["value"] = float(r["value"])
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            column_config={
                "value": st.column_config.NumberColumn("value", format="%.6g"),
                "event": "evidence event",
            },
        )
    with t2:
        st.caption(
            "Language-model interpretations, labelled as hypotheses. Every number in them was "
            "checked against the cited measurements. The offline verifier cannot prove that a "
            "model wrote the text."
        )
        for aid, stmts in views.statements_by_agent(view.evidence).items():
            with st.expander(
                f"{views.agent_title(aid)} · {len(stmts)} statement(s)",
                expanded=aid != "supervisor",
            ):
                for s in stmts:
                    st.markdown(
                        f"`{s['event_id']}` {kind_badge(s['kind'])} status **{s['status']}** · cites "
                        + (
                            ", ".join(f"`{e}`" for e in s["cited_evidence_event_ids"])
                            or "no measurement"
                        ),
                        unsafe_allow_html=True,
                    )
                    st.markdown(plain(s["text"]))
                    if s.get("alternative_explanations"):
                        st.markdown("*Alternative explanations stated:*")
                        for a in s["alternative_explanations"]:
                            st.markdown(f"- {plain(a)}")
                    if s.get("limitations"):
                        st.markdown("*Limitations stated:*")
                        for a in s["limitations"]:
                            st.markdown(f"- {plain(a)}")
                    st.divider()
    with t3:
        st.markdown(
            "These are **research questions and follow-up analyses**, taken from what the agents "
            "recorded. None is a finding that ice exists."
        )
        steps = views.next_steps(view.evidence)
        st.markdown("##### Suggested follow-up (the agents' own closing statements)")
        for s in steps:
            st.markdown(
                f"{chip(s['agent_id'])} `{s['event']}` ({s['status']})", unsafe_allow_html=True
            )
            st.markdown(plain(s["text"]))
        qs = views.open_questions(view.evidence)
        st.markdown("##### Open questions (limitations the agents recorded)")
        for q in qs:
            st.markdown(f"- **{q['agent']}** `{q['event']}`: {plain(q['text'])}")
        if not qs:
            st.caption("No open questions were recorded.")
    with t4:
        st.warning(views.ATTENTION_NOTICE, icon="⚠️")
        rows = [
            {
                "rank": int(r["rank"]),
                "target": r["target_id"],
                "kind": r["kind"],
                "lat": float(r["latitude"]) if r["latitude"] else None,
                "lon": float(r["longitude"]) if r["longitude"] else None,
                "agents covering": int(r["n_distinct_agents"]),
                "evidence": r["evidence_event_ids"],
                "tied with": int(r.get("tie_group_size") or 1),
            }
            for r in view.rankings
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "Columns follow `rankings.csv` in the bundle. A tie group of N means N targets have "
            "identical coverage; their rank order is alphabetical and carries no importance."
        )
    with t5:
        st.markdown("##### Source files")
        src = view.manifest["source_package"]["files"]
        st.dataframe(
            [
                {
                    "file": fid,
                    "layer": v["modality"],
                    "kind": v["kind"],
                    "unit": v["unit"],
                    "sha256": v["sha256"][:16] + "…",
                }
                for fid, v in src.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.markdown("##### Attribution")
        for a in view.manifest.get("attribution", []):
            st.markdown(f"- {plain(a)}")
        st.markdown("##### Scientific limitations")
        for a in view.manifest.get("limitations", []):
            st.markdown(f"- {plain(a)}")
        with st.expander("Full generated report (report.md)"):
            st.markdown(view.report_md)


# ------------------------------------------------------ 5. research follow-up screening
def render_review(rec: dict) -> None:
    """One saved targeted review: everything shown comes from its recorded event log."""
    m, t = rec["metrics"], rec["target"]
    badge = "COMPLETE" if rec["status"] == "COMPLETE" else "INCOMPLETE"
    box = st.success if badge == "COMPLETE" else st.warning
    box(
        f"**{badge}** review of {t['label']} ({t['category']} window) by the "
        f"{rec['arm'].replace('_', ' ')} · agreement among filed reviews: **{rec['agreement']}** · "
        f"{m['n_reviews_filed']}/{m['n_agents']} reviews filed.",
        icon="🧭",
    )
    st.caption(f"Review `{rec['review_id']}` · {rec['label']}")
    for reason in rec["incomplete_reasons"]:
        st.error(f"INCOMPLETE: {reason}. No verdict was substituted.", icon="⚠️")
    st.dataframe(
        [
            {
                "specialist": views.agent_title(a),
                "finding (AI)": r["finding"],
                "cites evidence from": ", ".join(
                    views.agent_title(x) for x in r["cited_suppliers"]
                ),
                "rationale": r["rationale"],
            }
            for a, r in rec["reviews"].items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Independent measurements considered (who supplied what)**")
    st.dataframe(
        [
            {
                "event": x["event_id"],
                "supplied by": views.agent_title(x["supplier"]),
                "tool": x["tool"],
                "layers": ", ".join(x["layers"]),
                "on target window": "yes" if x["window_scoped"] else "no",
                "recomputation": "yes" if x["is_recomputation"] else "",
                "answers request": x["answers_request"] or "",
            }
            for x in rec["evidence"]
        ],
        hide_index=True,
        width="stretch",
    )
    nv = rec["numerical_verification"]
    st.caption(
        f"Independent recomputation: {nv['reproduced']}/{nv['checked']} recorded measurements "
        f"reproduced · requests {m['n_test_requests']} (answered {m['n_test_requests_answered']}) · "
        f"challenges {m['n_challenges']} · revisions {m['n_revisions']} · cross-agent citations "
        f"{m['n_cross_agent_citations']} · {m['usage']['llm_calls']} model calls, "
        f"{m['usage']['total_tokens']:,} tokens, {m['usage']['tool_calls']} tool calls, "
        f"{m['usage']['elapsed_s']} s. Events are in this review's events.jsonl."
    )
    if not (rec["requests"] or rec["challenges"] or rec["revisions"]):
        st.info("No test request, challenge or revision occurred in this review.", icon="ℹ️")
    for title, key in (
        ("Missing evidence", "missing_evidence"),
        ("Competing explanations (as stated by the specialists)", "competing_explanations"),
        ("Suggested follow-up work", "suggested_follow_up"),
    ):
        if rec[key]:
            st.markdown(f"**{title}**")
            for x in rec[key]:
                st.markdown(f"- {views.agent_title(x['agent'])}: {plain(x['text'])}")
    st.caption(rec["label"])


def triage_section(view: views.BundleView, settings: Settings, mgr, status: dict) -> None:
    from iceclaw.reports import triage_artifact as ta
    from iceclaw.reports.package import PackageError
    from iceclaw.science.triage import TriageError

    st.subheader("5 · Research follow-up screening")
    st.warning(views.TRIAGE_LABEL, icon="🧮")
    st.caption(
        "Separate from the investigation-attention ranking in section 4, which counts agent "
        "scrutiny. This screen is computed from the pixels themselves."
    )
    try:
        art = cached_triage(str(view.path), stamp_of(view.path))
        geom, layers = layers_from_bundle(str(view.path), stamp_of(view.path))
    except (TriageError, PackageError, views.BundleLoadError, KeyError, OSError) as e:
        st.error(
            f"Screening is unavailable for this bundle ({type(e).__name__}: "
            f"{runner.scrub(str(e))}).",
            icon="📁",
        )
        return
    t = art["triage"]
    c = t["counts"]
    cols = st.columns(5)
    cols[0].metric("Windows screened", c["n_windows"], help="16×16 native pixels, no resampling")
    cols[1].metric("Follow-up windows", c["n_followup_windows"])
    cols[2].metric(
        "Pixels meeting all conditions",
        f"{c['n_joint_pixels']:,}",
        help=f"of {c['n_valid_all_layer_pixels']:,} valid pixels",
    )
    cols[3].metric("Review windows", c["n_review_windows"])
    cols[4].metric("Index threshold (P90)", f"{t['index_threshold_raw']:.4f}")
    st.markdown(views.TRIAGE_CRITERIA)
    st.caption(
        "Thresholds are analyst-chosen screening choices fixed before the candidate list was "
        "inspected, not physical ice-stability limits or calibrated probabilities. Prediction–input "
        "agreement is not independent confirmation of ice: the model was trained on related inputs."
    )
    for msg in t["messages"]:
        st.info(msg, icon="ℹ️")
    st.plotly_chart(
        views.triage_overlay_figure(geom, layers["nasa_ibm_prediction_raw"], t),
        width="stretch",
        config={"displaylogo": False},
    )
    cum = c["pixels_cumulative"]
    st.caption(
        f"Pixels surviving each condition in turn: high index {cum['high_index']:,} → also shadowed "
        f"{cum['high_and_shadow']:,} → also TMAX ≤ 110 K {cum['high_shadow_cold']:,} → also slope "
        f"≤ 15° {cum['all_four']:,}. Same projected-kilometre axes as section 1."
    )
    t1, t2, t3, t4 = st.tabs(
        ["Follow-up windows", "Review windows", "Threshold sensitivity", "Method and provenance"]
    )
    with t1:
        rows = views.triage_window_rows(t, "followup")
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("Ordered by pixels meeting all conditions, then window id.")
        else:
            st.info("No follow-up windows under the default configuration.")
    with t2:
        rows = views.triage_window_rows(t, "review")
        if rows:
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                "Evidence discrepancies for human review (high model index, weaker observational "
                "support), not confirmed model faults."
            )
        else:
            st.info("No window met the review rule.")
    with t3:
        sens = t["sensitivity"]
        st.caption(sens["note"])
        base = {
            "variant": "DEFAULT (reported above)",
            "overrides": "",
            "index threshold": t["index_threshold_raw"],
            "follow-up windows": c["n_followup_windows"],
            "pixels meeting all": c["n_joint_pixels"],
            "review windows": c["n_review_windows"],
        }
        st.dataframe(
            [base]
            + [
                {
                    "variant": v["variant"],
                    "overrides": ", ".join(f"{k}={x}" for k, x in v["overrides"].items()),
                    "index threshold": v["index_threshold_raw"],
                    "follow-up windows": v["n_followup_windows"],
                    "pixels meeting all": v["n_joint_pixels"],
                    "review windows": v["n_review_windows"],
                }
                for v in sens["variants"]
            ],
            hide_index=True,
            width="stretch",
        )
    with t4:
        src = art["source"]
        st.markdown(
            f"Source package `{src['package_sha256'][:16]}…` of `{src['bundle']}` (region "
            f"`{src['region_id']}`), native grid {src['geometry']['width']}×"
            f"{src['geometry']['height']} px at {src['geometry']['resolution_m']} m, no "
            "resampling. Window coordinates come from the bundle's own lunar projection."
        )
        st.dataframe(
            [
                {
                    "layer": k,
                    "file": v["file_id"],
                    "unit": v["unit"],
                    "sha256": v["sha256"][:16] + "…",
                }
                for k, v in src["files"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        st.json(t["config"], expanded=False)
        for lim in t["limitations"]:
            st.markdown(f"- {plain(lim)}")
        st.code(
            f"python -m iceclaw.cli triage --run {view.run_id}\n"
            f"python -m iceclaw.cli verify-triage --artifact runs/_derived/{view.run_id}/triage "
            f"--run {view.run_id}",
            language="bash",
        )
        d1, d2 = st.columns(2)
        import json as _json

        d1.download_button(
            "⬇ triage.json (post-processing)",
            data=_json.dumps(art, indent=1, sort_keys=True),
            file_name=f"{view.run_id}-triage.json",
            mime="application/json",
        )
        d2.download_button(
            "⬇ triage_windows.csv (post-processing)",
            data=ta.windows_csv(art),
            file_name=f"{view.run_id}-triage_windows.csv",
            mime="text/csv",
        )

    st.markdown("##### Optional targeted collaborative review of one window")
    st.caption(
        "The four specialists review ONE window together through the same Microsoft Agent "
        "Framework runtime: each measures its own layers on that window, the Skeptic may request "
        "tests from the owner of a layer, and each files its own review. Nothing is scripted; "
        "'consistent' and 'insufficient evidence' are valid findings, and a missing review makes "
        "the review INCOMPLETE."
    )
    ids = t["selection"]["followup_ids"] + t["selection"]["review_ids"]
    by_id = {w["id"]: w for w in t["windows"]}
    snap = mgr.snapshot()
    running = snap.state == runner.RUNNING
    if not ids:
        st.info("No follow-up or review windows to review.")
    else:
        pick = st.selectbox(
            "Window",
            ids,
            key="review_window",
            format_func=lambda i: (
                f"{by_id[i]['label']} · "
                f"{'follow-up' if i in t['selection']['followup_ids'] else 'review'} · "
                f"({by_id[i]['centre_lat']:.3f}, {by_id[i]['centre_lon']:.3f})"
            ),
        )
        b = services_budget()
        st.caption(
            f"Budget for the whole review (shared by all four): ≤ {b.max_llm_calls} model calls, "
            f"≤ {b.max_tool_calls} tool calls, ≤ 64,000 input tokens. A comparable benchmark case "
            "used about 58k tokens and 15 calls. Paid OpenAI calls."
        )
        if not status["ok"]:
            st.warning(
                f"Targeted review is not available: **{status['code']}**. {status['message']}",
                icon="🔑",
            )
        can = bool(status["ok"] and not running)
        confirm = st.checkbox(
            "I understand this makes paid OpenAI requests.", key="confirm_review", disabled=not can
        )

        def _go_review() -> None:
            res = mgr.start_review(
                source_dir=view.source_dir, window_id=int(pick), bundle_name=view.run_id
            )
            st.session_state["review_start_result"] = (res.started, res.code, res.message)
            st.session_state["confirm_review"] = False

        st.button(
            "🧭 Start targeted review",
            type="primary",
            disabled=not (can and confirm),
            on_click=_go_review,
            help="Starts one real review. Disabled until confirmed; refused while a run is active.",
        )
        res = st.session_state.pop("review_start_result", None)
        if res and not res[0]:
            st.error(f"Review not started: **{res[1]}** {runner.scrub(res[2])}", icon="⛔")
    reviews = views.saved_reviews(settings.runs_root, view.run_id)
    if reviews:
        st.markdown(f"**Saved targeted reviews of this bundle ({len(reviews)})**")
        for rec in reviews:
            with st.expander(
                f"{rec['target']['label']} · {rec['arm'].replace('_', ' ')} · {rec['status']}",
                expanded=len(reviews) == 1,
            ):
                render_review(rec)
    else:
        st.caption("No targeted review has been run for this bundle.")


def services_budget():
    from iceclaw.services import DEFAULT_REVIEW_BUDGET

    return DEFAULT_REVIEW_BUDGET


# ---------------------------------------------------------------------- 6. verify + export
def verify_section(view: views.BundleView, svc: InvestigationService) -> None:
    st.subheader("6 · Verify and export")
    results = st.session_state.setdefault("verify_results", {})
    key = view.run_id
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button(
            "✔ Verify findings",
            type="primary",
            width="stretch",
            help="Runs the existing offline verifier on this bundle: no model call, no API key, "
            "no network.",
        ):
            with st.spinner("Recomputing measurements from the embedded source arrays…"):
                try:
                    rep = svc.verify(view.path)
                    results[key] = {
                        "ok": rep.ok,
                        "mode": rep.mode,
                        "rows": views.verification_rows(rep),
                        "at": datetime.now(UTC).strftime("%H:%M:%S UTC"),
                    }
                except Exception as e:  # noqa: BLE001
                    results[key] = {
                        "ok": False,
                        "error": f"{type(e).__name__}: {runner.scrub(str(e))}",
                        "rows": [],
                        "at": datetime.now(UTC).strftime("%H:%M:%S UTC"),
                    }
    res = results.get(key)
    with c2:
        st.caption(
            "Re-derives evidence, rankings and report from the events, recomputes every recorded "
            "measurement from the embedded rasters, checks hashes, units, references and that no "
            "agent statement claims detected ice."
        )
    if res is None:
        st.info("Not verified in this session yet. Press **Verify findings**.")
    elif res.get("error"):
        st.error(f"Verification could not run: {res['error']}", icon="⛔")
    else:
        n_fail = sum(r["outcome"] == "FAIL" for r in res["rows"])
        n_warn = sum(r["outcome"] == "WARN" for r in res["rows"])
        n_pass = sum(r["outcome"] == "PASS" for r in res["rows"])
        if res["ok"]:
            st.success(
                f"**VERIFIED** at {res['at']}: {n_pass} checks passed, {n_fail} failed, "
                f"{n_warn} warning(s). Source: {res['mode']}.",
                icon="✅",
            )
        else:
            st.error(
                f"**VERIFICATION FAILED** at {res['at']}: {n_fail} check(s) failed. Do not rely on "
                "this bundle. Download is disabled.",
                icon="❌",
            )
        icon = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "WARN": "⚠️ WARN"}
        st.dataframe(
            [
                {"outcome": icon[r["outcome"]], "check": r["check"], "detail": r["detail"]}
                for r in res["rows"]
            ],
            hide_index=True,
            width="stretch",
            height=min(35 * len(res["rows"]) + 40, 560),
        )
    st.markdown("##### Download")
    failed = bool(res and not res.get("ok"))
    try:
        data = cached_zip(str(view.path), stamp_of(view.path))
    except views.DownloadRefused as e:
        st.error(f"Download unavailable: {e}", icon="🚫")
        return
    st.download_button(
        "⬇ Download public bundle (iceclaw-bundle/2, .zip)",
        data=data,
        file_name=f"{view.run_id}.zip",
        mime="application/zip",
        disabled=failed,
        help="Public-redacted package: events, evidence, report, rankings, source rasters.",
    )
    st.caption(
        f"{views.PRIVATE_EXCLUSION} Verify it anywhere with "
        f"`python -m iceclaw.cli verify --run {view.run_id}` (no key or GPU needed)."
    )


# ------------------------------------------------------------------------------ main page
def main() -> None:
    settings = get_settings()
    svc = InvestigationService(settings)
    cat = cached_catalog(settings, roots_key(settings))
    bundles = views.discover_public_bundles(settings.runs_root)
    mgr = runner.get_manager(
        settings.runs_root,
        lambda: InvestigationService(settings),
        allowed_regions=lambda: set(cat.investigable_ids),
    )
    header()
    st.sidebar.header("Region")
    row, rows = pick_region(cat, bundles)
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Evidence classes**  \n• Model output (NASA–IBM prediction)  \n• Observation-derived layer "
        "(TMAX, LPSR, SLOPE)  \n• Deterministic calculation  \n• AI hypothesis (language-model text)"
    )
    st.sidebar.caption(
        f"Runs folder: `{settings.runs_root}`  \nModel config: "
        + (
            f"`{settings.llm_model}`"
            if settings.llm_model
            else "no model configured (live disabled)"
        )
    )
    if cat.warning:
        st.warning(cat.warning, icon="📁")
    if row is None:
        st.error(
            "No regions are available: no local catalog and no saved investigation was found.",
            icon="📁",
        )
        return
    if row.status == regions_mod.OBSERVATION_ONLY:
        unavailable_panel(row, sum(r.status == regions_mod.OBSERVATION_ONLY for r in rows))
        return

    status = mgr.config_status()
    mine, view, load_error = choose_bundle(row, bundles, mgr)
    region_section(row, view, settings, cat)
    st.markdown("---")
    st.subheader("2 · Investigate")
    left, right = st.columns(2, gap="large")
    with left:
        recorded_card(mine, view, load_error)
    with right:
        live_card(row, mgr, status)
    progress_panel(mgr)
    if view is None:
        st.info("Load a saved investigation or run a live one to see findings.")
        return
    banner_for(view, mgr)
    st.markdown("---")
    agents_section(view)
    st.markdown("---")
    findings_section(view)
    st.markdown("---")
    triage_section(view, settings, mgr, status)
    st.markdown("---")
    verify_section(view, svc)


main()
