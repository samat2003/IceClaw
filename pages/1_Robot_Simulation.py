"""Offline mission choreography demo, available from the workbench sidebar."""

import json
from pathlib import Path

import plotly.express as px
import streamlit as st

from iceclaw.config import load_settings
from iceclaw.reports.package import load_package
from iceclaw.reports.verify import verify_bundle
from iceclaw.science.robot_sim import DEMO_TARGETS, Target, simulate
from iceclaw.science.triage import screen_arrays

st.set_page_config(page_title="IceClaw · Robot simulation", page_icon="🤖", layout="wide")
st.title("Multi-robot follow-up simulation")
st.warning("SIMULATION — mock sampling produces no ice measurement or detection.")
st.caption(
    "Rule-based agents select, dispatch, navigate and sample. No LLM calls or hardware control. "
    "One cell per tick; no obstacles, collision avoidance, return-to-base or energy physics."
)
settings = load_settings()
bundles = sorted(p for p in settings.runs_root.glob("*") if (p / "input_manifest.json").is_file())
choice = st.selectbox(
    "Target source", ["Synthetic demo (no data needed)"] + [str(p) for p in bundles]
)
targets = list(DEMO_TARGETS)
size = 20
if choice != "Synthetic demo (no data needed)":
    with st.spinner("Verifying source bundle and screening follow-up pixels…"):
        report = verify_bundle(Path(choice))
        if not report.ok:
            st.error("Bundle verification failed; simulation refused.")
            st.code(report.render())
            st.stop()
        manifest = json.loads((Path(choice) / "input_manifest.json").read_text())
        arrays = load_package(Path(choice) / manifest["source_package"]["embedded_path"]).arrays
        triage = screen_arrays(arrays, sensitivity=False)
        selected = set(triage["selection"]["followup_ids"])
        targets = []
        for window in triage["windows"]:
            if window["id"] in selected:
                pixel = window["joint_pixels_example"][0]
                targets.append(Target(window["label"], pixel["row"], pixel["col"]))
        size = max(triage["grid"]["height"], triage["grid"]["width"])
    st.info("Targets are screened follow-up pixels from verified source arrays, not confirmed ice.")
if not targets:
    st.info("No follow-up targets meet the screening rule; no robots dispatched.")
    st.stop()
robots = st.slider("Robots", 1, 8, 3)
budget = st.slider("Travel budget per robot (grid moves)", 0, 2048, min(4 * size, 2048))
result = simulate(targets, robots=robots, size=size, travel_budget=budget)
result["source"] = choice
result["targets"] = [dict(id=t.id, row=t.row, col=t.col) for t in targets]
st.write(
    f"{len(result['sampled'])}/{len(targets)} targets mock sampled · "
    f"{len(result['unvisited'])} unreachable within remaining budgets"
)
rows = [dict(tick=tick, **rover) for tick, frame in enumerate(result["frames"]) for rover in frame]
fig = px.scatter(
    rows,
    x="col",
    y="row",
    color="id",
    animation_frame="tick",
    animation_group="id",
    hover_data=["state", "target", "moves"],
    range_x=[-1, size],
    range_y=[size, -1],
    title="▶ Play to follow dispatch → travel → mock sampling",
)
# Layout shapes persist across all animation frames.
for target in targets:
    fig.add_shape(
        type="circle",
        x0=target.col - 0.4,
        x1=target.col + 0.4,
        y0=target.row - 0.4,
        y1=target.row + 0.4,
        line_color="gray",
    )
    fig.add_annotation(x=target.col, y=target.row, text=target.id, showarrow=False, yshift=16)
fig.update_traces(marker_size=14)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
st.plotly_chart(fig, use_container_width=True)
st.caption("Gray rings: targets. Colored dots: rovers. Co-location is permitted in this toy model.")
with st.expander("Agent action log", expanded=True):
    st.dataframe(result["events"], use_container_width=True)
st.download_button(
    "Download simulation JSON",
    json.dumps(result, indent=2),
    "robot-simulation.json",
    "application/json",
)
