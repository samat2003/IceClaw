"""Evidence-driven mission playback and dynamic role leases."""

import hashlib
import json

import plotly.graph_objects as go
import streamlit as st
from pydantic import TypeAdapter, ValidationError

from iceclaw.agents.mission import Observation, demo_observations, run_mission

st.set_page_config(page_title="IceClaw · Adaptive mission", page_icon="🛰️", layout="wide")
st.title("Adaptive multi-agent mission")
st.warning("SIMULATED ROVERS · synthetic local map · mock samples, no ice detection")
st.caption(
    "Policy agents share evidence and lease new roles as conditions change. "
    "No LLM calls. Uploaded observations are replayed; no hardware is connected."
)
demo = demo_observations()
upload = st.file_uploader(
    "Optional sensor replay (JSON array; download the example below)", type=["json"]
)
try:
    observations = (
        TypeAdapter(list[Observation]).validate_json(upload.getvalue()) if upload else demo
    )
    result = run_mission(observations, ticks=40)
except (ValidationError, ValueError) as error:
    st.error(f"Observation input rejected: {error}")
    st.stop()
st.download_button(
    "Download example sensor input",
    json.dumps([o.model_dump() for o in demo], indent=2),
    "sensor-replay.json",
)
st.caption(
    "Input ticks are relative replay time. Index is a normalized demo score; "
    "temperature is kelvin; hazard/fault are binary flags. Sensor source labels are "
    "declared by the uploader, not independently authenticated."
)
fingerprint = hashlib.sha256(json.dumps(result["observations"]).encode()).hexdigest()
if st.session_state.get("mission_input") != fingerprint:
    st.session_state.mission_tick = 0
    st.session_state.mission_input = fingerprint
playing = st.toggle("Auto-play sensor replay", value=False)
st.markdown(
    "**Default story:** tick 3: hazard → cancel A · tick 5: C priority rises → redirect · "
    "tick 7: rover fault → recover task · tick 9: B warms → withhold · "
    "tick 4: A hazard clears · tick 12: A temperature refreshed"
)


@st.fragment(run_every=0.8 if playing else None)
def playback():
    if playing:
        st.session_state.mission_tick = min(40, st.session_state.mission_tick + 1)
    tick = st.slider("Mission tick (pause auto-play to inspect)", 0, 40, key="mission_tick")
    frame = result["frames"][tick]
    targets = frame["targets"]
    colors = {
        "screened": "#38bdf8",
        "hazard": "#ef4444",
        "withheld": "#f59e0b",
        "insufficient": "#94a3b8",
    }
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[t["col"] for t in targets],
            y=[t["row"] for t in targets],
            mode="markers+text",
            text=[t["id"] + ": " + t["status"] for t in targets],
            textposition="top center",
            name="Evidence map",
            marker=dict(
                size=26,
                symbol="circle-open",
                line=dict(width=3),
                color=[colors[t["status"]] for t in targets],
            ),
        )
    )
    for i, rover in enumerate(frame["rovers"]):
        trail = [f["rovers"][i] for f in result["frames"][: tick + 1]]
        fig.add_trace(
            go.Scatter(
                x=[r["col"] for r in trail],
                y=[r["row"] for r in trail],
                mode="lines",
                line=dict(width=2, dash="dot"),
                name=rover["id"] + " trail",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[rover["col"]],
                y=[rover["row"]],
                mode="markers+text",
                marker=dict(size=16, symbol="x" if rover["state"] == "fault" else "square"),
                text=[rover["id"]],
                textposition="bottom center",
                name=rover["id"],
            )
        )
    fig.update_layout(
        height=500,
        title=f"Shared evidence map · tick {tick}",
        xaxis=dict(range=[-1, 20], title="Local grid column"),
        yaxis=dict(range=[20, -1], title="Local grid row", scaleanchor="x"),
    )
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Current agent roles")
        st.dataframe(frame["agents"], hide_index=True)
        st.subheader("Fleet")
        st.dataframe(frame["rovers"], hide_index=True)
    st.subheader("Evidence and sampling state")
    st.dataframe(
        [
            {k: t[k] for k in ("id", "index", "temperature", "hazard", "status", "sampled")}
            for t in targets
        ],
        hide_index=True,
    )
    visible = result["events"][: frame["event_count"]]
    st.subheader("Agent decisions and handoffs")
    st.dataframe(
        [
            {
                k: e.get(k)
                for k in (
                    "id",
                    "tick",
                    "agent",
                    "previous_role",
                    "role",
                    "action",
                    "target",
                    "robot",
                    "reason",
                    "parents",
                )
            }
            for e in visible[-18:]
        ],
        hide_index=True,
        use_container_width=True,
    )


playback()
st.download_button(
    "Download full mission evidence and replay",
    json.dumps(result, indent=2),
    "adaptive-mission.json",
    "application/json",
)
with st.expander("Limits and extension points"):
    st.write(result["limitations"])
    st.write(
        "Optical index, temperature, hazard and fault adapters are supported. "
        "Optical imagery, InSAR rasters, authenticated live telemetry and LLM mission "
        "planning are not integrated in this demo. Routes are toy grid movement."
    )
