"""Automatically polled satellite catalog view, with no upload workflow."""

import json

import plotly.graph_objects as go
import streamlit as st

from iceclaw.data import discovery, online


@st.cache_data(ttl=60, show_spinner=False)
def cached_feed(region):
    return online.fetch_scenes(region)


@st.cache_data(ttl=600, show_spinner=False)
def cached_discovery(region, needs):
    return discovery.discover(region, list(needs))


def render():
    st.subheader("Automatic online satellite monitoring")
    st.caption(
        "Earth satellite observations · Sentinel-2 L2A / Element 84 Earth Search. "
        "Catalog queried automatically every 60 seconds while this page is open."
    )
    region = st.selectbox("Watch region", list(online.REGIONS))
    st.info(
        "Live catalog access; acquisition times can be days or weeks old. "
        "This checks scene availability, age and cloud metadata—not ice presence. "
        "No robot hardware is connected."
    )

    @st.fragment(run_every=60)
    def monitor():
        store = st.session_state.setdefault("online_missions", {})
        state = store.setdefault(region, {})
        try:
            snapshot = cached_feed(region)
            state["snapshot"] = snapshot
            healthy = True
        except online.FeedError as error:
            st.error(str(error))
            healthy = False
            snapshot = state.get("snapshot")
            if snapshot is None:
                st.warning("No observations available. No synthetic readings are substituted.")
                return
            st.warning(
                "Connection lost: showing last successful catalog response; new tasks paused."
            )
        rows = online.assess(snapshot)
        if healthy:
            online.update_agents(state, snapshot, rows)
            try:
                found = cached_discovery(region, tuple(discovery.discovery_needs(rows)))
                state["discovery"] = found
                discovery.handoff(state, found)
            except online.FeedError as error:
                st.warning(str(error))
                st.caption("Resource discovery failed; any previous results below are historical.")
        st.write(f"Last successful fetch: {snapshot['fetched_at']} · {len(rows)} scenes")
        st.caption(
            "Scene review policy: acquisition within 7 days and reported cloud cover ≤35%. "
            "These are analyst-chosen quality checks; scene-wide cloud cover does not "
            "guarantee a clear target pixel."
        )
        valid = [r for r in rows if "lat" in r]
        fig = go.Figure()
        for row in valid:
            x0, y0, x1, y1 = row["bbox"]
            fig.add_trace(
                go.Scattermap(
                    lon=[x0, x1, x1, x0, x0],
                    lat=[y0, y0, y1, y1, y0],
                    mode="lines",
                    name=row["id"],
                    text=f"{row['acquired_at']} · {row['status']}",
                    showlegend=False,
                )
            )
        bounds = online.REGIONS[region]
        fig.update_layout(
            map=dict(
                style="open-street-map",
                zoom=5,
                center=dict(lon=(bounds[0] + bounds[2]) / 2, lat=(bounds[1] + bounds[3]) / 2),
            ),
            height=450,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            [{k: v for k, v in row.items() if k not in ("assets", "bbox")} for row in rows],
            hide_index=True,
        )
        st.subheader("Resources the agents sought automatically")
        found = state.get("discovery")
        if found:
            st.caption(
                f"Registry checked: {found['fetched_at']} · "
                "Discovery refreshes every 10 minutes; at most three candidate searches."
            )
            st.dataframe(found["resources"], hide_index=True, use_container_width=True)
            if not found["resources"]:
                st.info("No matching complementary collections advertised by this registry.")
            st.caption(
                "Availability confirms a catalog item covers this region. It does not "
                "confirm freshness, permission for every asset, sensor suitability or ice. "
                "Radar, thermal and terrain sources need their own validation tools."
            )
        st.subheader("Agents and automatic handoffs")
        st.dataframe(
            [dict(agent=k, role=v) for k, v in state.get("roles", {}).items()], hide_index=True
        )
        st.dataframe(state.get("events", [])[-24:], hide_index=True)
        if valid:
            scene = st.selectbox("Inspect detected scene", [r["id"] for r in valid])
            selected = next(r for r in valid if r["id"] == scene)
            thumbnail = selected["assets"].get("thumbnail", {}).get("href", "")
            if thumbnail.startswith("https://"):
                st.image(thumbnail, caption=f"Provider thumbnail · {selected['acquired_at']}")
        st.download_button(
            "Download fetched catalog and agent evidence",
            json.dumps(
                dict(
                    snapshot=snapshot,
                    discovery=state.get("discovery"),
                    events=state.get("events", []),
                ),
                indent=2,
            ),
            "online-satellite-evidence.json",
        )
        st.caption(
            "Source: https://element84.com/earth-search/ · ESA/Copernicus Sentinel-2. "
            "Footprints and thumbnails are map products; pixel-level analysis is not run."
        )

    monitor()
