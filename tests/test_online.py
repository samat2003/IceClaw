from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from iceclaw.data import online


@pytest.fixture(autouse=True)
def no_discovery_network(monkeypatch):
    from iceclaw.ui import online as ui

    monkeypatch.setattr(
        ui,
        "cached_discovery",
        lambda region, needs: dict(
            region=region,
            fetched_at="2026-11-01T12:00:00Z",
            registry_sha256="fixture",
            resources=[],
        ),
    )


def snapshot():
    return dict(
        sha256="test-hash",
        fetched_at="2026-11-01T12:00:00Z",
        payload=dict(
            features=[
                dict(
                    id="test-scene",
                    bbox=[15, 77, 16, 78],
                    assets={},
                    properties={"datetime": "2026-11-01T11:00:00Z", "eo:cloud_cover": 10},
                )
            ]
        ),
    )


def test_quality_and_changes_drive_tasks_without_duplicate_handoffs():
    snap = snapshot()
    now = datetime(2026, 11, 1, 12, tzinfo=UTC)
    rows = online.assess(snap, now=now)
    assert rows[0]["status"] == "image review candidate"
    state = online.update_agents({}, snap, rows)
    assert state["events"][-1]["action"] == "request image analysis"
    online.update_agents(state, snap, rows)
    assert len(state["events"]) == 3
    changed = deepcopy(snap)
    changed["payload"]["features"][0]["properties"]["eo:cloud_cover"] = 90
    online.update_agents(state, changed, online.assess(changed, now=now))
    assert len(state["events"]) == 6
    assert state["events"][-1]["action"].startswith("withhold")
    assert state["events"][0]["agent"] != state["events"][3]["agent"]


def test_stale_and_invalid_metadata_never_become_candidates():
    snap = snapshot()
    rows = online.assess(snap, now=datetime(2026, 12, 1, tzinfo=UTC))
    assert rows[0]["status"] == "stale"
    snap["payload"]["features"][0]["properties"]["eo:cloud_cover"] = float("nan")
    assert online.assess(snap)[0]["status"] == "invalid metadata"


def test_online_is_default_and_fetches_without_upload(monkeypatch):
    from iceclaw.ui import online as ui

    calls = []

    def fetch(region):
        calls.append(region)
        return snapshot()

    monkeypatch.setattr(ui, "cached_feed", fetch)
    path = Path(__file__).resolve().parents[1] / "pages/2_Adaptive_Mission.py"
    app = AppTest.from_file(path).run(timeout=30)
    assert not app.exception
    assert app.radio[0].value == "Online satellite monitoring"
    assert calls
    assert not app.get("file_uploader")


def test_offline_failure_is_visible_without_synthetic_fallback(monkeypatch):
    from iceclaw.ui import online as ui

    def fail(region):
        raise online.FeedError("network unavailable")

    monkeypatch.setattr(ui, "cached_feed", fail)
    path = Path(__file__).resolve().parents[1] / "pages/2_Adaptive_Mission.py"
    app = AppTest.from_file(path).run(timeout=30)
    assert not app.exception
    assert app.error and "network unavailable" in app.error[0].value


def test_connection_loss_retains_evidence_and_pauses_new_handoffs(monkeypatch):
    from iceclaw.ui import online as ui

    monkeypatch.setattr(ui, "cached_feed", lambda region: snapshot())
    path = Path(__file__).resolve().parents[1] / "pages/2_Adaptive_Mission.py"
    app = AppTest.from_file(path).run(timeout=30)
    assert not app.exception
    before = deepcopy(app.session_state["online_missions"])

    def fail(region):
        raise online.FeedError("disconnected")

    monkeypatch.setattr(ui, "cached_feed", fail)
    app.run()
    assert not app.exception and app.error
    assert app.session_state["online_missions"] == before


def test_resource_seeker_runs_automatically_on_online_page(monkeypatch):
    from iceclaw.ui import online as ui

    monkeypatch.setattr(ui, "cached_feed", lambda region: snapshot())
    resource = dict(
        collection="new-radar",
        reason="Optical evidence gap",
        kind="radar",
        status="available; needs source-specific validation",
        item_id="radar-item",
        source_sha256="item-hash",
        assets=["measurement"],
    )
    monkeypatch.setattr(
        ui,
        "cached_discovery",
        lambda region, needs: dict(
            resources=[resource], fetched_at="2026-11-01T12:00:00Z", registry_sha256="registry"
        ),
    )
    path = Path(__file__).resolve().parents[1] / "pages/2_Adaptive_Mission.py"
    app = AppTest.from_file(path).run(timeout=30)
    assert not app.exception
    state = app.session_state["online_missions"]["Svalbard (Earth Arctic)"]
    assert "new-radar" in state["resource_seen"]
    assert any(e["role"] == "resource-seeker" for e in state["events"])
