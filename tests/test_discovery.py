from iceclaw.data import discovery
from iceclaw.data.online import FeedError


def test_gap_priorities_change_with_observations():
    assert discovery.discovery_needs([dict(status="cloudy")])[0] == "radar"
    assert discovery.discovery_needs([dict(status="image review candidate")])[0] == "thermal"


def test_dynamic_registry_discovery_and_handoff(monkeypatch):
    calls = []

    def request(url, body=None):
        calls.append((url, body))
        if body is None:
            return dict(
                collections=[
                    dict(id="new-radar", title="Synthetic aperture radar", license="public"),
                    dict(id="new-dem", title="Digital elevation map"),
                    dict(id="new-thermal", title="Thermal infrared"),
                ]
            ), "registry"
        return dict(
            type="FeatureCollection",
            features=[dict(id="found-item", assets={"measurement": {}}, properties={})],
        ), "item-hash"

    monkeypatch.setattr(discovery, "_request", request)
    result = discovery.discover("Svalbard (Earth Arctic)", ["radar", "thermal", "terrain"])
    assert len(calls) == 4
    assert result["resources"][0]["collection"] == "new-radar"
    assert result["resources"][0]["assets"] == ["measurement"]
    state = {}
    discovery.handoff(state, result)
    assert len(state["events"]) == 9
    discovery.handoff(state, result)
    assert len(state["events"]) == 9
    seen = set()
    for event in state["events"]:
        assert event["parent"] is None or event["parent"] in seen
        seen.add(event["id"])
    assert any(e["role"] == "resource-seeker" for e in state["events"])


def test_missing_coverage_and_failed_sources_do_not_become_measurements(monkeypatch):
    def request(url, body=None):
        if body is None:
            return dict(
                collections=[dict(id="radar", title="Radar"), dict(id="thermal", title="Thermal")]
            ), "registry"
        if body["collections"] == ["radar"]:
            return dict(type="FeatureCollection", features=[]), "empty"
        raise FeedError("unavailable")

    monkeypatch.setattr(discovery, "_request", request)
    result = discovery.discover("Iceland (Earth)", ["radar", "thermal"])
    assert all(r["item_id"] is None for r in result["resources"])
    state = {}
    discovery.handoff(state, result)
    assert not any(e["action"] == "request source-specific validation" for e in state["events"])
