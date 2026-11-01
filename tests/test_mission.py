from pathlib import Path

import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from iceclaw.agents.mission import Observation, demo_observations, run_mission


def test_changes_drive_replanning_and_role_handoffs():
    result = run_mission(demo_observations())
    events = result["events"]
    assert any(e["action"] == "redirect" and e["tick"] == 3 for e in events)
    assert any(e["action"] == "redirect" and e["tick"] == 5 and e["target"] == "C" for e in events)
    assert any(e["action"] == "retire_robot" and e["tick"] == 7 for e in events)
    for agent in ("agent-1", "agent-2", "agent-3"):
        assert len({e["role"] for e in events if e["agent"] == agent}) >= 3
    seen = set()
    for e in events:
        assert set(e["parents"]) <= seen
        seen.add(e["id"])
    assert result == run_mission(demo_observations())


def test_no_sampling_without_current_evidence_and_no_motion_after_fault():
    result = run_mission(demo_observations())
    for frame in result["frames"]:
        assigned = [r["target"] for r in frame["rovers"] if r["target"]]
        assert len(assigned) == len(set(assigned))
        if frame["tick"] >= 7:
            assert frame["rovers"][1]["state"] == "fault"
            assert frame["rovers"][1]["row"] == result["frames"][7]["rovers"][1]["row"]
    for event in result["events"]:
        if event["action"] == "mock_sample":
            frame = result["frames"][event["tick"]]
            target = next(t for t in frame["targets"] if t["id"] == event["target"])
            rover = next(r for r in frame["rovers"] if r["id"] == event["robot"])
            assert target["status"] == "screened"
            assert (target["row"], target["col"]) == (rover["row"], rover["col"])
            assert event["parents"]
    assert any(t["sampled"] for t in result["frames"][-1]["targets"])


def test_absent_evidence_and_rejected_sensor_input():
    assert not any(t["sampled"] for t in run_mission([])["frames"][-1]["targets"])
    for value in (float("nan"), -1, 900):
        with pytest.raises(ValidationError):
            Observation(
                id="x",
                tick=0,
                source="test",
                mode="recorded",
                kind="temperature_k",
                target="A",
                value=value,
            )
    with pytest.raises(ValueError):
        run_mission([demo_observations()[0]] * 2)


def test_counterfactual_observations_change_decisions():
    baseline = demo_observations()
    altered = [o for o in baseline if o.tick != 5]
    original = run_mission(baseline)
    changed = run_mission(altered)
    assert original["frames"][5]["rovers"][0]["target"] == "C"
    assert changed["frames"][5]["rovers"][0]["target"] == "A"
    # A new warm reading excludes B even though its optical index is unchanged.
    b = next(t for t in original["frames"][9]["targets"] if t["id"] == "B")
    assert b["status"] == "withheld" and not b["sampled"]


def test_fault_releases_task_for_healthy_rover():
    observations = [o for o in demo_observations() if o.tick < 1]
    observations.append(
        Observation(
            id="fault",
            tick=1,
            source="test",
            mode="simulated",
            kind="robot_fault",
            robot="rover-1",
            value=1,
        )
    )
    result = run_mission(observations)
    assert any(
        e["action"] in ("assign", "redirect")
        and e.get("target") == "A"
        and e.get("robot") == "rover-2"
        for e in result["events"]
    )


def test_page_and_timeline_without_credentials():
    path = Path(__file__).resolve().parents[1] / "pages/2_Adaptive_Mission.py"
    app = AppTest.from_file(path).run(timeout=30)
    assert not app.exception
    app.slider[0].set_value(9).run()
    assert not app.exception
    assert app.slider[0].value == 9
