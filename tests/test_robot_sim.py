from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from iceclaw.science.robot_sim import DEMO_TARGETS, Target, simulate


def test_multiple_robots_sample_each_target_once_and_only_after_arrival():
    result = simulate(list(DEMO_TARGETS))
    assert set(result["sampled"]) == {t.id for t in DEMO_TARGETS}
    assert len(result["sampled"]) == len(set(result["sampled"]))
    assert result["unvisited"] == []
    assignments = [e for e in result["events"] if e["action"] == "assign"]
    assert len({e["rover"] for e in assignments}) == 3
    for event in result["events"]:
        if event["action"] == "mock_sample_no_assay":
            target = next(t for t in DEMO_TARGETS if t.id == event["target"])
            prior = next(
                r for r in result["frames"][event["tick"] - 1] if r["id"] == event["rover"]
            )
            assert (prior["row"], prior["col"]) == (target.row, target.col)


def test_motion_and_budgets_and_reproducibility():
    result = simulate(list(DEMO_TARGETS), robots=2, travel_budget=12)
    assert result == simulate(list(DEMO_TARGETS), robots=2, travel_budget=12)
    for previous, current in zip(result["frames"], result["frames"][1:], strict=False):
        for a, b in zip(previous, current, strict=True):
            assert abs(a["row"] - b["row"]) + abs(a["col"] - b["col"]) <= 1
            assert b["moves"] <= 12
    assert result["unvisited"]


def test_zero_budget_and_empty_targets():
    assert simulate(list(DEMO_TARGETS), travel_budget=0)["sampled"] == []
    assert simulate([])["unvisited"] == []
    assert simulate([Target("base", 0, 0)], travel_budget=0)["sampled"] == ["base"]


def test_one_rover_can_visit_multiple_targets():
    result = simulate(list(DEMO_TARGETS), robots=1)
    assert len(result["sampled"]) == 3


def test_invalid_targets_rejected():
    with pytest.raises(ValueError):
        simulate([Target("outside", 20, 20)])
    with pytest.raises(ValueError):
        simulate([Target("same", 1, 1), Target("same", 2, 2)])


def test_page_works_without_data_or_api_key():
    page = Path(__file__).resolve().parents[1] / "pages/1_Robot_Simulation.py"
    app = AppTest.from_file(page).run(timeout=30)
    assert not app.exception
    assert "3/3 targets mock sampled" in app.markdown[0].value
