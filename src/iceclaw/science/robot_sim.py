"""Toy mission choreography, not navigation physics or evidence of ice.

Rule-based agents select targets, dispatch robots, and authorize mock sampling.
Robots move one grid cell per tick, then spend a separate tick sampling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    id: str
    row: int
    col: int


DEMO_TARGETS = (Target("demo-A", 4, 7), Target("demo-B", 12, 5), Target("demo-C", 9, 14))


def simulate(
    targets: list[Target], *, robots: int = 3, size: int = 20, travel_budget: int = 80
) -> dict:
    """Deterministic nearest-target dispatch, with stable ties and exclusive assignments.

    Each robot starts at (0, 0). Budget counts moves; sampling is a mock action.
    No obstacles, collision avoidance, return trip, energy model, or physical assay.
    """
    if not 1 <= robots <= 8 or not 2 <= size <= 512 or not 0 <= travel_budget <= 2048:
        raise ValueError("robots 1..8, size 2..512, travel_budget 0..2048 required")
    if len({t.id for t in targets}) != len(targets):
        raise ValueError("target IDs must be unique")
    if any(not (0 <= t.row < size and 0 <= t.col < size) for t in targets):
        raise ValueError("target outside grid")
    pending = sorted(targets, key=lambda t: t.id)
    fleet = [
        dict(id=f"rover-{i + 1}", row=0, col=0, target=None, moves=0, state="idle")
        for i in range(robots)
    ]
    events = []
    frames = []
    sampled = []

    def emit(tick, agent, action, rover=None, target=None):
        events.append(dict(tick=tick, agent=agent, action=action, rover=rover, target=target))

    emit(0, "science-agent", "select_followup_targets")
    frames.append([dict(r) for r in fleet])
    # Every active tick assigns, moves, samples, or retires a rover. Finite bound.
    for tick in range(1, robots * travel_budget + 3 * len(targets) + 2):
        for rover in fleet:
            if rover["state"] == "exhausted":
                continue
            if rover["target"] is None:
                reachable = [
                    t
                    for t in pending
                    if abs(t.row - rover["row"]) + abs(t.col - rover["col"])
                    <= travel_budget - rover["moves"]
                ]
                if not reachable:
                    rover["state"] = "idle"
                    continue
                target = min(
                    reachable,
                    key=lambda t: (abs(t.row - rover["row"]) + abs(t.col - rover["col"]), t.id),
                )
                pending.remove(target)
                rover["target"] = target.id
                rover["state"] = "assigned"
                emit(tick, "dispatch-agent", "assign", rover["id"], target.id)
                continue
            target = next(t for t in targets if t.id == rover["target"])
            if (rover["row"], rover["col"]) == (target.row, target.col):
                sampled.append(target.id)
                emit(tick, "sampling-agent", "mock_sample_no_assay", rover["id"], target.id)
                rover["state"], rover["target"] = "sampled", None
            elif rover["moves"] >= travel_budget:
                rover["state"] = "exhausted"
            else:
                axis = "row" if rover["row"] != target.row else "col"
                destination = target.row if axis == "row" else target.col
                rover[axis] += 1 if destination > rover[axis] else -1
                rover["moves"] += 1
                rover["state"] = "moving"
                emit(tick, "navigation-agent", "move", rover["id"], target.id)
        frames.append([dict(r) for r in fleet])
        if all(r["target"] is None for r in fleet):
            # Stop only if no idle rover can reach an unassigned target.
            if not any(
                abs(t.row - r["row"]) + abs(t.col - r["col"]) <= travel_budget - r["moves"]
                for t in pending
                for r in fleet
            ):
                break
    return dict(
        format="iceclaw-robot-simulation/1",
        simulated=True,
        policy="nearest reachable target; row-first grid movement",
        size=size,
        travel_budget=travel_budget,
        frames=frames,
        events=events,
        sampled=sampled,
        unvisited=[t.id for t in pending],
    )
