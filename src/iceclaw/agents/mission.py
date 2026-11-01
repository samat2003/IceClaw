"""Adaptive policy agents over replayed sensor messages; no LLM or hardware calls.

Agents lease roles from a shared capability queue rather than holding a fixed pipeline
position. Every handoff cites its triggering event. Sensor data never establishes ice.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=100)
    tick: int = Field(ge=0, le=120)
    source: str = Field(min_length=1, max_length=100)
    mode: Literal["simulated", "recorded", "live"]
    kind: Literal["optical_index", "temperature_k", "hazard", "robot_fault"]
    target: Literal["A", "B", "C"] | None = None
    value: float
    robot: Literal["rover-1", "rover-2"] | None = None

    @model_validator(mode="after")
    def valid_measurement(self):
        if self.kind == "robot_fault":
            if self.robot is None or self.value != 1:
                raise ValueError("robot_fault requires robot and value=1")
        elif self.target is None:
            raise ValueError("map observations require a target")
        if self.kind == "optical_index" and not 0 <= self.value <= 1:
            raise ValueError("optical_index requires a normalized demo index in [0,1]")
        if self.kind == "temperature_k" and not 0 <= self.value <= 500:
            raise ValueError("temperature_k must be in [0,500] K")
        if self.kind == "hazard" and self.value not in (0, 1):
            raise ValueError("hazard must be 0 or 1")
        return self


def demo_observations() -> list[Observation]:
    rows = [
        (0, "optical_index", "A", 0.9, None),
        (0, "temperature_k", "A", 90, None),
        (0, "optical_index", "B", 0.7, None),
        (0, "temperature_k", "B", 95, None),
        (0, "optical_index", "C", 0.5, None),
        (0, "temperature_k", "C", 100, None),
        (3, "hazard", "A", 1, None),
        (4, "hazard", "A", 0, None),
        (5, "optical_index", "C", 0.98, None),
        (7, "robot_fault", None, 1, "rover-2"),
        (9, "temperature_k", "B", 160, None),
        (12, "temperature_k", "A", 85, None),
    ]
    return [
        Observation(
            id=f"obs-{i:03d}",
            tick=t,
            kind=k,
            target=target,
            value=v,
            robot=r,
            source="synthetic-sensor-replay",
            mode="simulated",
        )
        for i, (t, k, target, v, r) in enumerate(rows)
    ]


def run_mission(observations: list[Observation], *, ticks: int = 40) -> dict:
    if not 1 <= ticks <= 120 or len(observations) > 1000:
        raise ValueError("ticks must be 1..120 and at most 1000 observations are accepted")
    if len({o.id for o in observations}) != len(observations):
        raise ValueError("observation IDs must be unique")
    targets = {
        k: dict(
            id=k,
            row=r,
            col=c,
            index=None,
            temperature=None,
            hazard=False,
            status="insufficient",
            sampled=False,
            evidence=[],
        )
        for k, r, c in [("A", 3, 6), ("B", 10, 4), ("C", 7, 12)]
    }
    rovers = [
        dict(id=f"rover-{i + 1}", row=0, col=0, target=None, state="idle", moves=0)
        for i in range(2)
    ]
    agents = [dict(id=f"agent-{i + 1}", role="available", tasks=0) for i in range(3)]
    events, frames = [], []
    cursor = 0

    def event(tick, agent, role, action, reason, parents=(), **details):
        eid = f"evt-{len(events):04d}"
        events.append(
            dict(
                id=eid,
                tick=tick,
                agent=agent,
                role=role,
                action=action,
                reason=reason,
                parents=list(parents),
                **details,
            )
        )
        return eid

    def lease(tick, role, action, reason, parents=(), **details):
        nonlocal cursor
        agent = agents[cursor % len(agents)]
        cursor += 1
        old = agent["role"]
        agent["role"], agent["tasks"] = role, agent["tasks"] + 1
        eid = event(tick, agent["id"], role, action, reason, parents, previous_role=old, **details)
        return eid

    schedule = sorted(observations, key=lambda o: (o.tick, o.id))
    for tick in range(ticks + 1):
        updates = []
        for obs in (o for o in schedule if o.tick == tick):
            oid = event(
                tick,
                "sensor-adapter",
                "ingestion",
                "observation",
                "Received typed sensor observation",
                observation=obs.model_dump(),
            )
            if obs.kind == "robot_fault":
                rover = next(r for r in rovers if r["id"] == obs.robot)
                rover["state"], rover["target"] = "fault", None
                updates.append(
                    lease(
                        tick,
                        "fleet-recovery",
                        "retire_robot",
                        "Fault telemetry: release task for another rover",
                        [oid],
                        robot=obs.robot,
                    )
                )
            else:
                target = targets[obs.target]
                key = {
                    "temperature_k": "temperature",
                    "optical_index": "index",
                    "hazard": "hazard",
                }[obs.kind]
                target[key] = obs.value
                target["evidence"].append(oid)
                review = lease(
                    tick,
                    "sensor-analysis",
                    "update_map",
                    f"{obs.kind} updated target {obs.target}",
                    [oid],
                    target=obs.target,
                )
                updates.append(review)
        # Evidence assessment is leased again on every update, including after samples.
        if updates:
            assessment = lease(
                tick, "evidence-audit", "reassess", "New evidence changes shared map", updates
            )
            for target in targets.values():
                if target["hazard"]:
                    target["status"] = "hazard"
                elif target["temperature"] is None or target["index"] is None:
                    target["status"] = "insufficient"
                elif target["temperature"] > 110:
                    target["status"] = "withheld"
                else:
                    target["status"] = "screened"
            lease(
                tick,
                "mission-planning",
                "replan",
                "Rank eligible unsampled targets by current index, then travel distance",
                [assessment],
            )
        assigned = {r["target"] for r in rovers if r["target"]}
        for rover in rovers:
            if rover["state"] == "fault":
                continue
            eligible = [
                t
                for t in targets.values()
                if t["status"] == "screened"
                and not t["sampled"]
                and (t["id"] not in assigned or t["id"] == rover["target"])
            ]
            eligible.sort(
                key=lambda t: (
                    -t["index"],
                    abs(t["row"] - rover["row"]) + abs(t["col"] - rover["col"]),
                    t["id"],
                )
            )
            best = eligible[0]["id"] if eligible else None
            if best != rover["target"]:
                previous = rover["target"]
                assigned.discard(previous)
                rover["target"] = best
                if best:
                    assigned.add(best)
                lease(
                    tick,
                    "mission-planning",
                    "redirect" if previous else "assign",
                    "Current evidence and fleet availability changed the preferred task",
                    targets[best]["evidence"] if best else updates,
                    robot=rover["id"],
                    target=best,
                    previous_target=previous,
                )
            if best is None:
                rover["state"] = "idle"
                continue
            target = targets[best]
            if (rover["row"], rover["col"]) == (target["row"], target["col"]):
                lease(
                    tick,
                    "sampling",
                    "mock_sample",
                    "Arrived at currently eligible target",
                    target["evidence"],
                    robot=rover["id"],
                    target=best,
                )
                target["sampled"] = True
                assigned.discard(best)
                rover["target"], rover["state"] = None, "sampled"
            else:
                axis = "row" if rover["row"] != target["row"] else "col"
                rover[axis] += 1 if target[axis] > rover[axis] else -1
                rover["moves"] += 1
                rover["state"] = "moving"
        frames.append(
            deepcopy(
                dict(
                    tick=tick,
                    targets=list(targets.values()),
                    rovers=rovers,
                    agents=agents,
                    event_count=len(events),
                )
            )
        )
    return dict(
        format="iceclaw-adaptive-mission/1",
        controller="deterministic policy agents",
        coordinates="synthetic 20x20 local grid",
        simulated_robots=True,
        limitations=[
            "No ice detection or physical sampling",
            "No collision avoidance",
            "Hazards exclude target locations; routes are not terrain validated",
            "Uploaded live labels are user declarations, not authenticated feeds",
        ],
        observations=[o.model_dump() for o in schedule],
        events=events,
        frames=frames,
    )
