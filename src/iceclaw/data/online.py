"""Automatic public STAC discovery. Scene metadata is evidence, not ice detection."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from urllib.request import Request, urlopen

ENDPOINT = "https://earth-search.aws.element84.com/v1/search"
REGIONS = {
    "Svalbard (Earth Arctic)": [15, 77, 18, 79],
    "Iceland (Earth)": [-20, 63, -17, 65],
    "Alaska (Earth)": [-151, 60, -148, 62],
}


class FeedError(RuntimeError):
    pass


def fetch_scenes(region: str) -> dict:
    if region not in REGIONS:
        raise ValueError("Unknown region")
    request = Request(
        ENDPOINT,
        data=json.dumps(
            {
                "collections": ["sentinel-2-l2a"],
                "bbox": REGIONS[region],
                "limit": 12,
                "sortby": [{"field": "properties.datetime", "direction": "desc"}],
            }
        ).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "IceClaw/0.0.1"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Catalog response exceeds size limit")
        payload = json.loads(raw)
        if payload.get("type") != "FeatureCollection" or not isinstance(
            payload.get("features"), list
        ):
            raise ValueError("Expected a STAC FeatureCollection")
        return dict(
            payload=payload,
            fetched_at=datetime.now(UTC).isoformat(),
            sha256=hashlib.sha256(raw).hexdigest(),
            endpoint=ENDPOINT,
            region=region,
        )
    except (OSError, ValueError, AttributeError) as error:
        raise FeedError(f"Satellite catalog unavailable: {error}") from error


def assess(snapshot: dict, *, now: datetime | None = None) -> list[dict]:
    """Quality triage using actual acquisition time and reported cloud cover."""
    now = now or datetime.now(UTC)
    rows = []
    for feature in snapshot["payload"]["features"]:
        if not isinstance(feature, dict):
            rows.append(dict(id="unknown", status="invalid metadata"))
            continue
        try:
            props = feature["properties"]
            acquired = datetime.fromisoformat(props["datetime"].replace("Z", "+00:00"))
            if acquired.tzinfo is None:
                raise ValueError("Missing acquisition timezone")
            age = (now - acquired).total_seconds() / 86400
            cloud = props.get("eo:cloud_cover")
            if cloud is not None:
                cloud = float(cloud)
                if not math.isfinite(cloud) or not 0 <= cloud <= 100:
                    raise ValueError("Invalid cloud cover")
            bbox = feature["bbox"]
            if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox):
                raise ValueError("Invalid footprint bounds")
            if not (-180 <= bbox[0] <= bbox[2] <= 180 and -90 <= bbox[1] <= bbox[3] <= 90):
                raise ValueError("Footprint outside geographic coordinates")
            status = (
                "future timestamp"
                if age < 0
                else "stale"
                if age > 7
                else "cloud unknown"
                if cloud is None
                else "cloudy"
                if cloud > 35
                else "image review candidate"
            )
            rows.append(
                dict(
                    id=feature["id"],
                    acquired_at=props["datetime"],
                    age_days=round(age, 2),
                    cloud_percent=cloud,
                    status=status,
                    lon=(bbox[0] + bbox[2]) / 2,
                    lat=(bbox[1] + bbox[3]) / 2,
                    bbox=bbox,
                    assets=feature.get("assets", {}),
                )
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            # A malformed scene cannot silently become a task candidate.
            rows.append(dict(id=str(feature.get("id", "unknown")), status="invalid metadata"))
    return rows


def update_agents(state: dict, snapshot: dict, rows: list[dict]) -> dict:
    """Persistent change-driven role leases, deduped by per-scene content hash."""
    for key, default in (("seen", {}), ("events", []), ("roles", {}), ("cursor", 0)):
        state.setdefault(key, default)
    for row in rows:
        content = {k: v for k, v in row.items() if k != "age_days"}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        if state["seen"].get(row["id"]) == digest:
            continue
        state["seen"][row["id"]] = digest
        parent = None
        for role, action in (
            ("catalog-observer", "detect scene or metadata change"),
            ("quality-auditor", row["status"]),
            (
                "task-planner",
                "request image analysis"
                if row["status"] == "image review candidate"
                else "withhold task; await better imagery",
            ),
        ):
            agent = f"agent-{state['cursor'] % 3 + 1}"
            state["cursor"] += 1
            # Rotate leases between updates so agents can move between roles.
            if role == "task-planner":
                state["cursor"] += 1
            eid = f"online-{state['cursor']:06d}"
            state["roles"][agent] = role
            state["events"].append(
                dict(
                    id=eid,
                    agent=agent,
                    role=role,
                    action=action,
                    scene=row["id"],
                    parent=parent,
                    source_sha256=snapshot["sha256"],
                    fetched_at=snapshot["fetched_at"],
                )
            )
            parent = eid
    return state
