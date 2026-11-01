"""Bounded resource seeking from a trusted public STAC collection registry.

Discovery uses catalog metadata to propose sources; available assets are not
automatically trusted as scientific measurements or executed as agent instructions.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from urllib.parse import quote
from urllib.request import Request, urlopen

from iceclaw.data.online import ENDPOINT, REGIONS, FeedError

ROOT = ENDPOINT.removesuffix("/search")


def _request(url: str, body: dict | None = None) -> tuple[dict, str]:
    request = Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "User-Agent": "IceClaw/0.0.1"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            raw = response.read(5_000_001)
        if len(raw) > 5_000_000:
            raise ValueError("Resource response too large")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected catalog JSON object")
        return data, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as error:
        raise FeedError(f"Resource discovery unavailable: {error}") from error


def resource_kind(collection: dict) -> str | None:
    text = " ".join(str(collection.get(k, "")) for k in ("id", "title", "description"))
    for kind, pattern in (
        ("radar", r"sentinel-1|\bradar\b|\bsar\b"),
        ("thermal", r"landsat|thermal"),
        ("terrain", r"elevation|\bdem\b|cop-dem"),
        ("optical", r"sentinel-2|optical|multispectral"),
    ):
        if re.search(pattern, text, re.I):
            return kind
    return None


def discovery_needs(rows: list[dict]) -> list[str]:
    if not rows or any(r["status"] in ("stale", "cloudy", "cloud unknown") for r in rows):
        return ["radar", "thermal", "terrain"]
    return ["thermal", "terrain", "radar"]


def discover(region: str, needs: list[str]) -> dict:
    """Discover collection IDs dynamically; verify local item availability with <=3 searches."""
    if (
        region not in REGIONS
        or not needs
        or any(k not in ("radar", "thermal", "terrain") for k in needs)
    ):
        raise ValueError("Unsupported resource search")
    registry, registry_hash = _request(f"{ROOT}/collections")
    collections = registry.get("collections")
    if not isinstance(collections, list):
        raise FeedError("Resource registry has no collection list")
    ranked = []
    for collection in collections:
        if not isinstance(collection, dict) or not isinstance(collection.get("id"), str):
            continue
        kind = resource_kind(collection)
        if kind in needs:
            ranked.append((needs.index(kind), collection["id"], kind, collection))
    ranked.sort(key=lambda x: (x[0], x[1]))
    # Diversify before considering a second collection of the same kind.
    selected = []
    for kind in needs:
        candidates = [item for item in ranked if item[2] == kind]
        if candidates:
            selected.append(candidates[0])
    results = []
    for _, cid, kind, collection in selected[:3]:
        result = dict(
            collection=cid,
            title=collection.get("title", cid),
            kind=kind,
            license=collection.get("license", "unspecified"),
            url=f"{ROOT}/collections/{quote(cid, safe='')}",
            reason=(
                "Seek radar metadata because optical coverage may be stale or cloudy"
                if kind == "radar" and needs[0] == "radar"
                else f"Seek complementary {kind} observations for independent review"
            ),
            status="unavailable",
            assets=[],
            item_id=None,
        )
        try:
            payload, digest = _request(
                ENDPOINT, {"collections": [cid], "bbox": REGIONS[region], "limit": 1}
            )
            features = payload.get("features")
            if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
                raise FeedError("Invalid item search response")
            result["source_sha256"] = digest
            if features:
                item = features[0]
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("assets"), dict)
                    or not isinstance(item.get("id"), str)
                    or not isinstance(item.get("properties", {}), dict)
                ):
                    raise FeedError("Invalid resource item metadata")
                result.update(
                    status="available; needs source-specific validation",
                    item_id=item.get("id"),
                    assets=sorted(item["assets"]),
                    acquired_at=item.get("properties", {}).get("datetime"),
                )
            else:
                result["status"] = "no items covering selected region"
        except FeedError as error:
            result["status"] = str(error)
        results.append(result)
    return dict(
        region=region,
        needs=needs,
        registry_sha256=registry_hash,
        fetched_at=datetime.now(UTC).isoformat(),
        resources=results,
        budget="one registry request and at most three item searches per refresh",
        scope="Earth Search registry; metadata discovery, no arbitrary web crawling",
    )


def handoff(state: dict, discovery: dict) -> None:
    """Resource seeker hands new/changed resources to auditors and planners, once per change."""
    state.setdefault("resource_seen", {})
    state.setdefault("events", [])
    state.setdefault("roles", {})
    state.setdefault("cursor", 0)
    for resource in discovery["resources"]:
        digest = hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest()
        key = resource["collection"]
        if state["resource_seen"].get(key) == digest:
            continue
        state["resource_seen"][key] = digest
        parent = state["events"][-1]["id"] if state["events"] else None
        for role, action in (
            ("resource-seeker", resource["reason"]),
            ("resource-auditor", resource["status"]),
            (
                "task-planner",
                "request source-specific validation"
                if resource["item_id"]
                else "keep searching; no usable item yet",
            ),
        ):
            agent = f"agent-{state['cursor'] % 3 + 1}"
            state["cursor"] += 1
            if role == "task-planner":
                state["cursor"] += 1
            eid = f"resource-{state['cursor']:06d}"
            state["roles"][agent] = role
            state["events"].append(
                dict(
                    id=eid,
                    agent=agent,
                    role=role,
                    action=action,
                    scene=key,
                    parent=parent,
                    source_sha256=resource.get("source_sha256", discovery["registry_sha256"]),
                    fetched_at=discovery["fetched_at"],
                )
            )
            parent = eid
