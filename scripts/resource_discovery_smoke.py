"""Opt-in public-network resource-seeker check; no credentials or model calls."""

import json
from pathlib import Path

from iceclaw.data.discovery import discover, handoff

result = discover("Svalbard (Earth Arctic)", ["radar", "thermal", "terrain"])
state = {}
handoff(state, result)
Path("/tmp/iceclaw-resource-discovery.json").write_text(
    json.dumps(dict(discovery=result, events=state["events"]), indent=2)
)
for resource in result["resources"]:
    print(resource["collection"], resource["kind"], resource["status"], resource["item_id"])
print(f"Agent handoffs: {len(state['events'])}")
