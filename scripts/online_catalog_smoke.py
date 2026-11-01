"""Opt-in real-network check: python scripts/online_catalog_smoke.py."""

import json
from pathlib import Path

from iceclaw.data.online import assess, fetch_scenes, update_agents

snapshot = fetch_scenes("Svalbard (Earth Arctic)")
rows = assess(snapshot)
state = update_agents({}, snapshot, rows)
path = Path("/tmp/iceclaw-online-evidence.json")
path.write_text(json.dumps(dict(snapshot=snapshot, scenes=rows, agents=state), indent=2))
print(f"Real catalog fetch: {snapshot['fetched_at']}")
print(f"Scenes: {len(rows)}; agent handoffs: {len(state['events'])}")
for row in rows[:3]:
    print(row["id"], row.get("acquired_at"), row.get("cloud_percent"), row["status"])
print(f"Evidence saved to {path}")
