"""Deterministic, bounded evidence digest for per-turn prompts.

The full history always stays in SQLite; the model sees at most ``max_chars`` of digest per turn.
Selection is a pure function of (events, agent) -- no randomness, no model:

1. events addressed to this agent first (test requests to it, challenges of its events, results of
   its requests), then claims/revisions/completions, then measurements; newest first in a class;
2. lines are added until the character cap is reached; the chosen lines are printed
   chronologically;
3. every line keeps the event id, ALL parent ids (to 12), status and short source-file ids; only
   the free text is clipped (``get_event`` returns any event in full);
4. anything omitted is summarised by id (compressed ranges), so nothing silently disappears.
"""

from __future__ import annotations

import json
import re
from typing import Any

from iceclaw.contracts import EventKind, EvidenceEvent

_PRIORITY = {
    "addressed": 0,
    EventKind.CHALLENGE: 1,
    EventKind.REVISION: 1,
    EventKind.CLAIM: 2,
    EventKind.TEST_RESULT: 2,
    EventKind.COMPLETION: 3,
    EventKind.FAILURE: 3,
    EventKind.MEASUREMENT: 4,
    EventKind.TEST_REQUEST: 4,
}


def _num(e: EvidenceEvent) -> int:
    m = re.search(r"(\d+)$", e.event_id)
    return int(m.group(1)) if m else 0


def _addressed_to(e: EvidenceEvent, agent_id: str, rt: Any) -> bool:
    if e.event_kind is EventKind.TEST_REQUEST:
        try:
            return str(json.loads(e.claim_or_request or "{}").get("to")) == agent_id
        except json.JSONDecodeError:
            return False
    if e.event_kind in (EventKind.CHALLENGE, EventKind.TEST_RESULT) and e.parent_event_ids:
        target = rt.store.get_event(e.parent_event_ids[0])
        return bool(target and target.agent_id == agent_id)
    return False


def compress_ids(ids: list[str]) -> str:
    """['evt-0003','evt-0004','evt-0005','evt-0009'] -> 'evt-0003..evt-0005, evt-0009'."""
    nums = sorted(int(re.search(r"(\d+)$", i).group(1)) for i in ids if re.search(r"(\d+)$", i))
    if not nums:
        return ""
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:  # type: ignore[list-item]
        if n is not None and n == prev + 1:
            prev = n
            continue
        out.append(f"evt-{start:04d}" if start == prev else f"evt-{start:04d}..evt-{prev:04d}")
        if n is not None:
            start = prev = n
    return ", ".join(out)


def select_events(
    rt: Any, agent_id: str, events: list[EvidenceEvent], max_chars: int, text_chars: int
) -> tuple[list[EvidenceEvent], list[EvidenceEvent]]:
    """(chosen in chronological order, omitted). Deterministic."""

    def key(e: EvidenceEvent) -> tuple[int, int]:
        cls = "addressed" if _addressed_to(e, agent_id, rt) else e.event_kind
        return (_PRIORITY[cls], -_num(e))

    chosen: list[EvidenceEvent] = []
    used = 0
    for e in sorted(events, key=key):
        line = rt.compact_event(e, text_chars)
        if used + len(line) + 1 > max_chars:
            continue  # skipped lines are reported by id in the digest's omission note
        chosen.append(e)
        used += len(line) + 1
    chosen_ids = {e.event_id for e in chosen}
    omitted = [e for e in events if e.event_id not in chosen_ids]
    return sorted(chosen, key=_num), sorted(omitted, key=_num)


def build_digest(
    rt: Any, agent_id: str, new_events: list[EvidenceEvent], max_chars: int, text_chars: int = 240
) -> str:
    """Bounded digest of events published by others since the agent's last turn."""
    if not new_events:
        return ""
    chosen, omitted = select_events(rt, agent_id, new_events, max_chars, text_chars)
    lines = [rt.compact_event(e, text_chars) for e in chosen]
    if omitted:
        lines.append(
            f"({len(omitted)} further new event(s) omitted for length: "
            f"{compress_ids([e.event_id for e in omitted])}; "
            "use list_events/get_event for any of them)"
        )
    return "\n".join(lines)


def build_own_summary(rt: Any, own_ids: list[str], limit: int = 4, text_chars: int = 160) -> str:
    events = [e for i in own_ids[-limit:] if (e := rt.store.get_event(i))]
    older = len(own_ids) - len(events)
    lines = [rt.compact_event(e, text_chars) for e in events]
    if older > 0:
        lines.insert(
            0, f"({older} earlier event(s) of yours omitted: {compress_ids(own_ids[:-limit])})"
        )
    return "\n".join(lines)
