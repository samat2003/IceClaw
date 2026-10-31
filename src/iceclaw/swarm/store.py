"""Append-only SQLite evidence store (events, tool runs, model calls).

Immutability is enforced by SQLite triggers that abort every UPDATE and DELETE, so the log cannot
be rewritten through this API or by a stray statement. Parent references are validated on append.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from iceclaw.contracts import EvidenceEvent, ToolMeasurement

_TABLES = """
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  investigation_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  event_kind TEXT NOT NULL,
  hop INTEGER NOT NULL,
  timestamp_utc TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_parents(
  event_id TEXT NOT NULL REFERENCES events(event_id),
  parent_id TEXT NOT NULL REFERENCES events(event_id),
  PRIMARY KEY (event_id, parent_id)
);
CREATE TABLE IF NOT EXISTS tool_runs(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  tool_run_id TEXT NOT NULL UNIQUE,
  investigation_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  event_id TEXT NOT NULL REFERENCES events(event_id),
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_calls(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  investigation_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  started_utc TEXT NOT NULL,
  finished_utc TEXT NOT NULL,
  ok INTEGER NOT NULL,
  error TEXT,
  input_tokens INTEGER,
  output_tokens INTEGER,
  total_tokens INTEGER,
  request_json TEXT NOT NULL,
  response_json TEXT
);
"""
_APPEND_ONLY = ("events", "event_parents", "tool_runs", "llm_calls")


class StoreError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class EvidenceStore:
    """Single-process append-only log. Use ``":memory:"`` or a file path."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_TABLES)
        for t in _APPEND_ONLY:
            for op in ("UPDATE", "DELETE"):
                self._db.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
                    f"BEGIN SELECT RAISE(ABORT, 'append-only: {t}'); END"
                )
        self._db.commit()

    @classmethod
    def open_readonly(cls, path: str | Path) -> EvidenceStore:
        """Open an existing history read-only (the file is never modified)."""
        self = cls.__new__(cls)
        self.path = str(path)
        self._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        return self

    def close(self) -> None:
        self._db.close()

    # ---------------------------------------------------------------- events
    def append_event(self, event: EvidenceEvent, hop: int) -> int:
        """Append one event; returns its sequence number. Parents must already exist."""
        for pid in event.parent_event_ids:
            row = self._db.execute(
                "SELECT investigation_id FROM events WHERE event_id=?", (pid,)
            ).fetchone()
            if row is None:
                raise StoreError("UNKNOWN_PARENT", f"parent event {pid!r} does not exist")
            if row["investigation_id"] != event.investigation_id:
                raise StoreError("CROSS_INVESTIGATION_PARENT", pid)
        try:
            cur = self._db.execute(
                "INSERT INTO events(event_id, investigation_id, agent_id, event_kind, hop,"
                " timestamp_utc, payload) VALUES (?,?,?,?,?,?,?)",
                (
                    event.event_id,
                    event.investigation_id,
                    event.agent_id,
                    event.event_kind.value,
                    hop,
                    event.timestamp_utc.isoformat(),
                    event.model_dump_json(),
                ),
            )
        except sqlite3.IntegrityError as e:
            raise StoreError("DUPLICATE_EVENT_ID", event.event_id) from e
        for pid in event.parent_event_ids:
            self._db.execute(
                "INSERT INTO event_parents(event_id, parent_id) VALUES (?,?)",
                (event.event_id, pid),
            )
        self._db.commit()
        return int(cur.lastrowid)

    @staticmethod
    def _event(row: sqlite3.Row) -> EvidenceEvent:
        return EvidenceEvent.model_validate_json(row["payload"])

    def get_event(self, event_id: str) -> EvidenceEvent | None:
        row = self._db.execute(
            "SELECT payload FROM events WHERE event_id=?", (event_id,)
        ).fetchone()
        return None if row is None else self._event(row)

    def seq_of(self, event_id: str) -> int | None:
        row = self._db.execute("SELECT seq FROM events WHERE event_id=?", (event_id,)).fetchone()
        return None if row is None else int(row["seq"])

    def hop_of(self, event_id: str) -> int | None:
        row = self._db.execute("SELECT hop FROM events WHERE event_id=?", (event_id,)).fetchone()
        return None if row is None else int(row["hop"])

    def events(self, since_seq: int = 0) -> list[tuple[int, EvidenceEvent]]:
        rows = self._db.execute(
            "SELECT seq, payload FROM events WHERE seq>? ORDER BY seq", (since_seq,)
        ).fetchall()
        return [(int(r["seq"]), self._event(r)) for r in rows]

    def count_events(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def max_seq(self) -> int:
        return int(self._db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0])

    # ------------------------------------------------------------- tool runs
    def append_tool_run(self, m: ToolMeasurement, agent_id: str, event_id: str, inv: str) -> None:
        try:
            self._db.execute(
                "INSERT INTO tool_runs(tool_run_id, investigation_id, agent_id, event_id, payload)"
                " VALUES (?,?,?,?,?)",
                (m.tool_run_id, inv, agent_id, event_id, m.model_dump_json()),
            )
        except sqlite3.IntegrityError as e:
            raise StoreError("DUPLICATE_TOOL_RUN", m.tool_run_id) from e
        self._db.commit()

    def get_tool_run(self, tool_run_id: str) -> tuple[ToolMeasurement, str, str] | None:
        """(measurement, event_id, agent_id) or None."""
        row = self._db.execute(
            "SELECT payload, event_id, agent_id FROM tool_runs WHERE tool_run_id=?", (tool_run_id,)
        ).fetchone()
        if row is None:
            return None
        return ToolMeasurement.model_validate_json(row["payload"]), row["event_id"], row["agent_id"]

    def tool_runs(self) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT * FROM tool_runs ORDER BY seq").fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # ------------------------------------------------------------- llm calls
    def append_llm_call(self, **f: Any) -> None:
        self._db.execute(
            "INSERT INTO llm_calls(investigation_id, agent_id, provider, model, started_utc,"
            " finished_utc, ok, error, input_tokens, output_tokens, total_tokens, request_json,"
            " response_json) VALUES (:investigation_id,:agent_id,:provider,:model,:started_utc,"
            ":finished_utc,:ok,:error,:input_tokens,:output_tokens,:total_tokens,:request_json,"
            ":response_json)",
            f,
        )
        self._db.commit()

    def llm_calls(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute("SELECT * FROM llm_calls ORDER BY seq")]

    # ---------------------------------------------------------- export/verify
    def export_jsonl(self, path: str | Path) -> int:
        """Write ``events.jsonl`` (one validated EvidenceEvent per line, in log order)."""
        n = 0
        with Path(path).open("w", encoding="utf-8") as f:
            for _, e in self.events():
                f.write(e.model_dump_json() + "\n")
                n += 1
        return n

    def verify(self) -> list[str]:
        """Machine check of the saved history; returns problems (empty = consistent)."""
        problems: list[str] = []
        seen: dict[str, int] = {}
        for seq, e in self.events():
            for pid in e.parent_event_ids:
                if pid not in seen:
                    problems.append(f"{e.event_id}: parent {pid} missing or later in the log")
            parent_hops = [self.hop_of(p) or 0 for p in e.parent_event_ids]
            expect = 0 if not parent_hops else 1 + max(parent_hops)
            if self.hop_of(e.event_id) != expect:
                problems.append(f"{e.event_id}: hop {self.hop_of(e.event_id)} != expected {expect}")
            if e.tool_run_id and self.get_tool_run(e.tool_run_id) is None:
                problems.append(f"{e.event_id}: tool run {e.tool_run_id} not stored")
            seen[e.event_id] = seq
        return problems

    def __iter__(self) -> Iterator[EvidenceEvent]:
        return (e for _, e in self.events())
