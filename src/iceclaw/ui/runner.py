"""Deliberate, single-flight live investigations for the workbench.

* A live run starts only through ``LiveRunManager.start``: an explicit call that the page makes
  from a button click, never from a plain Streamlit rerun or page refresh.
* ``start`` is atomic: while a run is in flight (in this process, or recorded by a live process on
  this machine) every further call is refused, so reruns and refreshes cannot trigger duplicate
  paid investigations.
* Configuration is checked BEFORE a run directory is created or any client is built.
* Progress is read from the run's own SQLite evidence history (read-only): real events only.
* When a live run finishes, its verified private bundle is published to a public-redacted bundle
  (no model call); only the public bundle is offered by the UI.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iceclaw.config import ConfigError, Settings, load_dotenv_file, load_settings
from iceclaw.services import InvestigationService
from iceclaw.ui.views import TimelineRow, build_timeline

IDLE, RUNNING, SUCCEEDED, FAILED = "idle", "running", "succeeded", "failed"
MARKER = ".ui_live_run.json"
_KEYLIKE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def scrub(text: str) -> str:
    """Never surface anything key-like from an exception message."""
    return _KEYLIKE.sub("sk-[redacted]", str(text))[:500]


def app_settings() -> Settings:
    """Runtime settings exactly as the CLI builds them (the .env loader never prints values)."""
    load_dotenv_file()
    return load_settings()


def live_config_status(settings: Settings) -> dict[str, Any]:
    """Whether a live run can start. Reports presence only, never any secret value."""
    try:
        settings.require_live_llm()
    except ConfigError as e:
        return {
            "ok": False,
            "code": e.code,
            "message": scrub(str(e).split(": ", 1)[-1]),  # the code is shown separately
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }
    return {
        "ok": True,
        "code": "",
        "message": "",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@dataclass
class JobSnapshot:
    state: str = IDLE
    run_id: str = ""
    region_id: str = ""
    started_utc: str = ""
    finished_utc: str = ""
    error_code: str = ""
    error_message: str = ""
    public_run_id: str = ""
    ended_reason: str = ""
    problems: list[str] = field(default_factory=list)


@dataclass
class StartResult:
    started: bool
    code: str  # "STARTED" | "ALREADY_RUNNING" | "REGION_NOT_ALLOWED" | a ConfigError code
    message: str = ""


@dataclass
class Progress:
    events: list[TimelineRow]
    model_calls: int
    total_tokens: int
    ok: bool = True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


class LiveRunManager:
    """One manager per runs directory (see ``get_manager``)."""

    def __init__(
        self,
        runs_root: Path,
        service_factory: Callable[[], InvestigationService],
        *,
        allowed_regions: Callable[[], set[str]] | None = None,
    ) -> None:
        self.runs_root = Path(runs_root)
        self._factory = service_factory
        self._allowed = allowed_regions
        self._lock = threading.Lock()
        self._snap = JobSnapshot()
        self._thread: threading.Thread | None = None
        self.starts_attempted = 0  # for tests: how many real starts were accepted

    # ------------------------------------------------------------------------------ state
    def snapshot(self) -> JobSnapshot:
        with self._lock:
            s = self._snap
            return JobSnapshot(**{**s.__dict__, "problems": list(s.problems)})

    def _marker(self) -> Path:
        return self.runs_root / MARKER

    def _foreign_run(self) -> dict[str, Any] | None:
        """A live run recorded by ANOTHER live process (e.g. a second Streamlit server)."""
        p = self._marker()
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        pid = int(d.get("pid", -1))
        if pid == os.getpid() or not _pid_alive(pid):
            return None
        return dict(d)

    def config_status(self) -> dict[str, Any]:
        """Can this manager's service start a real run? (presence flags only, no secrets)"""
        service = self._factory()
        if service.client_factory is not None:  # injected (test) client: nothing to configure
            return {"ok": True, "code": "", "message": "", "provider": "injected", "model": None}
        return live_config_status(service.settings)

    # -------------------------------------------------------------------------------- start
    def start(self, region_id: str) -> StartResult:
        """Begin one live investigation, or refuse. Never blocks on the run itself."""
        with self._lock:
            if self._snap.state == RUNNING or (self._thread and self._thread.is_alive()):
                return StartResult(False, "ALREADY_RUNNING", f"{self._snap.run_id} is in progress")
            other = self._foreign_run()
            if other:
                return StartResult(
                    False, "ALREADY_RUNNING", f"{other.get('run_id')} (other process)"
                )
            if self._allowed is not None and region_id not in self._allowed():
                return StartResult(
                    False,
                    "REGION_NOT_ALLOWED",
                    f"{region_id} has no verified NASA–IBM prediction paired with observations",
                )
            service = self._factory()
            if service.client_factory is None:  # real OpenAI client: configuration must be present
                status = live_config_status(service.settings)
                if not status["ok"]:
                    return StartResult(False, status["code"], status["message"])
            run_id = f"inv-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-ui"
            n = 1
            while (self.runs_root / run_id).exists():
                n += 1
                run_id = f"inv-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-ui{n}"
            self._snap = JobSnapshot(
                RUNNING, run_id, region_id, started_utc=datetime.now(UTC).isoformat()
            )
            self.runs_root.mkdir(parents=True, exist_ok=True)
            self._marker().write_text(
                json.dumps({"pid": os.getpid(), "run_id": run_id, "region_id": region_id}),
                encoding="utf-8",
            )
            self.starts_attempted += 1
            self._thread = threading.Thread(
                target=self._work, args=(service, region_id, run_id), daemon=True, name=run_id
            )
            self._thread.start()
            return StartResult(True, "STARTED", run_id)

    def _work(self, service: InvestigationService, region_id: str, run_id: str) -> None:
        err_code = err_msg = ""
        ended = ""
        problems: list[str] = []
        public_id = ""
        try:
            res = service.investigate(region_id, run_id=run_id)
            ended = str(res.summary.get("ended_reason", ""))
            problems = list(res.summary.get("problems") or [])
            if not res.verification.ok:
                err_code = "PRIVATE_BUNDLE_NOT_VERIFIED"
                err_msg = ", ".join(c.id for c in res.verification.failures)
            else:
                pub = service.publish(res.run_dir, out_id=f"{run_id}-public")
                public_id = pub.run_id
                if not pub.verification.ok:
                    err_code = "PUBLIC_BUNDLE_NOT_VERIFIED"
                    err_msg = ", ".join(c.id for c in pub.verification.failures)
        except BaseException as e:  # noqa: BLE001 - surface any failure to the page, never crash
            err_code = str(getattr(e, "code", type(e).__name__))
            err_msg = scrub(f"{type(e).__name__}: {e}")
        finally:
            try:
                self._marker().unlink(missing_ok=True)
            except OSError:
                pass
            with self._lock:
                s = self._snap
                s.finished_utc = datetime.now(UTC).isoformat()
                s.ended_reason, s.problems, s.public_run_id = ended, problems, public_id
                s.error_code, s.error_message = err_code, err_msg
                s.state = FAILED if err_code else SUCCEEDED

    def wait(self, timeout: float | None = None) -> None:
        """Join the worker (tests / scripts)."""
        t = self._thread
        if t is not None:
            t.join(timeout)

    def reset(self) -> None:
        """Forget a finished/failed job so the page can offer a new one. Refused while running."""
        with self._lock:
            if self._snap.state != RUNNING:
                self._snap = JobSnapshot()

    # ----------------------------------------------------------------------------- progress
    def progress(self, run_id: str | None = None) -> Progress:
        """Real events and token use read from the run's SQLite history (read-only)."""
        from iceclaw.swarm.store import EvidenceStore

        rid = run_id or self.snapshot().run_id
        db = self.runs_root / rid / "internal" / "events.sqlite"
        if not rid or not db.is_file():
            return Progress([], 0, 0)
        try:
            store = EvidenceStore.open_readonly(db)
            try:
                events = [e.model_dump(mode="json") for _, e in store.events()]
                calls = store.llm_calls()
            finally:
                store.close()
        except sqlite3.Error:  # writer mid-commit: show the next tick
            return Progress([], 0, 0, ok=False)
        total = sum(int(c.get("total_tokens") or 0) for c in calls)
        return Progress(build_timeline(events), len(calls), total)


_MANAGERS: dict[str, LiveRunManager] = {}
_REG_LOCK = threading.Lock()


def get_manager(
    runs_root: Path,
    service_factory: Callable[[], InvestigationService],
    *,
    allowed_regions: Callable[[], set[str]] | None = None,
) -> LiveRunManager:
    """Process-wide manager per runs directory, so reruns/refreshes share one in-flight guard."""
    key = str(Path(runs_root).resolve())
    with _REG_LOCK:
        mgr = _MANAGERS.get(key)
        if mgr is None:
            mgr = _MANAGERS[key] = LiveRunManager(
                runs_root, service_factory, allowed_regions=allowed_regions
            )
        return mgr


def forget_managers() -> None:
    """Tests only."""
    with _REG_LOCK:
        _MANAGERS.clear()
