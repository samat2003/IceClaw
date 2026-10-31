"""Application-facing service functions shared by the CLI and (later) the UI.

Catalog listing (P2/P4) and the P6 ``InvestigationService``: run (live), export a recorded
run, verify offline, replay without any model call. The CLI and a future UI call the same methods.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iceclaw.config import Settings
from iceclaw.contracts import RunBudget
from iceclaw.data.catalog import RegionCatalog, discover


def load_catalog(settings: Settings) -> RegionCatalog:
    return discover(settings)


def describe_regions(catalog: RegionCatalog) -> list[dict[str, Any]]:
    """JSON-ready rows for operators. Includes private source paths; not an agent-facing view."""
    rows = []
    for rid in sorted(catalog.records):
        rec = catalog.records[rid]
        e = rec.entry
        lat, lon = rec.source.center_lat_lon
        rows.append(
            {
                "region_id": e.region_id,
                "status": rec.status,
                "prediction_ready": rec.prediction_ready,
                "observation_ready": rec.observation_ready,
                "pairing_error": rec.pairing_error,
                "pole": e.pole,
                "readiness": e.readiness.value,
                "readiness_reason": e.readiness_reason,
                "source_mode": e.source_mode.value,
                "source_dataset": e.source_dataset,
                "source_split": e.source_split,
                "original_tile_id": e.original_tile_id,
                "source_revision": e.source_revision,
                "processing_version": e.processing_version,
                "checkpoint": e.prediction_checkpoint_id,
                "prediction_sha256": e.prediction_sha256,
                "center_lat_lon": [lat, lon],
                "grid": [e.width, e.height],
                "spatial_resolution_m": e.spatial_resolution_m,
                "pixelwise_modalities": sorted(e.modalities),
                "modality_units": {k: v.unit for k, v in sorted(e.modalities.items())},
                "other_metadata": [
                    {"name": m.name, "kind": m.status, "value": m.value, "unit": m.unit}
                    for m in rec.availability
                    if m.status in ("tile_scalar", "single_anchor_point")
                ],
                "unavailable_modalities": [
                    m.name for m in rec.availability if m.status == "unavailable"
                ],
                "unit_unknown_layers_not_exposed": rec.unit_unknown_layers,
            }
        )
    return rows


def describe_observation_only(catalog: RegionCatalog) -> list[dict[str, Any]]:
    """Tiles with verified observations but NO prediction. Not selectable for investigation."""
    rows = []
    for sid in sorted(catalog.observation_only):
        o = catalog.observation_only[sid]
        rows.append(
            {
                "region_id": sid,
                "status": o.status,
                "prediction_ready": o.prediction_ready,
                "observation_ready": o.observation_ready,
                "reason": o.reason,
                "original_tile_id": o.patch.patch_id,
                "source_split": o.patch.split,
                "grid": [o.patch.grid["widthPx"], o.patch.grid["heightPx"]],
                "layer_finite_fraction": {
                    k: round(v.finite_fraction, 4)
                    for k, v in sorted(o.patch.layers.items())
                    if v.finite_fraction < 1.0
                },
            }
        )
    return rows


# ================================================================================ P6 service
DEFAULT_BUDGET = RunBudget(
    max_events=40, max_hops=5, max_tool_calls=16, max_llm_calls=28, timeout_s=420.0
)
DEFAULT_MAX_INPUT_TOKENS = 150_000  # runaway guard; observed live runs used 71k-113k
ClientFactory = Callable[[Any, str], Any]


@dataclass
class InvestigationResult:
    run_id: str
    run_dir: Path
    summary: dict[str, Any]
    verification: Any  # reports.verify.VerificationReport

    @property
    def ok(self) -> bool:
        return bool(self.verification.ok) and not self.summary.get("problems")


@dataclass
class ReplayResult:
    run_id: str
    run_dir: Path
    header: str
    lines: list[str]
    n_events: int
    model_calls_made: int = 0  # replay never calls a model; kept explicit and asserted in tests


class InvestigationService:
    """Single entry point for investigations. ``client_factory`` injects a fake model for tests;
    when it is None the real OpenAI client from IceClaw settings is used (live mode)."""

    def __init__(
        self,
        settings: Settings,
        *,
        client_factory: ClientFactory | None = None,
        provider_label: str | None = None,
        model_label: str | None = None,
        spec_provider: Callable[[str], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.spec_provider = spec_provider  # alternative source (tests); default: real catalog
        self.client_factory = client_factory
        self.provider_label = provider_label
        self.model_label = model_label

    # -------------------------------------------------------------------------- helpers
    def resolve_run(self, run: str | Path) -> Path:
        p = Path(run)
        if p.is_dir():
            return p
        q = self.settings.runs_root / str(run)
        if q.is_dir():
            return q
        raise FileNotFoundError(f"run {run!r} not found (looked in {p} and {q})")

    def _new_run_dir(self, run_id: str | None, suffix: str) -> tuple[str, Path]:
        base = run_id or f"inv-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{suffix}"
        rid, n = base, 1
        while (self.settings.runs_root / rid).exists():
            n += 1
            rid = f"{base}-{n}"
        d = self.settings.runs_root / rid
        d.mkdir(parents=True)
        return rid, d

    def _paired_spec(self, region_id: str):
        if self.spec_provider is not None:
            return self.spec_provider(region_id)
        from iceclaw.reports.package import spec_from_catalog_record
        from iceclaw.swarm.runtime import RegionNotInvestigable

        rec = discover(self.settings).get(region_id)  # observation-only / unknown raise here
        if rec.status != "PAIRED" or rec.entry.readiness.value != "verified":
            raise RegionNotInvestigable(
                f"{region_id} is {rec.status} (readiness {rec.entry.readiness.value}); "
                "only verified PAIRED regions are investigated"
            )
        return spec_from_catalog_record(rec)

    # ----------------------------------------------------------------------------- live
    def investigate(
        self,
        region_id: str,
        *,
        budget: RunBudget | None = None,
        max_rounds_per_agent: int = 3,
        max_input_tokens: int | None = DEFAULT_MAX_INPUT_TOKENS,
        run_id: str | None = None,
    ) -> InvestigationResult:
        """Run the four specialists on a verified paired region and export a verified bundle.

        The swarm works on arrays loaded FROM the bundle's own source package, so the offline
        verifier recomputes from exactly the data the agents saw.
        """
        from iceclaw.agents.provider import build_openai_client
        from iceclaw.agents.specialists import build_collective, build_prompt, build_specs
        from iceclaw.reports import export as ex
        from iceclaw.reports import package as pkg
        from iceclaw.swarm.runtime import RunLimits, SwarmRuntime
        from iceclaw.swarm.store import EvidenceStore

        scripted = self.client_factory is not None
        if not scripted:
            self.settings.require_live_llm()  # clear MISSING_API_KEY / MISSING_LLM_MODEL
        budget = budget or DEFAULT_BUDGET
        spec = self._paired_spec(region_id)
        rid, run_dir = self._new_run_dir(run_id, "p6")
        pkg.write_package(spec, run_dir / "source")
        loaded = pkg.load_package(run_dir / "source")
        context = pkg.context_from_spec(spec, loaded.arrays, budget)
        provider = self.provider_label or ("scripted" if scripted else self.settings.llm_provider)
        model = self.model_label or ("scripted-v0" if scripted else (self.settings.llm_model or ""))
        limits = RunLimits(
            budget=budget,
            max_rounds_per_agent=max_rounds_per_agent,
            max_input_tokens=max_input_tokens,
        )
        (run_dir / "internal").mkdir()
        store = EvidenceStore(run_dir / "internal" / "events.sqlite")
        rt = SwarmRuntime(
            investigation_id=rid,
            context=context,
            arrays=loaded.arrays,
            store=store,
            specs=build_specs(),
            limits=limits,
            provider=provider,
            model=model,
        )
        factory = self.client_factory or (
            lambda runtime, aid: build_openai_client(self.settings, runtime, aid, limits)
        )
        summary = asyncio.run(rt.run(build_collective(rt, factory), build_prompt))
        sdict = {**summary.__dict__, "provider": provider, "model": model}
        (run_dir / "internal" / "run_summary.json").write_text(
            json.dumps(sdict, indent=2, default=str), encoding="utf-8"
        )
        record = ex.record_from_store(
            store,
            summary=sdict,
            mode="scripted-test" if scripted else "live",
            provider=provider,
            model=model,
            code_revision=ex.current_code_revision(),
        )
        result = ex.export_bundle(record, run_dir)
        store.close()
        return InvestigationResult(rid, run_dir, sdict, result.verification)

    # ------------------------------------------------------------------ recorded -> bundle
    def export_recorded(
        self, source_run: str | Path, *, out_id: str | None = None
    ) -> InvestigationResult:
        """Export a RECORDED run (e.g. the P5 live run) into a bundle. No model call; the source
        run directory is only read (SQLite opened read-only) and never modified."""
        from iceclaw.reports import export as ex

        src = self.resolve_run(source_run)
        rec = ex.record_from_run_dir(
            src,
            notes=[
                f"Exported from the recorded run directory '{src.name}' without any new model "
                "call. The run predates code-revision recording, so code_revision is the "
                "revision at EXPORT time, not at run time.",
                "That run used the pre-P6 unbounded prompt digest; its prompts are retained "
                "unchanged in llm_calls.jsonl.",
            ],
        )
        spec = self._paired_spec(rec.region_id)
        rid, out_dir = self._new_run_dir(out_id or f"{src.name}-p6bundle", "export")
        rec.investigation_id = rid
        # event/measurement rows keep their original ids; only the bundle id differs
        rec.summary = {**rec.summary, "investigation_id": rid}
        result = ex.export_bundle(rec, out_dir, spec)
        return InvestigationResult(rid, out_dir, rec.summary, result.verification)

    # ----------------------------------------------------------------------------- publish
    def publish(self, run: str | Path, *, out_id: str | None = None) -> InvestigationResult:
        """Create a PUBLIC-REDACTED bundle (no opaque provider reasoning data, no internal
        history) from a verified bundle. No model call; the source bundle is only read."""
        import shutil

        from iceclaw.reports import export as ex

        src = self.resolve_run(run)
        rid, out_dir = self._new_run_dir(out_id or f"{src.name}-public", "public")
        try:
            result = ex.publish_bundle(src, out_dir)
        except Exception:
            shutil.rmtree(out_dir, ignore_errors=True)  # never leave a half-made public bundle
            raise
        return InvestigationResult(
            rid, out_dir, {"public_of": src.name, "problems": []}, result.verification
        )

    # ------------------------------------------------------------------------------ verify
    def verify(
        self,
        run: str | Path,
        *,
        source_package: str | Path | None = None,
        local_source: bool = False,
    ):
        from iceclaw.reports.verify import verify_bundle

        return verify_bundle(
            self.resolve_run(run),
            source_package=source_package,
            local_settings=self.settings if local_source else None,
        )

    # ------------------------------------------------------------------------------ replay
    def replay(self, run: str | Path) -> ReplayResult:
        """Display a saved run from its events only. Makes NO model call (and builds no client)."""
        from iceclaw.contracts import EvidenceEvent
        from iceclaw.reports import export as ex

        d = self.resolve_run(run)
        manifest: dict[str, Any] = {}
        if (d / "events.jsonl").is_file() and (d / "input_manifest.json").is_file():
            events = [
                EvidenceEvent.model_validate_json(x)
                for x in (d / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            manifest = json.loads((d / "input_manifest.json").read_text(encoding="utf-8"))
        else:
            events = ex.record_from_run_dir(d).events
        if not events:
            raise ValueError(f"run {d.name} contains no events")
        t0 = events[0].timestamp_utc
        lines = []
        for e in events:
            dt = (e.timestamp_utc - t0).total_seconds()
            parents = f" (parents: {', '.join(e.parent_event_ids)})" if e.parent_event_ids else ""
            text = " ".join((e.claim_or_request or "").split())
            lines.append(
                f"+{dt:7.2f}s [{e.event_id}] {e.agent_id:<18} "
                f"{e.event_kind.value:<12}{parents} {text}"
            )
        mu = manifest.get("model_usage", {})
        header = (
            f"RECORDED RUN {d.name}: replayed from saved events only; no model calls are made "
            f"(recorded model {manifest.get('llm_provider', '?')}/"
            f"{manifest.get('llm_model', '?')}, "
            f"{mu.get('calls', '?')} calls, {mu.get('total_tokens', '?')} tokens)"
        )
        return ReplayResult(d.name, d, header, lines, len(events))
