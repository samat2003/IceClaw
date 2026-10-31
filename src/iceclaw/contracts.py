"""Frozen data contracts (PROJECT_SPEC §5, IMPLEMENTATION_PLAN §2).

``RegionCatalogEntry`` is private (holds the evaluator-only reference path).
Agents only ever see ``AgentRegionContext``, built by explicit allowlist.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_sha256(v: str) -> str:
    if not _SHA256_RE.match(v):
        raise ValueError("must be a lowercase 64-hex SHA256")
    return v


# --------------------------------------------------------------------------- enums
class Readiness(StrEnum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    UNUSABLE = "unusable"


class SourceMode(StrEnum):
    UPSTREAM_PUBLISHED_EXAMPLE = "upstream_published_example"
    LOCAL_CHECKPOINT_RUN = "local_checkpoint_run"
    # P4 (additive): scenario holds physical observations only, no model prediction at all.
    OBSERVATION_ONLY = "observation_only"


class EventKind(StrEnum):
    CLAIM = "claim"
    MEASUREMENT = "measurement"
    CHALLENGE = "challenge"
    TEST_REQUEST = "test_request"
    TEST_RESULT = "test_result"
    REVISION = "revision"
    COMPLETION = "completion"
    FAILURE = "failure"


class EvidenceType(StrEnum):
    OBSERVATION = "observation"
    MODEL_PREDICTION = "model_prediction"
    COMPUTED_METRIC = "computed_metric"
    HYPOTHESIS = "hypothesis"


class HypothesisStatus(StrEnum):
    PROVISIONAL = "provisional"
    CHALLENGED = "challenged"
    SUPPORTED = "supported"
    UNRESOLVED = "unresolved"


class MeasurementFailure(StrEnum):
    UNAVAILABLE_MODALITY = "UNAVAILABLE_MODALITY"
    UNVERIFIED_SOURCE = "UNVERIFIED_SOURCE"
    OUT_OF_COVERAGE = "OUT_OF_COVERAGE"


# --------------------------------------------------------------------- region models
class ModalityInfo(_Strict):
    """One scientific layer actually present for a region."""

    name: str = Field(min_length=1)
    unit: str | None = None
    nodata: float | None = None
    resolution_m: float | None = Field(default=None, gt=0)
    path: str | None = None
    sha256: str | None = None

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str | None) -> str | None:
        return v if v is None else _check_sha256(v)


class RegionCatalogEntry(_Strict):
    """Private full catalog record. Never pass to agents."""

    region_id: str = Field(min_length=1)
    pole: Literal["south", "north"]
    source_dataset: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    original_tile_id: str = Field(min_length=1)
    source_mode: SourceMode
    source_revision: str = Field(min_length=1)
    processing_version: str = Field(min_length=1)
    source_manifest_path: str = Field(min_length=1)
    source_sha256: str
    lunar_crs: str = Field(min_length=1)
    affine: tuple[float, float, float, float, float, float]
    footprint: list[tuple[float, float]] = Field(min_length=3)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    spatial_resolution_m: float = Field(gt=0)
    modalities: dict[str, ModalityInfo] = Field(default_factory=dict)
    prediction_raw_path: str = Field(min_length=1)
    prediction_sha256: str
    prediction_checkpoint_id: str = Field(min_length=1)
    reference_path: str | None = None  # EVALUATION-ONLY
    readiness: Readiness
    readiness_reason: str | None = None

    @field_validator("source_sha256", "prediction_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _check_sha256(v)

    @model_validator(mode="after")
    def _check(self) -> RegionCatalogEntry:
        for key, m in self.modalities.items():
            if key != m.name:
                raise ValueError(f"modality key {key!r} != name {m.name!r}")
        if self.readiness is not Readiness.VERIFIED and not self.readiness_reason:
            raise ValueError("non-verified readiness requires readiness_reason")
        return self

    def to_agent_context(self, tool_names: list[str], budget: RunBudget) -> AgentRegionContext:
        """Explicit allowlist projection; no reference fields are copied."""
        if self.readiness is Readiness.UNUSABLE:
            raise ValueError("unusable region cannot be exposed to agents")
        return AgentRegionContext(
            region_id=self.region_id,
            pole=self.pole,
            footprint=self.footprint,
            width=self.width,
            height=self.height,
            spatial_resolution_m=self.spatial_resolution_m,
            lunar_crs=self.lunar_crs,
            source_ids=[self.source_dataset, self.original_tile_id, self.prediction_checkpoint_id],
            source_mode=self.source_mode,
            available_modalities=sorted(self.modalities),
            available_tools=sorted(tool_names),
            budget=budget,
        )


class RunBudget(_Strict):
    max_events: int = Field(gt=0)
    max_hops: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_llm_calls: int = Field(gt=0)
    timeout_s: float = Field(gt=0)


class AgentRegionContext(_Strict):
    """Restricted public view. ``extra=forbid`` rejects any reference/fault field."""

    region_id: str = Field(min_length=1)
    pole: Literal["south", "north"]
    footprint: list[tuple[float, float]]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    spatial_resolution_m: float = Field(gt=0)
    lunar_crs: str
    source_ids: list[str]
    source_mode: SourceMode
    available_modalities: list[str]
    available_tools: list[str]
    budget: RunBudget


# ------------------------------------------------------------------ tool measurement
class ToolMeasurement(_Strict):
    tool_run_id: str = Field(min_length=1)
    tool_id: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    region_id: str = Field(min_length=1)
    footprint: list[tuple[float, float]] | None = None
    source_file_ids: list[str] = Field(min_length=1)
    source_sha256: dict[str, str] = Field(min_length=1)
    value: float | list[float] | dict[str, float] | None = None
    unavailable_reason: MeasurementFailure | None = None
    unit: str | None = None
    valid_pixel_count: int | None = Field(default=None, ge=0)
    resolution_m: float | None = Field(default=None, gt=0)
    tolerance: float = Field(ge=0)
    implementation_version: str = Field(min_length=1)

    @field_validator("source_sha256")
    @classmethod
    def _shas(cls, v: dict[str, str]) -> dict[str, str]:
        for h in v.values():
            _check_sha256(h)
        return v

    @model_validator(mode="after")
    def _value_xor_reason(self) -> ToolMeasurement:
        if (self.value is None) == (self.unavailable_reason is None):
            raise ValueError("exactly one of value / unavailable_reason must be set")
        return self


# -------------------------------------------------------------------- evidence event
class EvidenceEvent(_Strict):
    event_id: str = Field(min_length=1)
    investigation_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    timestamp_utc: datetime
    event_kind: EventKind
    parent_event_ids: list[str] = Field(default_factory=list)
    claim_or_request: str | None = None
    evidence_type: EvidenceType | None = None
    source_artifact_ids: list[str] = Field(default_factory=list)
    tool_id: str | None = None
    tool_run_id: str | None = None
    spatial_footprint: list[tuple[float, float]] | None = None
    metric_value: float | None = None
    unit: str | None = None
    analysis_resolution_m: float | None = Field(default=None, gt=0)
    hypothesis_status: HypothesisStatus | None = None
    alternative_explanations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    error_code: str | None = None

    @field_validator("timestamp_utc")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("timestamp_utc must be timezone-aware")
        return v.astimezone(UTC)

    @model_validator(mode="after")
    def _kind_rules(self) -> EvidenceEvent:
        k = self.event_kind
        if self.event_id in self.parent_event_ids:
            raise ValueError("event cannot be its own parent")
        if k in (EventKind.CHALLENGE, EventKind.REVISION) and not self.parent_event_ids:
            raise ValueError(f"{k.value} must reference at least one parent event")
        if k in (EventKind.MEASUREMENT, EventKind.TEST_RESULT) and not self.tool_run_id:
            raise ValueError(f"{k.value} must cite a tool_run_id")
        if self.tool_run_id and not self.tool_id:
            raise ValueError("tool_run_id requires tool_id")
        if k is EventKind.FAILURE and not self.error_code:
            raise ValueError("failure requires error_code")
        if k in (EventKind.CLAIM, EventKind.CHALLENGE, EventKind.TEST_REQUEST) and not (
            self.claim_or_request
        ):
            raise ValueError(f"{k.value} requires claim_or_request")
        if self.metric_value is not None and self.unit is None:
            raise ValueError("metric_value requires unit")
        return self


# ------------------------------------------------------------------------- bundle
BUNDLE_REQUIRED_FILES = (
    "input_manifest.json",
    "events.jsonl",
    "evidence.json",
    "report.md",
    "rankings.csv",
    "verification.json",
)


class InputManifest(_Strict):
    investigation_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    source_mode: SourceMode
    source_dataset: str
    source_split: str
    source_revision: str
    source_sha256: str
    prediction_sha256: str
    prediction_checkpoint_id: str
    llm_provider: str | None = None
    llm_model: str | None = None  # never credentials
    code_revision: str = Field(min_length=1)
    visible_modalities: list[str]
    budget: RunBudget
    tools_executed: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @field_validator("source_sha256", "prediction_sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        return _check_sha256(v)


class InvestigationBundle(_Strict):
    """Location + manifest of ``runs/<investigation_id>/``."""

    investigation_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    root: str = Field(min_length=1)
    manifest: InputManifest

    @model_validator(mode="after")
    def _ids(self) -> InvestigationBundle:
        if self.manifest.investigation_id != self.investigation_id:
            raise ValueError("manifest investigation_id mismatch")
        return self
