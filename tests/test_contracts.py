from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iceclaw.contracts import (
    AgentRegionContext,
    EvidenceEvent,
    InputManifest,
    InvestigationBundle,
    ModalityInfo,
    Readiness,
    RegionCatalogEntry,
    RunBudget,
    ToolMeasurement,
)

SHA = "a" * 64
BUDGET = RunBudget(max_events=10, max_hops=3, max_tool_calls=5, max_llm_calls=5, timeout_s=60)
NOW = datetime(2026, 10, 31, tzinfo=UTC)


def region(**over):
    base = dict(
        region_id="r1",
        pole="south",
        source_dataset="ds",
        source_split="heldout",
        original_tile_id="t1",
        source_mode="upstream_published_example",
        source_revision="rev",
        processing_version="pv",
        source_manifest_path="m.json",
        source_sha256=SHA,
        lunar_crs="CRS",
        affine=(240.0, 0, 0, 0, -240.0, 0),
        footprint=[(0, 0), (1, 0), (1, 1)],
        width=256,
        height=256,
        spatial_resolution_m=240.0,
        modalities={"temp": ModalityInfo(name="temp", unit="K")},
        prediction_raw_path="p.tif",
        prediction_sha256=SHA,
        prediction_checkpoint_id="ckpt",
        reference_path="/secret/reference.tif",
        readiness="verified",
    )
    base.update(over)
    return RegionCatalogEntry(**base)


def event(**over):
    base = dict(
        event_id="e1",
        investigation_id="i1",
        region_id="r1",
        agent_id="thermal",
        timestamp_utc=NOW,
        event_kind="claim",
        claim_or_request="x",
    )
    base.update(over)
    return EvidenceEvent(**base)


# ---- region
def test_region_valid():
    assert region().readiness is Readiness.VERIFIED


@pytest.mark.parametrize("field,val", [("width", 0), ("height", -1), ("spatial_resolution_m", 0)])
def test_region_nonpositive_dims(field, val):
    with pytest.raises(ValidationError):
        region(**{field: val})


def test_region_bad_sha_and_missing_provenance():
    with pytest.raises(ValidationError):
        region(source_sha256="xyz")
    with pytest.raises(ValidationError):
        region(source_revision="")
    with pytest.raises(ValidationError):
        region(footprint=[(0, 0)])


def test_region_partial_requires_reason():
    with pytest.raises(ValidationError):
        region(readiness="partial")
    assert region(readiness="partial", readiness_reason="no thermal").readiness == "partial"


def test_modality_key_mismatch():
    with pytest.raises(ValidationError):
        region(modalities={"a": ModalityInfo(name="b")})


# ---- firewall
def test_agent_context_has_no_reference_fields():
    ctx = region().to_agent_context(["raster_summary"], BUDGET)
    dumped = ctx.model_dump_json()
    for forbidden in ("reference", "/secret", "fault", "source_manifest_path", "sha256"):
        assert forbidden not in dumped
    assert ctx.available_modalities == ["temp"]


def test_agent_context_rejects_reference_field():
    ctx = region().to_agent_context([], BUDGET).model_dump()
    with pytest.raises(ValidationError):
        AgentRegionContext(**ctx, reference_path="x")


def test_unusable_region_not_exposed():
    r = region(readiness="unusable", readiness_reason="bad")
    with pytest.raises(ValueError):
        r.to_agent_context([], BUDGET)


def test_agent_context_field_allowlist_excludes_private():
    private = {"reference_path", "prediction_raw_path", "source_manifest_path", "affine"}
    assert not private & set(AgentRegionContext.model_fields)


# ---- events
def test_event_valid_and_kinds():
    assert event().event_kind == "claim"
    with pytest.raises(ValidationError):
        event(event_kind="nonsense")


def test_challenge_and_revision_need_parents():
    with pytest.raises(ValidationError):
        event(event_kind="challenge")
    with pytest.raises(ValidationError):
        event(event_kind="revision", claim_or_request=None)
    assert event(event_kind="challenge", parent_event_ids=["e0"]).parent_event_ids == ["e0"]
    assert event(event_kind="revision", parent_event_ids=["e2"], claim_or_request=None)


def test_measurement_needs_tool_run_and_failure_needs_code():
    with pytest.raises(ValidationError):
        event(event_kind="measurement", claim_or_request=None)
    with pytest.raises(ValidationError):
        event(event_kind="failure", claim_or_request=None)
    ok = event(
        event_kind="measurement",
        claim_or_request=None,
        tool_id="t",
        tool_run_id="tr1",
        metric_value=1.0,
        unit="K",
    )
    assert ok.tool_run_id == "tr1"


def test_event_misc_rules():
    with pytest.raises(ValidationError):
        event(timestamp_utc=datetime(2026, 1, 1))  # naive
    with pytest.raises(ValidationError):
        event(metric_value=1.0)  # no unit
    with pytest.raises(ValidationError):
        event(tool_run_id="x")  # no tool_id
    with pytest.raises(ValidationError):
        event(event_kind="challenge", parent_event_ids=["e1"])  # self parent
    with pytest.raises(ValidationError):
        event(unexpected=1)


# ---- measurement
def meas(**over):
    base = dict(
        tool_run_id="tr",
        tool_id="t",
        region_id="r1",
        source_file_ids=["f"],
        source_sha256={"f": SHA},
        value=1.5,
        tolerance=1e-9,
        implementation_version="0.1",
    )
    base.update(over)
    return ToolMeasurement(**base)


def test_measurement_value_xor_reason():
    assert meas().value == 1.5
    assert meas(value=None, unavailable_reason="UNAVAILABLE_MODALITY")
    with pytest.raises(ValidationError):
        meas(value=None)
    with pytest.raises(ValidationError):
        meas(unavailable_reason="OUT_OF_COVERAGE")
    with pytest.raises(ValidationError):
        meas(source_sha256={"f": "bad"})
    with pytest.raises(ValidationError):
        meas(source_file_ids=[])


# ---- bundle
def test_bundle_manifest_consistency():
    m = InputManifest(
        investigation_id="i1",
        region_id="r1",
        source_mode="upstream_published_example",
        source_dataset="d",
        source_split="s",
        source_revision="v",
        source_sha256=SHA,
        prediction_sha256=SHA,
        prediction_checkpoint_id="c",
        code_revision="abc",
        visible_modalities=[],
        budget=BUDGET,
    )
    assert InvestigationBundle(investigation_id="i1", root="runs/i1", manifest=m)
    with pytest.raises(ValidationError):
        InvestigationBundle(investigation_id="i2", root="runs/i2", manifest=m)
    with pytest.raises(ValidationError):
        InvestigationBundle(investigation_id="../x", root="r", manifest=m.model_copy())
