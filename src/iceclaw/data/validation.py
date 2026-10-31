"""Source validation that decides a region's readiness (uses the P1 adapter; no new decoding)."""

from __future__ import annotations

from dataclasses import dataclass, field

from iceclaw.contracts import Readiness
from iceclaw.data import moonatlas_adapter as ma


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationResult:
    checks: list[Check] = field(default_factory=list)
    missing_pixelwise_inputs: list[str] = field(default_factory=list)

    @property
    def source_ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def readiness(self) -> tuple[Readiness, str | None]:
        """verified needs a validated source AND pixelwise input layers; else partial/unusable."""
        if not self.source_ok:
            failed = "; ".join(f"{c.name}: {c.detail}" for c in self.checks if not c.passed)
            return Readiness.UNUSABLE, f"SOURCE_VALIDATION_FAILED: {failed}"
        if self.missing_pixelwise_inputs:
            return (
                Readiness.PARTIAL,
                "UNAVAILABLE_MODALITY: no pixelwise input layers ("
                + ", ".join(self.missing_pixelwise_inputs)
                + "); prediction-only audit tools are enabled",
            )
        return Readiness.VERIFIED, None


def with_verified_pairing(
    base: ValidationResult, alignment_checks: list[Check], exposed_modalities: list[str]
) -> ValidationResult:
    """Return a copy of ``base`` extended with a PASSED observation pairing.

    The named pixelwise modalities stop being "missing". Call only when every alignment check
    passed; a failed pairing must leave ``base`` untouched (the prediction stays valid alone).
    """
    return ValidationResult(
        checks=[
            *base.checks,
            *(Check(f"pairing_{c.name}", c.passed, c.detail) for c in alignment_checks),
        ],
        missing_pixelwise_inputs=[
            m for m in base.missing_pixelwise_inputs if m not in exposed_modalities
        ],
    )


def validate_sector(src: ma.SectorSource) -> ValidationResult:
    """Re-run checksum, decode, statistics, peak and geometry checks for one sector."""
    res = ValidationResult()

    def add(name: str, ok: bool, detail: str = "") -> None:
        res.checks.append(Check(name, ok, detail))

    try:
        pred = ma.decode_prediction(src)
    except ma.SourceError as e:
        add("decode_raw_prediction", False, str(e))
        return res
    add("decode_raw_prediction", True)
    summary = ma.summarize_prediction(pred)
    for item in ma.compare_to_upstream(src, summary).items:
        add(
            f"matches_upstream_{item.quantity}",
            item.ok,
            "" if item.ok else f"|diff|={item.abs_diff:.3g} > {item.tolerance:.3g}",
        )
    problems = ma.verify_peak_and_geometry(src, pred)
    add("peak_and_geometry", not problems, "; ".join(problems))
    add(
        "grid_matches_metadata",
        pred.values.shape == (src.height, src.width),
        f"{pred.values.shape} vs {(src.height, src.width)}",
    )
    res.missing_pixelwise_inputs = [
        m.name for m in ma.describe_modalities(src) if m.status == "unavailable"
    ]
    return res
