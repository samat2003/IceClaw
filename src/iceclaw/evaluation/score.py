"""Case-level scoring. Scoring needs FROZEN outputs; the sealed key/reference are loaded only here.

Statistical unit is the CASE (never a pixel). All metrics with an empty denominator are ``None``
(explicitly undefined), never 0 or 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iceclaw.evaluation.fixtures import (
    SealedKey,
    SuiteIntegrityError,
    SuiteManifest,
    canonical_json,
    load_manifest,
    load_scenario,
    load_sealed,
    load_sealed_reference,
    sha256_bytes,
)


# ------------------------------------------------------------------------------- freezing
@dataclass(frozen=True)
class FrozenDetections:
    """Immutable detector output bound to one frozen suite (by manifest hash)."""

    suite_manifest_sha256: str
    flags: tuple[tuple[str, bool], ...]  # sorted (case_id, flagged)
    sha256: str
    # P7: an explicitly DECLARED subset of detection cases (e.g. a one-case pilot). None = all.
    declared_subset: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, bool]:
        return dict(self.flags)


def freeze_detections(
    manifest: SuiteManifest, flags: dict[str, bool], subset: list[str] | None = None
) -> FrozenDetections:
    """Freeze detector output. With ``subset`` the output must cover exactly that declared list
    of the suite's detection cases (the subset is part of the hash); otherwise all of them."""
    detection = set(manifest.detection_case_ids)
    expected = detection if subset is None else set(subset)
    if not expected <= detection or (subset is not None and len(set(subset)) != len(subset)):
        raise ValueError(f"declared subset is not a set of detection cases: {subset}")
    if not expected:
        raise ValueError("declared subset is empty")
    if set(flags) != expected:
        raise ValueError(
            f"detections must cover exactly the declared detection cases; "
            f"missing={sorted(expected - set(flags))} extra={sorted(set(flags) - expected)}"
        )
    items = tuple(sorted((k, bool(v)) for k, v in flags.items()))
    if subset is None:  # unchanged hash body for full-suite outputs (P3/P4 compatibility)
        body = canonical_json({"suite": manifest.body_sha256(), "flags": items})
        return FrozenDetections(manifest.body_sha256(), items, sha256_bytes(body.encode()))
    declared = tuple(sorted(subset))
    body = canonical_json({"suite": manifest.body_sha256(), "flags": items, "subset": declared})
    return FrozenDetections(manifest.body_sha256(), items, sha256_bytes(body.encode()), declared)


def frozen_detections_sha256(frozen: FrozenDetections) -> str:
    """Recompute the binding hash of a frozen output (detects edits after freezing)."""
    body: dict[str, Any] = {"suite": frozen.suite_manifest_sha256, "flags": frozen.flags}
    if frozen.declared_subset is not None:
        body["subset"] = frozen.declared_subset
    return sha256_bytes(canonical_json(body).encode())


@dataclass(frozen=True)
class FrozenRanking:
    suite_manifest_sha256: str
    case_id: str
    window_ids: tuple[int, ...]
    sha256: str


def freeze_ranking(manifest: SuiteManifest, window_ids: list[int]) -> FrozenRanking:
    sec = manifest.secondary_task
    ids = [int(i) for i in window_ids]
    if len(ids) != sec["top_k"] or len(set(ids)) != len(ids):
        raise ValueError(f"ranking must list exactly {sec['top_k']} distinct window ids")
    if any(not 0 <= i < sec["n_windows"] for i in ids):
        raise ValueError("window id out of range")
    body = canonical_json({"suite": manifest.body_sha256(), "case": sec["case_id"], "ids": ids})
    return FrozenRanking(
        manifest.body_sha256(), sec["case_id"], tuple(ids), sha256_bytes(body.encode())
    )


# ------------------------------------------------------------------------------- metrics
def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def detection_metrics(flags: dict[str, bool], labels: dict[str, bool]) -> dict[str, Any]:
    """Case-level confusion counts and metrics; undefined values are ``None``."""
    if set(flags) != set(labels):
        raise ValueError("flags and labels must cover the same cases")
    tp = sum(1 for c, f in flags.items() if f and labels[c])
    fp = sum(1 for c, f in flags.items() if f and not labels[c])
    fn = sum(1 for c, f in flags.items() if not f and labels[c])
    tn = sum(1 for c, f in flags.items() if not f and not labels[c])
    return {
        "unit": "case",
        "n_cases": len(flags),
        "n_fault_cases": tp + fn,
        "n_clean_cases": tn + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "confidence_intervals": None,
        "ci_note": "not reported: sample too small and cases share one source tile",
    }


def score_detections(
    frozen: FrozenDetections, scenarios_root: Path, eval_root: Path, suite_id: str
) -> dict[str, Any]:
    """Reveal the sealed labels ONLY for an already-frozen detector output."""
    manifest = load_manifest(scenarios_root, suite_id)
    if frozen.suite_manifest_sha256 != manifest.body_sha256():
        raise SuiteIntegrityError("frozen detections belong to a different suite manifest")
    if frozen_detections_sha256(frozen) != frozen.sha256:
        raise SuiteIntegrityError("frozen detections were modified after freezing")
    sealed = load_sealed(eval_root, manifest)
    flags = frozen.as_dict()
    labels = sealed.labels
    if frozen.declared_subset is not None:
        if set(frozen.declared_subset) != set(flags):
            raise SuiteIntegrityError("frozen flags do not match their declared subset")
        labels = {c: labels[c] for c in frozen.declared_subset}
    result = detection_metrics(flags, labels)
    result["declared_subset"] = (
        None if frozen.declared_subset is None else list(frozen.declared_subset)
    )
    result["frozen_sha256"] = frozen.sha256
    result["per_case"] = _per_case(flags, sealed)
    result["expectation_check"] = _expectation_check(flags, sealed)
    return result


def _per_case(flags: dict[str, bool], sealed: SealedKey) -> list[dict[str, Any]]:
    out = []
    for cid in sorted(flags):
        t = sealed.cases[cid]
        out.append(
            {
                "case_id": cid,
                "fault_id": t["fault_id"],
                "is_fault": t["is_fault"],
                "flagged": flags[cid],
                "outcome": ("TP" if flags[cid] else "FN")
                if t["is_fault"]
                else ("FP" if flags[cid] else "TN"),
                "expected_detectable": t["expected_detectable"],
            }
        )
    return out


def _expectation_check(flags: dict[str, bool], sealed: SealedKey) -> dict[str, Any]:
    """Compare pre-registered detectability claims with what the baseline did."""
    mism = [
        sealed.cases[c]["fault_id"]
        for c in flags
        if sealed.cases[c]["is_fault"] and flags[c] != sealed.cases[c]["expected_detectable"]
    ]
    return {"mismatches_vs_preregistered": sorted(mism)}


# ------------------------------------------------------------------- secondary task (B)
def window_residuals(
    prediction: np.ndarray, reference: np.ndarray, windows: list[list[int]]
) -> np.ndarray:
    """Per-window mean |prediction - reference| over pixels where both are finite."""
    out = np.full(len(windows), np.nan)
    diff = np.abs(prediction - reference)
    for i, (r0, r1, c0, c1) in enumerate(windows):
        d = diff[r0:r1, c0:c1]
        d = d[np.isfinite(d)]
        if d.size:
            out[i] = d.mean()
    return out


def topk_recovery(residuals: np.ndarray, submitted: list[int], k: int) -> dict[str, Any]:
    """Top-K recovery vs the K truly highest-residual windows (ties by id).

    ``cumulative_residual_fraction`` = residual captured by the submission / best possible.
    ``random_expectation`` = K / N_windows.
    """
    if k < 1 or len(submitted) != k or len(set(submitted)) != k:
        raise ValueError("submission must contain exactly k distinct window ids")
    finite = np.where(np.isfinite(residuals), residuals, -np.inf)
    truth = sorted(range(len(residuals)), key=lambda i: (-finite[i], i))[:k]
    hits = len(set(truth) & set(submitted))
    best = float(np.sum(residuals[truth]))
    got = float(np.nansum(residuals[submitted]))
    return {
        "k": k,
        "n_windows": int(len(residuals)),
        "hits": hits,
        "top_k_recovery": hits / k,
        "random_expectation": k / len(residuals),
        "cumulative_residual_fraction": _ratio(got, best),
        "unit": "window",
        "note": "windows from one tile are spatially correlated; exploratory only",
    }


def score_ranking(
    frozen: FrozenRanking, scenarios_root: Path, eval_root: Path, suite_id: str
) -> dict[str, Any]:
    """Load the sealed reference ONLY now, for a frozen ranking, and score top-K recovery.

    Interpretation: agreement with expert-reference *disagreement*; not ice detection and the
    reference is not physical ground truth.
    """
    manifest = load_manifest(scenarios_root, suite_id)
    if frozen.suite_manifest_sha256 != manifest.body_sha256():
        raise SuiteIntegrityError("frozen ranking belongs to a different suite manifest")
    sealed = load_sealed(eval_root, manifest)
    ref = load_sealed_reference(eval_root, manifest, sealed)
    sec = manifest.secondary_task
    scenario = load_scenario(scenarios_root, suite_id, sec["case_id"])  # hash-verified
    pred = next(iter(scenario.arrays.layers.values())).values
    res = window_residuals(pred, ref, sec["windows"])
    out = topk_recovery(res, list(frozen.window_ids), sec["top_k"])
    out["frozen_sha256"] = frozen.sha256
    return out
