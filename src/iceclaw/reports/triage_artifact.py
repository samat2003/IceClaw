"""Versioned, independently verifiable research-triage artifact (``iceclaw-triage/1``).

A SEPARATE derived artifact: it never modifies or extends an ``iceclaw-bundle`` (the bundle
contract and every existing bundle are untouched). It is computed from a bundle's embedded,
hash-verified source package and is a post-processing result, not an action of any agent that
produced the bundle. ``verify_artifact`` recomputes it from the source package offline (no key, no
model) and compares exactly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

from iceclaw.reports import package as pkg
from iceclaw.reports.verify import VerificationReport
from iceclaw.science import triage as tg

ARTIFACT_FORMAT = "iceclaw-triage/1"
ARTIFACT_FILE = "triage.json"
CSV_FILE = "triage_windows.csv"
LABEL = (
    "POST-PROCESSING: computed now from the bundle's source package by deterministic code; not an "
    "action of the agents that produced the bundle. A research-triage screen, not an ice "
    "detection, abundance, calibrated likelihood or landing-site assessment."
)
_ABS_PATH = re.compile(r"(/home/|/mnt/|[A-Za-z]:\\\\)")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def build_artifact(
    source_dir: Path, bundle_name: str, cfg: tg.TriageConfig | None = None
) -> dict[str, Any]:
    """Load the (hash-verified) source package and run the deterministic screen."""
    loaded = pkg.load_package(Path(source_dir))  # raises PackageError on any tampering
    m = loaded.manifest
    result = tg.screen_arrays(loaded.arrays, cfg)
    return {
        "format": ARTIFACT_FORMAT,
        "label": LABEL,
        "source": {
            "bundle": bundle_name,
            "package_sha256": m["package_sha256"],
            "region_id": m["region"]["region_id"],
            "source_mode": m["region"]["source_mode"],
            "files": {
                f["modality"]: {"file_id": f["file_id"], "sha256": f["sha256"], "unit": f["unit"]}
                for f in m["files"]
            },
            "geometry": {
                "lunar_crs": m["geometry"]["lunar_crs"],
                "resolution_m": m["geometry"]["resolution_m"],
                "width": m["geometry"]["width"],
                "height": m["geometry"]["height"],
            },
        },
        "triage": result,
        "result_sha256": _sha(result),
    }


def write_artifact(artifact: dict[str, Any], out_dir: Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ARTIFACT_FILE).write_text(json.dumps(artifact, indent=1, sort_keys=True), "utf-8")
    (out / CSV_FILE).write_text(windows_csv(artifact), encoding="utf-8")
    return out / ARTIFACT_FILE


def load_artifact(path: Path) -> dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / ARTIFACT_FILE
    return json.loads(p.read_text(encoding="utf-8"))


def windows_csv(artifact: dict[str, Any]) -> str:
    t = artifact["triage"]
    sel = t["selection"]
    cat = {i: "followup" for i in sel["followup_ids"]} | {i: "review" for i in sel["review_ids"]}
    cols = [
        "label", "category", "rank", "row0", "row1", "col0", "col1", "centre_lat", "centre_lon",
        "coverage", "index_mean", "index_max", "shadow_fraction", "tmax_min_k", "tmax_median_k",
        "slope_median_deg", "slope_max_deg", "n_high_index", "n_joint", "review_reasons",
    ]  # fmt: skip
    rank = {i: k + 1 for k, i in enumerate(sel["followup_ids"])} | {
        i: k + 1 for k, i in enumerate(sel["review_ids"])
    }
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for x in t["windows"]:
        r0, r1, c0, c1 = x["window"]
        w.writerow(
            [x["label"], cat.get(x["id"], "none"), rank.get(x["id"], ""), r0, r1, c0, c1,
             x["centre_lat"], x["centre_lon"], x["coverage"], x["prediction_index"]["mean"],
             x["prediction_index"]["max"], x["shadow_fraction"], x["tmax_k"]["min"],
             x["tmax_k"]["median"], x["slope_deg"]["median"], x["slope_deg"]["max"],
             x["n_high_index"], x["n_joint"], ";".join(sel["review_reasons"].get(str(x["id"]), []))]
        )  # fmt: skip
    return buf.getvalue()


def verify_artifact(artifact: dict[str, Any], source_dir: Path) -> VerificationReport:
    """Offline verification: source integrity, identity, config hash, exact recomputation."""
    rep = VerificationReport(bundle=str(artifact.get("source", {}).get("bundle", "?")))
    rep.mode = "recomputed from the bundle's embedded source package"
    if not rep.add(
        "format",
        artifact.get("format") == ARTIFACT_FORMAT,
        f"{artifact.get('format')!r} (supported: {ARTIFACT_FORMAT})",
    ):
        return rep
    try:
        t = artifact["triage"]
        cfg = tg.TriageConfig(**t["config"])
    except (KeyError, TypeError) as e:
        rep.add("structure", False, f"malformed artifact ({type(e).__name__})")
        return rep
    rep.add("config_hash", cfg.sha256() == t.get("config_sha256"), "TriageConfig SHA256")
    rep.add(
        "result_hash",
        _sha(t) == artifact.get("result_sha256"),
        "result SHA256 over the triage block",
    )
    try:
        loaded = pkg.load_package(Path(source_dir))
    except pkg.PackageError as e:
        rep.add("source_package_integrity", False, f"{e.code}: source package failed: {e}")
        return rep
    rep.add("source_package_integrity", True, "every source file matches its recorded hash")
    m = loaded.manifest
    src = artifact["source"]
    same_files = all(
        src["files"].get(f["modality"], {}).get("sha256") == f["sha256"] for f in m["files"]
    ) and len(src["files"]) == len(m["files"])
    rep.add(
        "source_identity",
        src["package_sha256"] == m["package_sha256"] and same_files,
        "artifact names exactly this source package and its file hashes",
    )
    try:
        again = tg.screen_arrays(loaded.arrays, cfg)
        equal = canonical(again) == canonical(t)
        detail = "all windows, selections and sensitivity rows reproduced exactly"
    except tg.TriageError as e:
        equal, detail = False, f"recomputation failed: {e}"
    rep.add(
        "recomputation", equal, detail if equal else detail or "result differs from recomputation"
    )
    sel = t["selection"]
    ids = {x["id"] for x in t["windows"]}
    consistent = (
        t["counts"]["n_followup_windows"] == len(sel["followup_ids"])
        and t["counts"]["n_review_windows"] == len(sel["review_ids"])
        and set(sel["followup_ids"]) | set(sel["review_ids"]) <= ids
        and not set(sel["followup_ids"]) & set(sel["review_ids"])
    )
    rep.add("selection_consistent", consistent, "counts match lists; follow-up and review disjoint")
    text = canonical(artifact)
    rep.add(
        "no_forbidden_content",
        not _ABS_PATH.search(text) and not re.search(r"\bPRO\b|sk-[A-Za-z0-9_-]{16,}", text),
        "no absolute path, PRO layer name or key-like string",
    )
    rep.add("post_processing_label", artifact.get("label") == LABEL, "labelled as post-processing")
    return rep
