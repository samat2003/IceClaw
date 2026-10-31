"""Helpers for bundle tests: a SYNTHETIC source package + scripted agent policies.

The synthetic files are real GeoTIFFs/WebP written with Rasterio/Pillow, so the package loader and
the verifier exercise their genuine code paths; they are not lunar data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from agent_helpers import (
    AUDITOR,
    EVT,
    GEOLOGY,
    PRED,
    PROJ,
    SHADOW,
    SKEPTIC,
    TEMP,
    TERRAIN,
    THERMAL,
    call,
    first_event_id,
    quiet,
)
from PIL import Image

from iceclaw.agents.provider import ScriptedChatClient
from iceclaw.config import Settings
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.data.catalog import proj_string
from iceclaw.reports import package as pkg
from iceclaw.services import InvestigationService

N = 32


def seen(ctx, kind, agent):
    """Event ids of ``kind`` authored by ``agent`` that appear in the latest prompt digest."""
    return [m[0] for m in EVT.findall(ctx.last_user_text) if m[1] == agent and m[2] == kind]


def _grid() -> dict[str, Any]:
    return {"xMinM": 320000.0, "yMaxM": 740000.0, "pixelSizeXM": 240.0, "pixelSizeYM": 240.0,
            "widthPx": N, "heightPx": N}  # fmt: skip


def write_tif(path: Path, values: np.ndarray, grid: dict[str, Any]) -> None:
    transform = Affine(240.0, 0.0, grid["xMinM"], 0.0, -240.0, grid["yMaxM"])
    with rasterio.open(
        path, "w", driver="GTiff", height=N, width=N, count=1, dtype="float32",
        crs=rasterio.crs.CRS.from_string(proj_string(PROJ)), transform=transform, nodata=np.nan,
    ) as ds:  # fmt: skip
        ds.write(values.astype("float32"), 1)


def write_prediction(path: Path, values: np.ndarray) -> dict[str, float]:
    lo, hi = float(values.min()), float(values.max())
    scale = (hi - lo) / 65535
    levels = np.round((values - lo) / scale).astype(np.uint32)
    rgba = np.zeros((N, N, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 3] = levels >> 8, levels & 255, 255
    Image.fromarray(rgba, "RGBA").save(path, "WEBP", lossless=True, quality=100)
    return {"offset": lo, "scale": scale}


def synthetic_spec(root: Path, region_id: str = "synthetic-region") -> pkg.SourceSpec:
    """SYNTHETIC but structurally real source files + spec (one prediction, 3 observations)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    rr, cc = np.mgrid[0:N, 0:N]
    pred = 0.5 + 0.3 * np.sin(rr / 5.0) * np.cos(cc / 7.0) + rng.normal(0, 0.01, (N, N))
    shadow = np.zeros((N, N))
    shadow[: N // 4, : N // 4] = 1.0
    temp = np.where(shadow >= 0.5, 90.0, 250.0) + rng.normal(0, 2.0, (N, N))
    slope = 8.0 + 6.0 * np.abs(np.sin(rr / 6.0)) + rng.normal(0, 0.5, (N, N))
    grid = _grid()
    enc = write_prediction(root / "synthetic-prediction-raw.webp", pred)
    write_tif(root / "synthetic_TMAX.tif", temp, grid)
    write_tif(root / "synthetic_LPSR.tif", shadow, grid)
    write_tif(root / "synthetic_SLOPE.tif", slope, grid)
    files = [
        pkg.SourceFile(PRED, pkg.KIND_PREDICTION, root / "synthetic-prediction-raw.webp",
                       "synthetic/prediction-raw.webp", "prospectivity_index",
                       {"format": "u16-rg-linear-webp", **enc}, "SYNTHETIC test prediction"),
        pkg.SourceFile(TEMP, pkg.KIND_OBSERVATION, root / "synthetic_TMAX.tif",
                       "sombench:synthetic_TMAX.tif", "K", {"format": "geotiff-float32-band1"}, "SYNTHETIC"),
        pkg.SourceFile(SHADOW, pkg.KIND_OBSERVATION, root / "synthetic_LPSR.tif",
                       "sombench:synthetic_LPSR.tif", "fraction", {"format": "geotiff-float32-band1"}, "SYNTHETIC"),
        pkg.SourceFile(TERRAIN, pkg.KIND_OBSERVATION, root / "synthetic_SLOPE.tif",
                       "sombench:synthetic_SLOPE.tif", "deg", {"format": "geotiff-float32-band1"}, "SYNTHETIC"),
    ]  # fmt: skip
    corners = ((0, 0), (N, 0), (N, N), (0, N))
    return pkg.SourceSpec(
        region_id=region_id, pole="south", source_mode="upstream_published_example",
        source_dataset="synthetic-dataset", source_split="test", source_revision="r" * 40,
        processing_version="synthetic", original_tile_id="synthetic-tile",
        prediction_checkpoint_id="synthetic.ckpt@sha256:" + "0" * 64,
        upstream_sectors_sha256="1" * 64, lunar_crs=proj_string(PROJ), projection=dict(PROJ),
        grid=grid, footprint=[ma.pixel_to_lat_lon(PROJ, grid, c, r) for c, r in corners],
        width=N, height=N, resolution_m=240.0, files=files,
        scalars=[{"name": "psr_fraction", "kind": "tile_scalar", "value": 0.0625, "unit": "fraction"}],
        upstream={"note": "SYNTHETIC"},
        attribution=["SYNTHETIC test data; not lunar data."],
    )  # fmt: skip


# ------------------------------------------------------------------------- agent policies
def _first_turn(txt: str) -> bool:
    return "start of the investigation" in txt


def auditor_rich():
    """Auditor: summary + peak + discontinuities, then a claim citing all three, then conclude."""
    ids: dict[str, str] = {}

    def policy(ctx):
        n = len(ctx.last_tool_results)
        if not _first_turn(ctx.last_user_text):
            return quiet(ctx)
        plan = [
            ("valid_pixel_summary", {"tool_id": "valid_pixel_summary", "modality": PRED}),
            ("peak_location", {"tool_id": "peak_location", "modality": PRED}),
            (
                "disc",
                {"tool_id": "discontinuity_candidates", "modality": PRED, "k_mad": 4.0, "top_n": 3},
            ),
        ]
        if 1 <= n <= 3:  # the previous call's result is the latest tool result
            ids[plan[n - 1][0]] = first_event_id(ctx)
        if n < 3:
            return call("run_science_tool", **plan[n][1])
        if n == 3:
            return call("publish_claim", text="Prediction raster summarized; peak and gradient candidates located.",
                        cites=list(ids.values()), status="supported", alternatives=["ordinary spatial variation"],
                        limitations=["screening statistics only"])  # fmt: skip
        if n == 4:
            return call(
                "conclude", summary="Auditor done.", cites=list(ids.values()), status="supported"
            )
        return "done"

    return policy


def one_tool_policy(tool_id: str, **kw):
    """Run one tool, claim citing it, conclude."""
    st: dict[str, str] = {}

    def policy(ctx):
        n = len(ctx.last_tool_results)
        if not _first_turn(ctx.last_user_text):
            return quiet(ctx)
        if n == 0:
            return call("run_science_tool", tool_id=tool_id, **kw)
        if n == 1:
            st["m"] = first_event_id(ctx)
            return call(
                "publish_claim", text=f"{tool_id} measured.", cites=[st["m"]], status="supported"
            )
        if n == 2:
            return call(
                "conclude", summary="done", cites=[st["m"], first_event_id(ctx)], status="supported"
            )
        return "ok"

    return policy


def skeptic_windowed():
    """Skeptic: wakes on a claim, runs a windowed tool, publishes its own claim, concludes."""
    st = {"done": False}

    def policy(ctx):
        n = len(ctx.last_tool_results)
        claims = seen(ctx, "claim", AUDITOR)
        if claims and not st["done"]:
            if n == 0:
                return call(
                    "run_science_tool",
                    tool_id="valid_pixel_summary",
                    modality=PRED,
                    window=[4, 20, 4, 20],
                )
            if n == 1:
                st["m"] = first_event_id(ctx)
                return call("publish_claim", text="Window summary examined near the auditor's locations.",
                            cites=[st["m"], claims[-1]], status="provisional")  # fmt: skip
            if n == 2:
                st["done"] = True
                return call("conclude", summary="No contradiction warranted; pixelwise follow-up proposed.",
                            cites=[first_event_id(ctx)], status="provisional")  # fmt: skip
        return quiet(ctx)

    return policy


def rich_policies() -> dict[str, Any]:
    return {
        THERMAL: one_tool_policy("thermal_psr_consistency", cold_trap_k=110.0),
        GEOLOGY: one_tool_policy("terrain_slope_summary", steep_deg=15.0),
        AUDITOR: auditor_rich(),
        SKEPTIC: skeptic_windowed(),
    }


def scripted_service(
    runs_root: Path, spec_root: Path, policies=None, emit_reasoning: bool = False
) -> InvestigationService:
    pol = policies or rich_policies()

    def factory(rt, aid):
        c = ScriptedChatClient(pol[aid], aid)
        c.emit_reasoning = emit_reasoning
        c.bind(rt, aid)
        return c

    settings = Settings(data_root=Path("/none"), runs_root=runs_root)
    spec = synthetic_spec(spec_root)
    return InvestigationService(settings, client_factory=factory, spec_provider=lambda _rid: spec)


# ------------------------------------------------------------------------ tamper utilities
def copy_bundle(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("internal"))
    return dst


def rehash(bundle: Path, *names: str) -> None:
    """Attacker model: recompute the manifest integrity entries for edited files."""
    m = json.loads((bundle / "input_manifest.json").read_text())
    for n in names or list(m["integrity"]):
        m["integrity"][n] = pkg.sha256_bytes((bundle / n).read_bytes())
    (bundle / "input_manifest.json").write_text(json.dumps(m, indent=1))


def failed_ids(report) -> set[str]:
    return {c.id for c in report.failures}
