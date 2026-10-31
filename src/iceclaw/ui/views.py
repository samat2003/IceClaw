"""View-models for the workbench: read saved bundles and source rasters, shape them for display.

Nothing here computes science. Numbers come from the recorded ``evidence.json`` measurements
(each already recomputable by the offline verifier) and rasters come from hash-verified source
packages. Plotly figures are built from those arrays only.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iceclaw.agents.specialists import ROLES
from iceclaw.data import moonatlas_adapter as ma
from iceclaw.reports import export as ex
from iceclaw.reports import package as pkg
from iceclaw.reports.verify import opaque_violations

# ---------------------------------------------------------------------------------- wording
PROSPECTIVITY_NOTICE = (
    "The NASA–IBM layer is an **ice-prospectivity index** produced by a regression model. It is "
    "not detected or confirmed water ice, not a calibrated probability and not an ice abundance."
)
ATTENTION_NOTICE = (
    "**Investigation-attention order, not physical ice prospectivity.** It counts how many distinct "
    "agents' evidence covered a location. It is not independently validated, ties are listed "
    "alphabetically by target id (the order among tied targets means nothing), and it says nothing "
    "about where ice is."
)
RECORDED_LABEL = "RECORDED INVESTIGATION"
PRIVATE_EXCLUSION = (
    "The public bundle contains no provider-opaque reasoning payloads (`protected_data`) and no "
    "internal run history. The private transcript it was made from is not included."
)

# agent_id -> (accent colour, symbol). Colours are the Okabe-Ito colour-blind-safe set; every agent
# is also identified by text and a symbol, never by colour alone.
AGENT_STYLE: dict[str, tuple[str, str]] = {
    "thermal": ("#D55E00", "🌡"),
    "geology": ("#009E73", "⛰"),
    "prediction_auditor": ("#0072B2", "🔎"),
    "skeptic": ("#CC79A7", "⚖"),
    "supervisor": ("#595959", "⚙"),
}
AGENT_ORDER = ("thermal", "geology", "prediction_auditor", "skeptic")
FOLLOWUP_KINDS = ("test_request", "test_result", "challenge", "revision")


def agent_title(agent_id: str) -> str:
    if agent_id in ROLES:
        return ROLES[agent_id][0]
    return "Runtime supervisor" if agent_id == "supervisor" else agent_id


def agent_remit(agent_id: str) -> str:
    """The first sentence of the real role instruction the agent was given."""
    if agent_id not in ROLES:
        return "Neutral record written by the runtime (not a scientist)."
    return ROLES[agent_id][1].split(". ")[0].rstrip(".") + "."


# ------------------------------------------------------------------------------ layer config
@dataclass(frozen=True)
class LayerStyle:
    modality: str
    label: str
    short: str
    unit: str
    colorscale: str
    kind_badge: str
    note: str


LAYER_STYLES: dict[str, LayerStyle] = {
    "nasa_ibm_prediction_raw": LayerStyle(
        "nasa_ibm_prediction_raw",
        "NASA–IBM prediction (ice-prospectivity index)",
        "Prediction",
        "index (dimensionless, raw)",
        "Viridis",
        "MODEL OUTPUT",
        "Raw regression values, not clipped (some fall outside 0–1). Not an observation of ice.",
    ),
    "pixelwise_temperature": LayerStyle(
        "pixelwise_temperature",
        "TMAX: maximum surface temperature",
        "TMAX",
        "K",
        "Inferno",
        "OBSERVATION-DERIVED",
        "SomBench TMAX layer, byte-exact copy. Unit K rests on upstream naming and anchor checks.",
    ),
    "pixelwise_shadow_mask": LayerStyle(
        "pixelwise_shadow_mask",
        "LPSR: permanent-shadow fraction",
        "LPSR",
        "fraction (0–1)",
        "Blues",
        "OBSERVATION-DERIVED",
        "SomBench LPSR layer, byte-exact copy. 1 means permanently shadowed.",
    ),
    "pixelwise_terrain": LayerStyle(
        "pixelwise_terrain",
        "SLOPE: terrain slope",
        "SLOPE",
        "deg",
        "YlOrBr",
        "OBSERVATION-DERIVED",
        "SomBench SLOPE layer, byte-exact copy.",
    ),
}
LAYER_ORDER = tuple(LAYER_STYLES)


@dataclass
class LayerGrid:
    style: LayerStyle
    values: np.ndarray  # (H, W) float, NaN where not covered / not finite
    file_id: str
    sha256: str
    unit_in_source: str


@dataclass
class RegionGeometry:
    region_id: str
    x_km: np.ndarray  # (W,) cell centres, km from the pole in the lunar stereographic frame
    y_km: np.ndarray  # (H,) ascending (row H-1 first); layers are flipped to match
    lat: np.ndarray  # (H, W) aligned with the flipped layers
    lon: np.ndarray
    footprint: list[list[float]]
    resolution_m: float
    projection: dict[str, Any]
    grid: dict[str, Any]
    shape: tuple[int, int]


def build_layers(arrays: Any) -> tuple[RegionGeometry, dict[str, LayerGrid]]:
    """Display grids from a ``RegionArrays`` (source package or catalog). Georeferencing only:
    cell-centre coordinates via the existing projection helper; values are passed through."""
    g, proj = arrays.grid, arrays.projection
    any_layer = next(iter(arrays.layers.values()))
    h, w = any_layer.values.shape
    px, py = g["pixelSizeXM"], g["pixelSizeYM"]
    fe, fn = proj["falseEastingM"], proj["falseNorthingM"]
    x_km = np.array([(g["xMinM"] + (c + 0.5) * px - fe) / 1000.0 for c in range(w)])
    y_desc = np.array([(g["yMaxM"] - (r + 0.5) * py - fn) / 1000.0 for r in range(h)])
    lat = np.empty((h, w))
    lon = np.empty((h, w))
    for r in range(h):
        for c in range(w):
            lat[r, c], lon[r, c] = ma.pixel_to_lat_lon(proj, g, c + 0.5, r + 0.5)
    geom = RegionGeometry(
        region_id=arrays.region_id,
        x_km=x_km,
        y_km=y_desc[::-1].copy(),
        lat=lat[::-1].copy(),
        lon=lon[::-1].copy(),
        footprint=[list(p) for p in arrays.footprint],
        resolution_m=float(arrays.resolution_m),
        projection=dict(proj),
        grid=dict(g),
        shape=(h, w),
    )
    layers: dict[str, LayerGrid] = {}
    for name in LAYER_ORDER:
        lay = arrays.layers.get(name)
        if lay is None:
            continue
        vals = np.where(lay.covered & np.isfinite(lay.values), lay.values, np.nan)
        layers[name] = LayerGrid(
            LAYER_STYLES[name], vals[::-1].copy(), lay.file_id, lay.sha256, lay.unit
        )
    return geom, layers


def pixel_xy_km(geom: RegionGeometry, row: float, col: float) -> tuple[float, float]:
    """Plot coordinates (km) of a pixel centre given as (row, col) in the original raster."""
    g, proj = geom.grid, geom.projection
    x = g["xMinM"] + (col + 0.5) * g["pixelSizeXM"] - proj["falseEastingM"]
    y = g["yMaxM"] - (row + 0.5) * g["pixelSizeYM"] - proj["falseNorthingM"]
    return x / 1000.0, y / 1000.0


def make_heatmap_figure(
    geom: RegionGeometry,
    layer: LayerGrid,
    *,
    targets: list[dict[str, Any]] | None = None,
    height: int = 560,
):
    """Plotly heatmap in projected (stereographic) kilometres, 1:1 aspect, lat/lon on hover."""
    import plotly.graph_objects as go

    st = layer.style
    custom = np.stack([np.round(geom.lat, 4), np.round(geom.lon, 4)], axis=-1)
    fig = go.Figure(
        go.Heatmap(
            z=layer.values,
            x=geom.x_km,
            y=geom.y_km,
            customdata=custom,
            colorscale=st.colorscale,
            colorbar={"title": {"text": st.unit, "side": "right"}},
            hoverongaps=False,
            zsmooth=False,
            hovertemplate=(
                f"{st.short}: %{{z:.4g}} {st.unit.split(' ')[0]}<br>"
                "lat %{customdata[0]:.4f}°, lon %{customdata[1]:.4f}°<br>"
                "x %{x:.2f} km, y %{y:.2f} km<extra></extra>"
            ),
        )
    )
    if targets:
        xs, ys, texts = [], [], []
        for t in targets:
            xs.append(t["x_km"])
            ys.append(t["y_km"])
            texts.append(t["label"])
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name="Investigation-attention targets",
                text=texts,
                hovertemplate="%{text}<extra>attention target (not an ice location)</extra>",
                marker={
                    "symbol": "x",
                    "size": 11,
                    "color": "#FFFFFF",
                    "line": {"color": "#000000", "width": 2},
                },
            )
        )
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        title={"text": st.label, "x": 0.01, "font": {"size": 15}},
        xaxis={"title": "x, km (lunar south-polar stereographic)", "constrain": "domain"},
        yaxis={"title": "y, km", "scaleanchor": "x", "scaleratio": 1},
        legend={"orientation": "h", "y": -0.12},
    )
    return fig


def rank_targets(geom: RegionGeometry, rankings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rankings:
        if r.get("row") in ("", None) or r.get("col") in ("", None):
            continue
        x, y = pixel_xy_km(geom, float(r["row"]), float(r["col"]))
        out.append(
            {
                "x_km": x,
                "y_km": y,
                "label": f"#{r['rank']} {r['kind']} (row {r['row']}, col {r['col']})",
            }
        )
    return out


# ------------------------------------------------------------------------------ bundle view
class BundleLoadError(RuntimeError):
    pass


@dataclass
class TimelineRow:
    event_id: str
    t_offset_s: float
    agent_id: str
    kind: str
    text: str
    parents: list[str]
    tool_id: str | None
    followup: bool


@dataclass
class BundleView:
    path: Path
    manifest: dict[str, Any]
    evidence: dict[str, Any]
    events: list[dict[str, Any]]
    rankings: list[dict[str, Any]]
    report_md: str
    timeline: list[TimelineRow] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def region_id(self) -> str:
        return str(self.manifest["region_id"])

    @property
    def profile(self) -> str:
        return str(self.manifest.get("transcript_profile", "full-private"))

    @property
    def is_public(self) -> bool:
        return self.profile == ex.PROFILE_PUBLIC

    @property
    def recorded_mode(self) -> str:
        return str(self.manifest.get("mode", "?"))

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self.manifest.get("model_usage", {}))

    @property
    def summary(self) -> dict[str, Any]:
        return dict(self.evidence.get("summary", {}))

    @property
    def source_dir(self) -> Path:
        return self.path / self.manifest["source_package"]["embedded_path"]


def load_bundle_view(path: str | Path) -> BundleView:
    p = Path(path)
    need = ("input_manifest.json", "events.jsonl", "evidence.json", "rankings.csv", "report.md")
    missing = [n for n in need if not (p / n).is_file()]
    if missing:
        raise BundleLoadError(f"{p.name}: missing {', '.join(missing)}")
    try:
        manifest = json.loads((p / "input_manifest.json").read_text(encoding="utf-8"))
        evidence = json.loads((p / "evidence.json").read_text(encoding="utf-8"))
        events = [
            json.loads(x)
            for x in (p / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        rankings = list(
            csv.DictReader((p / "rankings.csv").read_text(encoding="utf-8").splitlines())
        )
    except (json.JSONDecodeError, KeyError, OSError) as e:
        raise BundleLoadError(f"{p.name}: unreadable bundle ({type(e).__name__})") from e
    view = BundleView(p, manifest, evidence, events, rankings, (p / "report.md").read_text("utf-8"))
    view.timeline = build_timeline(events)
    return view


def build_timeline(events: list[dict[str, Any]]) -> list[TimelineRow]:
    from datetime import datetime

    def ts(e: dict[str, Any]) -> float:
        return datetime.fromisoformat(e["timestamp_utc"].replace("Z", "+00:00")).timestamp()

    if not events:
        return []
    t0 = ts(events[0])
    return [
        TimelineRow(
            e["event_id"],
            ts(e) - t0,
            e["agent_id"],
            e["event_kind"],
            " ".join((e.get("claim_or_request") or "").split()),
            list(e.get("parent_event_ids") or []),
            e.get("tool_id"),
            e["event_kind"] in FOLLOWUP_KINDS,
        )
        for e in events
    ]


def followup_events(view: BundleView) -> list[TimelineRow]:
    return [r for r in view.timeline if r.followup]


@dataclass
class BundleSummary:
    run_id: str
    path: Path
    region_id: str
    profile: str
    created_utc: str
    model: str
    n_events: int
    calls: int


def discover_public_bundles(runs_root: Path) -> list[BundleSummary]:
    """Saved PUBLIC-redacted bundles only (private/full bundles are never offered)."""
    out: list[BundleSummary] = []
    root = Path(runs_root)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        mp = d / "input_manifest.json"
        if d.name.startswith("_") or not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if m.get("transcript_profile") != ex.PROFILE_PUBLIC:
            continue
        out.append(
            BundleSummary(
                d.name,
                d,
                str(m.get("region_id", "?")),
                ex.PROFILE_PUBLIC,
                str(m.get("created_utc", "")),
                f"{m.get('llm_provider', '?')}/{m.get('llm_model', '?')}",
                int(m.get("run", {}).get("events", 0)),
                int(m.get("model_usage", {}).get("calls", 0)),
            )
        )
    out.sort(key=lambda s: s.created_utc, reverse=True)
    return out


# --------------------------------------------------------------------------------- findings
UNIT_DISPLAY = {
    "prospectivity_index": "index (dimensionless)",
    "prospectivity_index/m": "index per metre",
    "pixel index (relative to window origin)": "pixel index",
    "dimensionless": "dimensionless fraction",
    "deg": "deg",
    "K": "K",
    "fraction": "fraction",
    "pixels": "pixels",
}
CATEGORY_ORDER = ("Thermal and shadow", "Terrain", "NASA–IBM prediction")


def measurement_category(layers: list[str]) -> str:
    if "nasa_ibm_prediction_raw" in layers:
        return "NASA–IBM prediction"
    if "pixelwise_terrain" in layers:
        return "Terrain"
    return "Thermal and shadow"


def evidence_basis(category: str) -> str:
    return (
        "DETERMINISTIC CALCULATION on NASA–IBM model output"
        if category == "NASA–IBM prediction"
        else "DETERMINISTIC CALCULATION on lunar observation data"
    )


def measurement_rows(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for m in evidence.get("measurements", []):
        cat = measurement_category(m.get("layers", []))
        params = {k: v for k, v in m.get("parameters", {}).items() if k != "modality"}
        for f in m.get("fields", []):
            v = f["value"]
            rows.append(
                {
                    "category": cat,
                    "event": m["event_id"],
                    "agent": agent_title(m["agent_id"]),
                    "tool": m["tool_id"],
                    "quantity": f["name"],
                    "value": v,
                    "unit": UNIT_DISPLAY.get(f["unit"], f["unit"]),
                    "parameters": json.dumps(params, sort_keys=True) if params else "",
                    "source files": ", ".join(m.get("source_file_ids", [])),
                    "basis": evidence_basis(cat),
                }
            )
    return rows


def headline_metrics(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """A few recorded numbers per domain for the summary strip (values copied, never derived)."""
    want = [
        ("Thermal and shadow", "thermal_psr_consistency", "psr_mean_k", "Mean temperature in PSR", "K"),
        ("Thermal and shadow", "thermal_psr_consistency", "non_psr_mean_k", "Mean temperature outside PSR", "K"),
        ("Thermal and shadow", "thermal_psr_consistency", "psr_below_cold_trap_fraction", "PSR pixels below the chosen 110 K screening threshold", "fraction"),
        ("Terrain", "terrain_slope_summary", "median_deg", "Median slope", "deg"),
        ("Terrain", "terrain_slope_summary", "steep_fraction", "Fraction of pixels above the chosen 15° threshold", "fraction"),
        ("NASA–IBM prediction", "unit_interval_check", "outside_unit_interval_pixels", "Pixels with raw index outside 0–1", "pixels"),
        ("NASA–IBM prediction", "peak_location", "edge_distance_px", "Raw maximum: distance from tile edge", "pixels"),
    ]  # fmt: skip
    out = []
    for cat, tool, name, label, unit in want:
        for m in evidence.get("measurements", []):
            if m["tool_id"] != tool:
                continue
            for f in m["fields"]:
                if f["name"] == name:
                    out.append(
                        {
                            "category": cat,
                            "label": label,
                            "value": f["value"],
                            "unit": unit,
                            "event": m["event_id"],
                        }
                    )
                    break
            else:
                continue
            break
    return out


def format_value(v: Any, unit: str = "") -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if unit in ("pixels",) or (float(v).is_integer() and abs(v) >= 1):
        return f"{int(v):,}"
    if unit == "fraction":
        return f"{v * 100:.2f} %" if 0 <= v <= 1 else f"{v:.4g}"
    return f"{v:.5g}"


def statements_by_agent(evidence: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for s in evidence.get("statements", []):
        out.setdefault(s["agent_id"], []).append(s)
    return out


def open_questions(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """Agents' own recorded limitations and unresolved items; nothing is added by the UI."""
    seen: set[tuple[str, str]] = set()
    rows = []
    for s in evidence.get("statements", []):
        for text in s.get("limitations", []) or []:
            key = (s["agent_id"], text)
            if key not in seen:
                seen.add(key)
                rows.append(
                    {"agent": agent_title(s["agent_id"]), "event": s["event_id"], "text": text}
                )
    return rows


def next_steps(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """Closing statements of each agent (their proposed follow-up), quoted verbatim."""
    rows = []
    for s in evidence.get("statements", []):
        if s["kind"] == "completion" and s["agent_id"] != "supervisor":
            rows.append(
                {
                    "agent": agent_title(s["agent_id"]),
                    "agent_id": s["agent_id"],
                    "event": s["event_id"],
                    "status": s["status"],
                    "text": s["text"],
                }
            )
    return rows


def alternatives(evidence: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for s in evidence.get("statements", []):
        for a in s.get("alternative_explanations", []) or []:
            rows.append({"agent": agent_title(s["agent_id"]), "event": s["event_id"], "text": a})
    return rows


# ---------------------------------------------------------------------------- verification
def verification_rows(report: Any) -> list[dict[str, str]]:
    """One row per real check, with the outcome exactly as the verifier reported it."""
    return [
        {
            "check": c.id,
            "outcome": "PASS" if c.ok else ("FAIL" if c.severity == "error" else "WARN"),
            "detail": c.detail,
        }
        for c in report.checks
    ]


# ------------------------------------------------------------------------------- download
class DownloadRefused(RuntimeError):
    pass


def public_bundle_zip(path: str | Path) -> bytes:
    """Zip a PUBLIC-redacted bundle for download. Refuses anything else, and anything that still
    contains provider-opaque data or private history (defence in depth; the verifier checks too)."""
    p = Path(path)
    try:
        m = json.loads((p / "input_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise DownloadRefused(f"{p.name}: not a readable bundle") from e
    if (
        m.get("bundle_format") != ex.BUNDLE_FORMAT_V2
        or m.get("transcript_profile") != ex.PROFILE_PUBLIC
    ):
        raise DownloadRefused("only public-redacted iceclaw-bundle/2 packages can be downloaded")
    if (p / "internal").exists():
        raise DownloadRefused("bundle contains internal run history")
    if opaque_violations((p / "llm_calls.jsonl").read_text(encoding="utf-8")):
        raise DownloadRefused("transcript still contains provider-opaque data")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p.rglob("*")):
            if f.is_file():
                z.write(f, f"{p.name}/{f.relative_to(p).as_posix()}")
    return buf.getvalue()


def source_arrays(view: BundleView):
    """Hash-verified arrays of the bundle's own embedded source package."""
    return pkg.load_package(view.source_dir).arrays
