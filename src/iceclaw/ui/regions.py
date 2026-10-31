"""Region list for the selector, from the existing catalog service (never invented).

Verified paired regions are investigable. Observation-only tiles are listed as UNAVAILABLE for
prediction investigations with the catalog's own reason. If the local data roots are missing the
catalog is empty and the caller falls back to regions named by saved bundles (clearly labelled).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iceclaw import services
from iceclaw.config import Settings

INVESTIGABLE = "PAIRED"
OBSERVATION_ONLY = "OBSERVATION_ONLY"


@dataclass
class RegionRow:
    region_id: str
    status: str  # PAIRED | OBSERVATION_ONLY | other catalog status
    investigable: bool
    reason: str
    tile: str = ""
    grid: str = ""
    resolution_m: float | None = None
    modalities: list[str] = field(default_factory=list)
    center: tuple[float, float] | None = None
    split: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.investigable:
            return f"{self.region_id}  ·  paired: prediction + TMAX/LPSR/SLOPE"
        return f"{self.region_id}  ·  UNAVAILABLE: observations only, no prediction"


@dataclass
class CatalogView:
    rows: list[RegionRow]
    warning: str = ""

    @property
    def investigable_ids(self) -> set[str]:
        return {r.region_id for r in self.rows if r.investigable}

    def get(self, region_id: str) -> RegionRow | None:
        return next((r for r in self.rows if r.region_id == region_id), None)


def load_catalog_view(settings: Settings) -> CatalogView:
    """Existing ``discover`` + ``describe_*`` services; failures degrade to an empty view."""
    try:
        catalog = services.load_catalog(settings)
        paired = services.describe_regions(catalog)
        obs = services.describe_observation_only(catalog)
    except Exception as e:  # noqa: BLE001 - a broken data root must not crash the page
        return CatalogView([], f"Local catalog unavailable ({type(e).__name__}).")
    rows: list[RegionRow] = []
    for d in paired:
        ok = d["status"] == INVESTIGABLE and d["readiness"] == "verified"
        rows.append(
            RegionRow(
                d["region_id"],
                d["status"],
                ok,
                "" if ok else (d.get("pairing_error") or d.get("readiness_reason") or ""),
                d["original_tile_id"],
                f"{d['grid'][0]}×{d['grid'][1]}",
                d["spatial_resolution_m"],
                list(d["pixelwise_modalities"]),
                (d["center_lat_lon"][0], d["center_lat_lon"][1]),
                d["source_split"],
                {"checkpoint": d["checkpoint"], "dataset_revision": d["source_revision"]},
            )
        )
    for d in obs:
        rows.append(
            RegionRow(
                d["region_id"],
                OBSERVATION_ONLY,
                False,
                d["reason"],
                d["original_tile_id"],
                f"{d['grid'][0]}×{d['grid'][1]}",
                None,
                [],
                None,
                d["source_split"],
            )
        )
    warning = ""
    if not rows:
        warning = (
            "No local catalog data found (MOONATLAS export or SomBench roots). Saved "
            "investigations can still be reviewed and verified offline."
        )
    return CatalogView(rows, warning)


def load_catalog_arrays(settings: Settings, region_id: str):
    """Arrays for a verified paired region through the existing catalog (checksums re-verified)."""
    return services.load_catalog(settings).load_arrays(region_id)
