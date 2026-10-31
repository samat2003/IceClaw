"""Application-facing service functions shared by the CLI and (later) the UI.

P2 provides catalog listing only. Investigation/verification services arrive in P5.
"""

from __future__ import annotations

from typing import Any

from iceclaw.config import Settings
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
                "other_metadata": [
                    {"name": m.name, "kind": m.status, "value": m.value, "unit": m.unit}
                    for m in rec.availability
                    if m.status in ("tile_scalar", "single_anchor_point")
                ],
                "unavailable_modalities": [
                    m.name for m in rec.availability if m.status == "unavailable"
                ],
            }
        )
    return rows
