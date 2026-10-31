"""Application-facing service functions shared by the CLI and (later) the UI.

P2/P4 provide catalog listing only. Investigation/verification services arrive in P6.
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
