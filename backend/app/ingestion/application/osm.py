"""Compatibility facade; use the PostgreSQL adapter through app.ingestion."""

from app.ingestion.composition import upsert_osm_place

__all__ = ["upsert_osm_place"]
