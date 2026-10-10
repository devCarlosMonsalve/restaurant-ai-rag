"""Compatibility entry point for OSM metadata ingestion."""

from app.ingestion import upsert_osm_place

__all__ = ["upsert_osm_place"]
