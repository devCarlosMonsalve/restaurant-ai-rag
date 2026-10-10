"""Compatibility entry point for OSM metadata ingestion."""

from app.ingestion.application.osm import upsert_osm_place

__all__ = ["upsert_osm_place"]
