"""Compatibility exports for restaurant-search consumers.

The PostgreSQL query implementation lives in the persistence adapter module.
"""

from app.infrastructure.persistence.postgres.restaurant_queries import (
    OSM_ATTRIBUTION,
    OSM_ATTRIBUTION_URL,
    _evidence_summary,
    _requires_step_free_access,
    search_osm_places_by_text,
)

__all__ = [
    "OSM_ATTRIBUTION",
    "OSM_ATTRIBUTION_URL",
    "search_osm_places_by_text",
]
