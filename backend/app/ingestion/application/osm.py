"""Application use case for updating an OSM place."""

from uuid import UUID

from app.ingestion.application.ports import OSMPlaceRecord, OSMPlaceStore


def upsert_osm_place(
    place: OSMPlaceRecord,
    *,
    store: OSMPlaceStore,
) -> UUID:
    return store.upsert(place)
