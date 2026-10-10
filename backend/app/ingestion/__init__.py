"""Public ingestion API, composed with the configured adapters."""

from app.ingestion.composition import (
    ImageSourceMetadata,
    ingest_image_to_database,
    ingest_txt_to_database,
    upsert_osm_place,
)

__all__ = [
    "ImageSourceMetadata",
    "ingest_image_to_database",
    "ingest_txt_to_database",
    "upsert_osm_place",
]
