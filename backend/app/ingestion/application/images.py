"""Compatibility facade; use the application use case or app.ingestion API."""

from app.ingestion.composition import ImageSourceMetadata, ingest_image_to_database

__all__ = ["ImageSourceMetadata", "ingest_image_to_database"]
