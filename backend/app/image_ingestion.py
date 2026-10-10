"""Compatibility entry point for image ingestion."""

from app.ingestion import ImageSourceMetadata, ingest_image_to_database

__all__ = ["ImageSourceMetadata", "ingest_image_to_database"]
