"""Compatibility facade; use the application use case or app.ingestion API."""

from app.ingestion.composition import ingest_txt_to_database

__all__ = ["ingest_txt_to_database"]
