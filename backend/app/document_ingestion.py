"""Compatibility entry point for document ingestion."""

from app.ingestion.application.documents import ingest_txt_to_database

__all__ = ["ingest_txt_to_database"]
