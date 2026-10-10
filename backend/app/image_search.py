"""Compatibility exports for image-search consumers.

The PostgreSQL query implementation lives in the persistence adapter module.
"""

from app.infrastructure.persistence.postgres.image_queries import (
    _metadata_match_count,
    _tokens,
    search_images_by_text,
)

__all__ = ["search_images_by_text"]
