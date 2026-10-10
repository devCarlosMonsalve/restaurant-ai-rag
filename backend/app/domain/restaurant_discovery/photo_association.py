"""Compatibility export for exact Restaurant Discovery photo association."""

from app.restaurant_discovery.domain.photo_association import (
    is_verified_photo_association,
)

__all__ = ["is_verified_photo_association"]
