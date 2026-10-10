"""Compatibility exports for the Restaurant Discovery application."""

from app.restaurant_discovery.application.service import (
    CandidatePhotoSearchError,
    search_restaurant_photos,
    search_restaurants,
    search_verified_candidate_photos,
)

__all__ = [
    "CandidatePhotoSearchError",
    "search_restaurant_photos",
    "search_restaurants",
    "search_verified_candidate_photos",
]
