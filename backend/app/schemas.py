"""Compatibility exports for the former flat schema module."""

from app.interfaces.http.schemas import (
    DocumentSearchRequest,
    DocumentSearchResult,
    ImageSearchRequest,
    ImageSearchResult,
    OsmPlaceRead,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
    RagAnswerResponse,
    RagAnswerWithPhotosResponse,
    RagPhoto,
    RagQuestionRequest,
    RagQuestionWithPhotosRequest,
    RagSource,
    RestaurantCreate,
    RestaurantRead,
)

__all__ = [
    "DocumentSearchRequest",
    "DocumentSearchResult",
    "ImageSearchRequest",
    "ImageSearchResult",
    "OsmPlaceRead",
    "OsmRestaurantSearchRequest",
    "OsmRestaurantSearchResponse",
    "OsmRestaurantSearchResult",
    "RagAnswerResponse",
    "RagAnswerWithPhotosResponse",
    "RagPhoto",
    "RagQuestionRequest",
    "RagQuestionWithPhotosRequest",
    "RagSource",
    "RestaurantCreate",
    "RestaurantRead",
]
