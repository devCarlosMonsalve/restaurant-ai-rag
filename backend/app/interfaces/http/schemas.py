"""HTTP schema exports composed from application and context contracts."""

from app.application.document_answer import (
    RagAnswerWithPhotosResponse,
    RagPhoto,
    RagQuestionWithPhotosRequest,
)
from app.application.restaurant_catalog import (
    OsmPlaceRead,
    RestaurantCreate,
    RestaurantRead,
)
from app.knowledge.application.contracts import (
    DocumentSearchRequest,
    DocumentSearchResult,
    RagAnswerResponse,
    RagQuestionRequest,
    RagSource,
)
from app.restaurant_discovery.application.contracts import (
    ImageSearchRequest,
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
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
