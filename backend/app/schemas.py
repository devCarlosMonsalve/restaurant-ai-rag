from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.restaurant_discovery.application.contracts import (
    ImageSearchRequest,
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
)
from app.knowledge.application.contracts import (
    DocumentSearchRequest,
    DocumentSearchResult,
    RagAnswerResponse,
    RagQuestionRequest,
    RagSource,
)


class RestaurantCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1)
    cuisine: str = Field(min_length=1, max_length=100)
    price_range: str = Field(min_length=1, max_length=10)
    location: str = Field(min_length=1, max_length=255)


class RestaurantRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    cuisine: str
    price_range: str
    location: str


class OsmPlaceRead(BaseModel):
    id: UUID
    osm_type: str
    osm_id: int
    name: str
    city: str
    cuisine: str | None
    location: str | None
    latitude: float | None
    longitude: float | None
    features: list[str]
    wikimedia_commons: str | None
    source_url: str
    attribution: str
    attribution_url: str
    has_photos: bool


class RagQuestionWithPhotosRequest(RagQuestionRequest):
    include_photos: bool = False


class RagPhoto(BaseModel):
    source_name: str
    image_url: str
    similarity: float
    metadata_match_count: int = 0
    source_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    restaurant_name: str | None = None
    restaurant_location: str | None = None
    restaurant_cuisine: str | None = None
    restaurant_features: list[str] = Field(default_factory=list)
    restaurant_source_url: str | None = None
    restaurant_attribution: str | None = None
    restaurant_attribution_url: str | None = None


class RagAnswerWithPhotosResponse(RagAnswerResponse):
    photos: list[RagPhoto] = Field(default_factory=list)
