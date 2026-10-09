from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import OsmRestaurantSearchResult


class RestaurantPhotoWorkflowRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    city: str | None = Field(default=None, min_length=1, max_length=100)
    cuisine: str | None = Field(default=None, min_length=1, max_length=100)
    candidate_limit: int = Field(default=5, ge=1, le=20)
    photos_per_candidate: int = Field(default=1, ge=1, le=5)


class RestaurantPhoto(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID
    source_name: str
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
    image_url: str


class RestaurantWithPhotos(BaseModel):
    restaurant: OsmRestaurantSearchResult
    photos: list[RestaurantPhoto]


class RestaurantPhotoWorkflowResponse(BaseModel):
    query: str
    evidence_status: Literal[
        "not_required",
        "verified",
        "partial",
        "no_evidence",
        "stale_evidence",
        "unverified",
    ] | None
    photo_search_status: Literal["not_started", "completed"]
    restaurants_with_photos: list[RestaurantWithPhotos]
    candidates_without_returned_photos: list[OsmRestaurantSearchResult]
    warnings: list[str] = Field(default_factory=list)
