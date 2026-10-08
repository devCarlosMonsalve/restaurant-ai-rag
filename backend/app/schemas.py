from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
    wikimedia_commons: str
    source_url: str
    attribution: str
    attribution_url: str


class DocumentSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class DocumentSearchResult(BaseModel):
    document_id: UUID
    source_name: str
    chunk_index: int
    content: str
    similarity: float


class ImageSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    osm_places_only: bool = False


class ImageSearchResult(BaseModel):
    id: UUID
    source_name: str
    image_path: str
    similarity: float
    metadata_match_count: int = 0
    source_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    restaurant_name: str | None = None
    restaurant_location: str | None = None
    restaurant_cuisine: str | None = None
    restaurant_source_url: str | None = None
    restaurant_attribution: str | None = None
    restaurant_attribution_url: str | None = None


class RagQuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RagSource(BaseModel):
    document_id: UUID
    source_name: str
    chunk_index: int
    similarity: float


class RagAnswerResponse(BaseModel):
    answer: str
    sources: list[RagSource]
