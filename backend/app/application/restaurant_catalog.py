from typing import Protocol
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
    features: list[str]
    wikimedia_commons: str | None
    source_url: str
    attribution: str
    attribution_url: str
    has_photos: bool


class RestaurantCatalog(Protocol):
    def create_restaurant(self, data: RestaurantCreate) -> RestaurantRead: ...

    def list_restaurants(self) -> list[RestaurantRead]: ...

    def list_osm_places(self) -> list[OsmPlaceRead]: ...


def create_restaurant(
    data: RestaurantCreate,
    catalog: RestaurantCatalog,
) -> RestaurantRead:
    return catalog.create_restaurant(data)


def list_restaurants(catalog: RestaurantCatalog) -> list[RestaurantRead]:
    return catalog.list_restaurants()


def list_osm_places(catalog: RestaurantCatalog) -> list[OsmPlaceRead]:
    return catalog.list_osm_places()
