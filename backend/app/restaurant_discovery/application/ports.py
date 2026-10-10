from typing import Protocol
from uuid import UUID

from app.restaurant_discovery.application.contracts import (
    ImageSearchResult,
    OsmRestaurantSearchResponse,
)


class RestaurantDiscoveryPort(Protocol):
    def search_restaurants(
        self,
        query: str,
        *,
        top_k: int,
        city: str | None,
        cuisine: str | None,
        include_places_with_photos: bool,
    ) -> OsmRestaurantSearchResponse: ...

    def search_photos(
        self,
        query: str,
        *,
        top_k: int,
        osm_places_only: bool,
        city: str | None,
        cuisine: str | None,
        osm_place_id: UUID | None = None,
    ) -> list[ImageSearchResult]: ...
