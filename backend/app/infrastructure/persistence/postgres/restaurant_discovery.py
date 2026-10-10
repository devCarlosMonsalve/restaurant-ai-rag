from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.persistence.postgres.image_queries import (
    search_images_by_text,
)
from app.infrastructure.persistence.postgres.restaurant_queries import (
    search_osm_places_by_text,
)
from app.schemas import ImageSearchResult, OsmRestaurantSearchResponse


class PostgresRestaurantDiscoveryAdapter:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search_restaurants(
        self,
        query: str,
        *,
        top_k: int,
        city: str | None,
        cuisine: str | None,
        include_places_with_photos: bool,
    ) -> OsmRestaurantSearchResponse:
        return search_osm_places_by_text(
            query,
            self.session,
            top_k=top_k,
            city=city,
            cuisine=cuisine,
            include_places_with_photos=include_places_with_photos,
        )

    def search_photos(
        self,
        query: str,
        *,
        top_k: int,
        osm_places_only: bool,
        city: str | None,
        cuisine: str | None,
        osm_place_id: UUID | None = None,
    ) -> list[ImageSearchResult]:
        if osm_place_id is None:
            return search_images_by_text(
                query,
                self.session,
                top_k=top_k,
                osm_places_only=osm_places_only,
                city=city,
                cuisine=cuisine,
            )
        return search_images_by_text(
            query,
            self.session,
            top_k=top_k,
            osm_places_only=osm_places_only,
            city=city,
            cuisine=cuisine,
            osm_place_id=osm_place_id,
        )
