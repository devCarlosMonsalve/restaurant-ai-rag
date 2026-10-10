from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.persistence.postgres.image_queries import (
    search_images_by_text,
)
from app.infrastructure.persistence.postgres.restaurant_queries import (
    search_osm_places_by_text,
)
from app.observability import traced_span
from app.restaurant_discovery.application.contracts import (
    ImageSearchResult,
    OsmRestaurantSearchResponse,
)


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
        with traced_span(
            "retrieval.restaurants",
            {
                "retrieval.top_k": top_k,
                "retrieval.include_places_with_photos": include_places_with_photos,
            },
        ):
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
        with traced_span(
            "retrieval.restaurant_photos",
            {
                "retrieval.top_k": top_k,
                "retrieval.osm_places_only": osm_places_only,
                "retrieval.candidate_scoped": osm_place_id is not None,
            },
        ) as span:
            if osm_place_id is None:
                results = search_images_by_text(
                    query,
                    self.session,
                    top_k=top_k,
                    osm_places_only=osm_places_only,
                    city=city,
                    cuisine=cuisine,
                )
            else:
                results = search_images_by_text(
                    query,
                    self.session,
                    top_k=top_k,
                    osm_places_only=osm_places_only,
                    city=city,
                    cuisine=cuisine,
                    osm_place_id=osm_place_id,
                )
            span.set_attribute("retrieval.result_count", len(results))
            return results
