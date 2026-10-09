from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.image_search import search_images_by_text
from app.restaurant_search import search_osm_places_by_text
from app.schemas import (
    ImageSearchRequest,
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
)


def search_restaurants(
    request: OsmRestaurantSearchRequest,
    session: Session,
    *,
    include_places_with_photos: bool = False,
) -> OsmRestaurantSearchResponse:
    return search_osm_places_by_text(
        request.query,
        session,
        top_k=request.top_k,
        city=request.city,
        cuisine=request.cuisine,
        include_places_with_photos=include_places_with_photos,
    )


def search_restaurant_photos(
    request: ImageSearchRequest,
    session: Session,
    *,
    osm_place_id: UUID | None = None,
) -> list[ImageSearchResult]:
    search_options: dict[str, Any] = {
        "top_k": request.top_k,
        "osm_places_only": request.osm_places_only,
        "city": request.city,
        "cuisine": request.cuisine,
    }
    if osm_place_id is not None:
        search_options["osm_place_id"] = osm_place_id

    return search_images_by_text(
        request.query,
        session,
        **search_options,
    )
