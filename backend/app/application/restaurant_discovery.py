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
) -> OsmRestaurantSearchResponse:
    return search_osm_places_by_text(
        request.query,
        session,
        top_k=request.top_k,
        city=request.city,
        cuisine=request.cuisine,
    )


def search_restaurant_photos(
    request: ImageSearchRequest,
    session: Session,
) -> list[ImageSearchResult]:
    return search_images_by_text(
        request.query,
        session,
        top_k=request.top_k,
        osm_places_only=request.osm_places_only,
        city=request.city,
        cuisine=request.cuisine,
    )
