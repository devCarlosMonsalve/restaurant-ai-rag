from sqlalchemy.orm import Session

from app.restaurant_search import search_osm_places_by_text
from app.schemas import (
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
)


def search_restaurants(
    query: str,
    session: Session,
    *,
    top_k: int = 12,
    city: str | None = None,
    cuisine: str | None = None,
) -> OsmRestaurantSearchResponse:
    """Search restaurants using the frozen OSM semantic-search service.

    The host supplies the database session; only query parameters belong to
    the tool input.
    """
    request = OsmRestaurantSearchRequest(
        query=query,
        top_k=top_k,
        city=city,
        cuisine=cuisine,
    )
    return search_osm_places_by_text(
        request.query,
        session,
        top_k=request.top_k,
        city=request.city,
        cuisine=request.cuisine,
    )
