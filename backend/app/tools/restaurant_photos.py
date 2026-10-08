from sqlalchemy.orm import Session

from app.application.restaurant_discovery import (
    search_restaurant_photos as search_restaurant_photos_use_case,
)
from app.schemas import ImageSearchRequest, ImageSearchResult


def search_restaurant_photos(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
    city: str | None = None,
    cuisine: str | None = None,
    osm_places_only: bool = True,
) -> list[ImageSearchResult]:
    """Search photos, restricting to OSM restaurants by default for this Tool."""
    request = ImageSearchRequest(
        query=query,
        top_k=top_k,
        city=city,
        cuisine=cuisine,
        osm_places_only=osm_places_only,
    )
    return search_restaurant_photos_use_case(request, session)
