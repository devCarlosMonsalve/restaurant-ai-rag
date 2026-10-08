from sqlalchemy.orm import Session

from app.image_search import search_images_by_text
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
    return search_images_by_text(
        request.query,
        session,
        top_k=request.top_k,
        osm_places_only=request.osm_places_only,
        city=request.city,
        cuisine=request.cuisine,
    )
