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
) -> list[ImageSearchResult]:
    """Find indexed photographs explicitly associated with OSM restaurants.

    Use this when the user asks to see restaurant photos or when photographs
    would help inspect candidate places. Results are limited to OSM-linked
    restaurant photos; the Agent cannot select the sample-image corpus.
    Optionally restrict the results by ``city`` and exact OSM ``cuisine`` tag.

    The ranking uses visual similarity and indexed image/place metadata. A
    similarity value is not a confidence score and does not establish that a
    restaurant satisfies a request. Photos are associated only through the
    stored OSM/Wikimedia Commons link; they may not reflect current conditions.
    Available source, license, and attribution fields should be preserved when
    presenting a photo. For restaurant candidates without photos, use
    ``search_restaurants``.

    The host injects ``session``; it is infrastructure, not an Agent-selected
    input.

    Args:
        query: Natural-language description of the desired restaurant photo.
        session: Database session supplied by the host.
        top_k: Maximum number of photos, from 1 to 20.
        city: Optional exact city name filter.
        cuisine: Optional exact OSM cuisine-tag filter.

    Returns:
        Matching indexed photos with visual similarity and available image
        license, attribution, and associated restaurant metadata.
    """
    request = ImageSearchRequest(
        query=query,
        top_k=top_k,
        city=city,
        cuisine=cuisine,
        osm_places_only=True,
    )
    return search_restaurant_photos_use_case(request, session)
