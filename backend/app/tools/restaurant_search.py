from sqlalchemy.orm import Session

from app.application.restaurant_discovery import (
    search_restaurants as search_restaurants_use_case,
)
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
    """Find restaurant candidates from the OpenStreetMap catalog.

    Use this when the user asks to find restaurant options. This returns
    retrieval candidates, not a final personalized recommendation. Use
    ``city`` to restrict the city and ``cuisine`` to match an exact OSM cuisine
    tag; ``top_k`` controls the maximum number of candidates.

    Supported characteristics stated in ``query`` are filtered against
    explicit OSM tags. The result's evidence status explains whether those
    tags support the request. Unsupported details, including atmosphere, may
    remain unverified. OSM information can be incomplete or out of date;
    semantic similarity is a ranking value, not a confidence score. Kosher
    matches require an OSM check date within the configured freshness window,
    but the application does not independently validate the certifier.

    This searches restaurant records without indexed photos. For associated
    photographs, use ``search_restaurant_photos`` instead. The host injects
    ``session``; it is infrastructure, not an Agent-selected input.

    Args:
        query: Natural-language description of the restaurant request.
        session: Database session supplied by the host.
        top_k: Maximum number of restaurant candidates, from 1 to 20.
        city: Optional exact city name filter.
        cuisine: Optional exact OSM cuisine-tag filter.

    Returns:
        Restaurant candidates with OSM attributes, source links, semantic
        similarity, and evidence status/message.
    """
    request = OsmRestaurantSearchRequest(
        query=query,
        top_k=top_k,
        city=city,
        cuisine=cuisine,
    )
    return search_restaurants_use_case(request, session)
