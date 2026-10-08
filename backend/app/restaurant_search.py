from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import embed_search_query
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.place_filters import cuisine_filter
from app.schemas import OsmRestaurantSearchResult

OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_ATTRIBUTION_URL = "https://www.openstreetmap.org/copyright"


def search_osm_places_by_text(
    query: str,
    session: Session,
    *,
    top_k: int = 12,
    city: str | None = None,
    cuisine: str | None = None,
) -> list[OsmRestaurantSearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = embed_search_query(query)
    cosine_distance = OsmPlace.embedding.cosine_distance(query_embedding)
    has_photos = (
        select(ImageEmbedding.id)
        .where(ImageEmbedding.osm_place_id == OsmPlace.id)
        .exists()
    )
    statement = (
        select(OsmPlace, cosine_distance)
        .where(OsmPlace.embedding.is_not(None), ~has_photos)
        .order_by(cosine_distance)
        .limit(top_k)
    )
    if city:
        statement = statement.where(
            func.lower(OsmPlace.city) == city.strip().lower()
        )
    if cuisine:
        statement = statement.where(cuisine_filter(cuisine))

    return [
        OsmRestaurantSearchResult(
            id=place.id,
            name=place.name,
            city=place.city,
            cuisine=place.cuisine,
            location=place.location,
            latitude=place.latitude,
            longitude=place.longitude,
            features=place.features or [],
            source_url=place.source_url,
            attribution=OSM_ATTRIBUTION,
            attribution_url=OSM_ATTRIBUTION_URL,
            similarity=1.0 - float(distance),
        )
        for place, distance in session.execute(statement)
    ]
