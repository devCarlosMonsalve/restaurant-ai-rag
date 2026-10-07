from sqlalchemy import select
from sqlalchemy.orm import Session

from app.image_embeddings import embed_text_for_image_search
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.schemas import ImageSearchResult


def search_images_by_text(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> list[ImageSearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = embed_text_for_image_search(query)
    cosine_distance = ImageEmbedding.embedding.cosine_distance(query_embedding)
    statement = (
        select(ImageEmbedding, OsmPlace, cosine_distance)
        .outerjoin(OsmPlace, ImageEmbedding.osm_place_id == OsmPlace.id)
        .order_by(cosine_distance)
        .limit(top_k)
    )
    matches = session.execute(statement).all()

    return [
        ImageSearchResult(
            id=image.id,
            source_name=image.source_name,
            image_path=image.image_path,
            similarity=1.0 - float(distance),
            source_url=image.source_url,
            license_name=image.license_name,
            license_url=image.license_url,
            attribution=image.attribution,
            restaurant_name=place.name if place else None,
            restaurant_location=place.location if place else None,
            restaurant_cuisine=place.cuisine if place else None,
            restaurant_source_url=place.source_url if place else None,
            restaurant_attribution=(
                "© OpenStreetMap contributors" if place else None
            ),
            restaurant_attribution_url=(
                "https://www.openstreetmap.org/copyright" if place else None
            ),
        )
        for image, place, distance in matches
    ]
