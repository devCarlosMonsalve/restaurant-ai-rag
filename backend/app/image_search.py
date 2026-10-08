import re
import unicodedata
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.image_embeddings import embed_text_for_image_search
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.place_filters import cuisine_filter
from app.schemas import ImageSearchResult

_STOPWORDS = {
    "a",
    "al",
    "and",
    "at",
    "con",
    "de",
    "del",
    "el",
    "en",
    "for",
    "from",
    "in",
    "la",
    "las",
    "los",
    "of",
    "on",
    "para",
    "por",
    "the",
    "to",
    "un",
    "una",
    "y",
    "madrid",
    "restaurant",
    "restaurants",
    "restaurante",
    "restaurantes",
    "photo",
    "image",
    "commons",
    "category",
    "file",
    "node",
    "way",
    "osm",
}

_TRANSLATION_GROUPS = (
    ("mexican", "mexicano", "mexicana"),
    ("italian", "italiano", "italiana"),
    ("spanish", "espanol", "espanola"),
    ("japanese", "japones", "japonesa"),
    ("chinese", "chino", "china"),
    ("french", "frances", "francesa"),
    ("indian", "indio", "india"),
    ("facade", "fachada"),
    ("entrance", "entrada"),
    ("stew", "cocido"),
    ("chicken", "pollo"),
)
_TRANSLATION_CANONICAL = {
    token: group[0]
    for group in _TRANSLATION_GROUPS
    for token in group
}


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    tokens = {
        token
        for token in re.findall(r"[a-z]+", ascii_text)
        if len(token) > 1 and token not in _STOPWORDS
    }
    return {_TRANSLATION_CANONICAL.get(token, token) for token in tokens}


def _metadata_match_count(query_tokens: set[str], *values: str | None) -> int:
    metadata_tokens: set[str] = set()
    for value in values:
        metadata_tokens.update(_tokens(value))
    return len(query_tokens & metadata_tokens)


def search_images_by_text(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
    osm_places_only: bool = False,
    sample_images_only: bool = False,
    city: str | None = None,
    cuisine: str | None = None,
) -> list[ImageSearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if osm_places_only and sample_images_only:
        raise ValueError("Only one image corpus can be selected")
    if sample_images_only and (city is not None or cuisine is not None):
        raise ValueError("Place filters cannot be used with the sample image corpus")

    query_embedding = embed_text_for_image_search(query)
    query_tokens = _tokens(query)
    cosine_distance = ImageEmbedding.embedding.cosine_distance(query_embedding)
    metadata_statement = (
        select(
            ImageEmbedding.id,
            ImageEmbedding.source_name,
            OsmPlace.name,
            OsmPlace.cuisine,
            OsmPlace.wikimedia_commons,
            OsmPlace.features,
        )
        .outerjoin(OsmPlace, ImageEmbedding.osm_place_id == OsmPlace.id)
    )
    image_statement = (
        select(ImageEmbedding, OsmPlace, cosine_distance)
        .outerjoin(OsmPlace, ImageEmbedding.osm_place_id == OsmPlace.id)
        .order_by(cosine_distance)
        .limit(min(100, max(top_k, top_k * 5)))
    )
    if osm_places_only:
        metadata_statement = metadata_statement.where(OsmPlace.id.is_not(None))
        image_statement = image_statement.where(OsmPlace.id.is_not(None))
    elif sample_images_only:
        metadata_statement = metadata_statement.where(
            OsmPlace.id.is_(None),
            ImageEmbedding.source_url.is_(None),
        )
    if city:
        city_match = func.lower(OsmPlace.city) == city.strip().lower()
        metadata_statement = metadata_statement.where(city_match)
        image_statement = image_statement.where(city_match)
    if cuisine:
        cuisine_match = cuisine_filter(cuisine)
        metadata_statement = metadata_statement.where(cuisine_match)
        image_statement = image_statement.where(cuisine_match)

    metadata_matches: dict[UUID, tuple[int, int]] = {}
    if query_tokens:
        for image_id, source_name, name, cuisine, commons_reference, features in session.execute(
            metadata_statement
        ):
            place_tokens: set[str] = set()
            for value in (name, cuisine, commons_reference, " ".join(features or [])):
                place_tokens.update(_tokens(value))
            image_tokens = _tokens(source_name) - place_tokens
            match_counts = (
                len(query_tokens & place_tokens),
                len(query_tokens & image_tokens),
            )
            if any(match_counts):
                metadata_matches[image_id] = match_counts

    matches_by_id = {
        image.id: (image, place, distance)
        for image, place, distance in session.execute(image_statement)
    }
    missing_metadata_matches = set(metadata_matches) - set(matches_by_id)
    if missing_metadata_matches:
        matching_images_statement = (
            select(ImageEmbedding, OsmPlace, cosine_distance)
            .outerjoin(OsmPlace, ImageEmbedding.osm_place_id == OsmPlace.id)
            .where(ImageEmbedding.id.in_(missing_metadata_matches))
        )
        if osm_places_only:
            matching_images_statement = matching_images_statement.where(
                OsmPlace.id.is_not(None)
            )
        elif sample_images_only:
            matching_images_statement = matching_images_statement.where(
                OsmPlace.id.is_(None),
                ImageEmbedding.source_url.is_(None),
            )
        if city:
            matching_images_statement = matching_images_statement.where(
                func.lower(OsmPlace.city) == city.strip().lower()
            )
        if cuisine:
            matching_images_statement = matching_images_statement.where(
                cuisine_filter(cuisine)
            )
        matches_by_id.update(
            {
                image.id: (image, place, distance)
                for image, place, distance in session.execute(
                    matching_images_statement
                )
            }
        )

    matches = sorted(
        matches_by_id.values(),
        key=lambda match: (
            tuple(-count for count in metadata_matches.get(match[0].id, (0, 0))),
            float(match[2]),
        ),
    )[:top_k]

    return [
        ImageSearchResult(
            id=image.id,
            source_name=image.source_name,
            image_path=image.image_path,
            similarity=1.0 - float(distance),
            metadata_match_count=sum(metadata_matches.get(image.id, (0, 0))),
            source_url=image.source_url,
            license_name=image.license_name,
            license_url=image.license_url,
            attribution=image.attribution,
            restaurant_name=place.name if place else None,
            restaurant_location=place.location if place else None,
            restaurant_cuisine=place.cuisine if place else None,
            restaurant_features=place.features or [] if place else [],
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
