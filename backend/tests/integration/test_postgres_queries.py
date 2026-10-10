from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.application.restaurant_discovery import search_verified_candidate_photos
from app.infrastructure.persistence.postgres.restaurant_discovery import (
    PostgresRestaurantDiscoveryAdapter,
)
from app.image_search import search_images_by_text
from app.image_embeddings import IMAGE_EMBEDDING_DIMENSIONS
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.restaurant_search import search_osm_places_by_text
from app.schemas import OsmRestaurantSearchResult

pytestmark = pytest.mark.integration

TEXT_EMBEDDING_DIMENSIONS = 768


def test_postgres_image_search_scopes_results_and_checks_exact_osm_url(
    postgres_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_vector = _vector(IMAGE_EMBEDDING_DIMENSIONS, 1.0, 0.0)
    target = _add_place(postgres_session, "Casa Verde", "Madrid", "mexican")
    target_photos = [
        _add_photo(
            postgres_session,
            target,
            _vector(IMAGE_EMBEDDING_DIMENSIONS, 0.0, 1.0),
        ),
        _add_photo(
            postgres_session,
            target,
            _vector(IMAGE_EMBEDDING_DIMENSIONS, 0.6, 0.8),
        ),
    ]
    other_places = [
        _add_place(postgres_session, "Casa Verde", "Madrid", "mexican")
        for _ in range(11)
    ]
    for place in other_places:
        _add_photo(
            postgres_session,
            place,
            _vector(IMAGE_EMBEDDING_DIMENSIONS, 1.0, 0.0),
        )
    postgres_session.flush()

    monkeypatch.setattr(
        "app.infrastructure.persistence.postgres.image_queries.embed_text_for_image_search",
        lambda _: query_vector,
    )

    results = search_images_by_text(
        "zzzz",
        postgres_session,
        top_k=2,
        osm_places_only=True,
        osm_place_id=target.id,
    )

    assert [result.id for result in results] == [
        target_photos[1].id,
        target_photos[0].id,
    ]
    assert {result.restaurant_source_url for result in results} == {
        target.source_url
    }

    candidate = _candidate_result(target)
    verified_photos, warnings = search_verified_candidate_photos(
        [candidate],
        "zzzz",
        PostgresRestaurantDiscoveryAdapter(postgres_session),
        city="Madrid",
        cuisine="mexican",
        candidate_limit=1,
        photos_per_candidate=2,
    )
    assert {photo.id for photo in verified_photos} == {
        photo.id for photo in target_photos
    }
    assert warnings == []

    inconsistent_candidate = candidate.model_copy(
        update={"source_url": other_places[0].source_url}
    )
    rejected_photos, mismatch_warnings = search_verified_candidate_photos(
        [inconsistent_candidate],
        "zzzz",
        PostgresRestaurantDiscoveryAdapter(postgres_session),
        city="Madrid",
        cuisine="mexican",
        candidate_limit=1,
        photos_per_candidate=2,
    )
    assert rejected_photos == []
    assert any(
        "did not match the candidate" in warning for warning in mismatch_warnings
    )


def test_postgres_restaurant_search_photo_exclusion_and_opt_in_filters(
    postgres_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_vector = _vector(TEXT_EMBEDDING_DIMENSIONS, 1.0, 0.0)
    mexican_with_photo = _add_place(
        postgres_session,
        "Mexican With Photo",
        "Madrid",
        "mexican",
    )
    _add_photo(
        postgres_session,
        mexican_with_photo,
        _vector(IMAGE_EMBEDDING_DIMENSIONS, 1.0, 0.0),
    )
    mexican_without_photo = _add_place(
        postgres_session,
        "Mexican Without Photo",
        "Madrid",
        "mexican",
    )
    _add_place(
        postgres_session,
        "Barcelona Mexican With Photo",
        "Barcelona",
        "mexican",
    )
    madrid_italian_with_photo = _add_place(
        postgres_session,
        "Madrid Italian With Photo",
        "Madrid",
        "italian",
    )
    _add_photo(
        postgres_session,
        madrid_italian_with_photo,
        _vector(IMAGE_EMBEDDING_DIMENSIONS, 1.0, 0.0),
    )
    postgres_session.flush()

    monkeypatch.setattr(
        "app.infrastructure.persistence.postgres.restaurant_queries.embed_search_query",
        lambda _: query_vector,
    )

    search_options = {
        "top_k": 20,
        "city": "Madrid",
        "cuisine": "mexican",
    }
    default_results = search_osm_places_by_text(
        "mexican restaurants",
        postgres_session,
        **search_options,
    )
    included_results = search_osm_places_by_text(
        "mexican restaurants",
        postgres_session,
        include_places_with_photos=True,
        **search_options,
    )

    assert {place.id for place in default_results.results} == {
        mexican_without_photo.id
    }
    assert {place.id for place in included_results.results} == {
        mexican_with_photo.id,
        mexican_without_photo.id,
    }


def _add_place(
    session: Session,
    name: str,
    city: str,
    cuisine: str,
) -> OsmPlace:
    place = OsmPlace(
        osm_type="node",
        osm_id=uuid4().int % (2**63),
        name=name,
        city=city,
        cuisine=cuisine,
        location="Calle Mayor 1",
        latitude=40.4168,
        longitude=-3.7038,
        embedding=_vector(TEXT_EMBEDDING_DIMENSIONS, 1.0, 0.0),
        features=[],
        wikimedia_commons=None,
        source_url=f"https://www.openstreetmap.org/node/{uuid4().int}",
    )
    session.add(place)
    session.flush()
    return place


def _add_photo(
    session: Session,
    place: OsmPlace,
    embedding: list[float],
) -> ImageEmbedding:
    image_id = uuid4()
    photo = ImageEmbedding(
        source_name=f"commons-image-{image_id}.jpg",
        image_path=f"/integration-tests/{image_id}.jpg",
        embedding=embedding,
        source_url=f"https://commons.wikimedia.org/wiki/File:{image_id}.jpg",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="Integration test",
        osm_place_id=place.id,
    )
    session.add(photo)
    session.flush()
    return photo


def _candidate_result(place: OsmPlace) -> OsmRestaurantSearchResult:
    return OsmRestaurantSearchResult(
        id=place.id,
        name=place.name,
        city=place.city,
        cuisine=place.cuisine,
        location=place.location,
        latitude=place.latitude,
        longitude=place.longitude,
        features=place.features or [],
        source_url=place.source_url,
        attribution="© OpenStreetMap contributors",
        attribution_url="https://www.openstreetmap.org/copyright",
        similarity=0.9,
    )


def _vector(dimensions: int, first: float, second: float) -> list[float]:
    return [first, second, *([0.0] * (dimensions - 2))]
