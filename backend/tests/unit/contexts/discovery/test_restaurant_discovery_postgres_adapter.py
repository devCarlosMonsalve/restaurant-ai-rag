from contextlib import contextmanager
from unittest.mock import MagicMock, sentinel

import pytest

from app.restaurant_discovery.infrastructure import postgres


def test_restaurant_search_tracing_is_owned_by_postgres_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_calls = []
    monkeypatch.setattr(
        postgres,
        "search_osm_places_by_text",
        lambda *args, **kwargs: sentinel.results,
    )

    @contextmanager
    def traced_span(name, attributes):
        span_calls.append((name, attributes))
        yield MagicMock()

    monkeypatch.setattr(postgres, "traced_span", traced_span)

    result = postgres.PostgresRestaurantDiscoveryAdapter(MagicMock()).search_restaurants(
        "terrace",
        top_k=4,
        city="Madrid",
        cuisine="spanish",
        include_places_with_photos=True,
    )

    assert result is sentinel.results
    assert span_calls == [
        (
            "retrieval.restaurants",
            {
                "retrieval.top_k": 4,
                "retrieval.include_places_with_photos": True,
            },
        )
    ]


def test_photo_search_tracing_records_candidate_scope_and_result_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_calls = []
    results = [sentinel.photo]
    monkeypatch.setattr(
        postgres,
        "search_images_by_text",
        lambda *args, **kwargs: results,
    )

    @contextmanager
    def traced_span(name, attributes):
        span = MagicMock()
        span_calls.append((name, attributes, span))
        yield span

    monkeypatch.setattr(postgres, "traced_span", traced_span)
    place_id = sentinel.place_id

    response = postgres.PostgresRestaurantDiscoveryAdapter(
        MagicMock()
    ).search_photos(
        "terrace",
        top_k=3,
        osm_places_only=True,
        city="Madrid",
        cuisine=None,
        osm_place_id=place_id,
    )

    assert response is results
    name, attributes, span = span_calls[0]
    assert name == "retrieval.restaurant_photos"
    assert attributes == {
        "retrieval.top_k": 3,
        "retrieval.osm_places_only": True,
        "retrieval.candidate_scoped": True,
    }
    span.set_attribute.assert_called_once_with("retrieval.result_count", 1)
