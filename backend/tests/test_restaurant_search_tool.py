from unittest.mock import sentinel

import pytest
from pydantic import ValidationError

from app.schemas import OsmRestaurantSearchResponse
from app.tools.restaurant_search import search_restaurants


def test_search_restaurants_delegates_to_frozen_search_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = OsmRestaurantSearchResponse(
        results=[],
        evidence_status="no_evidence",
        evidence_message="No hay evidencia",
    )
    calls = {}

    def fake_search(query, session, *, top_k, city, cuisine):
        calls.update(
            query=query,
            session=session,
            top_k=top_k,
            city=city,
            cuisine=cuisine,
        )
        return expected

    monkeypatch.setattr(
        "app.tools.restaurant_search.search_osm_places_by_text",
        fake_search,
    )

    response = search_restaurants(
        "  terraza  ",
        sentinel.session,
        top_k=5,
        city=" Madrid ",
        cuisine="spanish",
    )

    assert response is expected
    assert calls == {
        "query": "terraza",
        "session": sentinel.session,
        "top_k": 5,
        "city": "Madrid",
        "cuisine": "spanish",
    }


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({"query": "  "}, "query"),
        ({"query": "restaurants", "top_k": 0}, "top_k"),
        ({"query": "restaurants", "top_k": 21}, "top_k"),
        ({"query": "restaurants", "city": " "}, "city"),
        ({"query": "restaurants", "cuisine": "x" * 101}, "cuisine"),
    ],
)
def test_search_restaurants_validates_tool_arguments(
    arguments: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(ValidationError) as error:
        search_restaurants(session=sentinel.session, **arguments)

    assert error.value.errors()[0]["loc"] == (field,)
