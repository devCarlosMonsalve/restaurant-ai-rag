from unittest.mock import sentinel
from uuid import uuid4

import pytest

from app.application import knowledge, restaurant_discovery
from app.schemas import (
    DocumentSearchRequest,
    ImageSearchRequest,
    OsmRestaurantSearchRequest,
    RagQuestionRequest,
)


def test_search_restaurants_use_case_delegates_to_frozen_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = sentinel.response
    calls = {}

    def fake_search(query, session, **kwargs):
        calls.update(query=query, session=session, **kwargs)
        return expected

    monkeypatch.setattr(restaurant_discovery, "search_osm_places_by_text", fake_search)

    response = restaurant_discovery.search_restaurants(
        OsmRestaurantSearchRequest(
            query="terrace",
            top_k=4,
            city="Madrid",
            cuisine="spanish",
        ),
        sentinel.session,
    )

    assert response is expected
    assert calls == {
        "query": "terrace",
        "session": sentinel.session,
        "top_k": 4,
        "city": "Madrid",
        "cuisine": "spanish",
        "include_places_with_photos": False,
    }


def test_search_restaurants_use_case_can_include_indexed_photo_places(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = sentinel.response
    calls = {}

    def fake_search(query, session, **kwargs):
        calls.update(query=query, session=session, **kwargs)
        return expected

    monkeypatch.setattr(restaurant_discovery, "search_osm_places_by_text", fake_search)

    response = restaurant_discovery.search_restaurants(
        OsmRestaurantSearchRequest(
            query="mexican restaurants",
            top_k=5,
            city="Madrid",
            cuisine="mexican",
        ),
        sentinel.session,
        include_places_with_photos=True,
    )

    assert response is expected
    assert calls == {
        "query": "mexican restaurants",
        "session": sentinel.session,
        "top_k": 5,
        "city": "Madrid",
        "cuisine": "mexican",
        "include_places_with_photos": True,
    }


def test_search_restaurant_photos_use_case_delegates_to_frozen_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [sentinel.photo]
    calls = {}

    def fake_search(query, session, **kwargs):
        calls.update(query=query, session=session, **kwargs)
        return expected

    monkeypatch.setattr(restaurant_discovery, "search_images_by_text", fake_search)

    response = restaurant_discovery.search_restaurant_photos(
        ImageSearchRequest(
            query="terrace",
            top_k=4,
            osm_places_only=True,
            city="Madrid",
            cuisine="spanish",
        ),
        sentinel.session,
    )

    assert response is expected
    assert calls == {
        "query": "terrace",
        "session": sentinel.session,
        "top_k": 4,
        "osm_places_only": True,
        "city": "Madrid",
        "cuisine": "spanish",
    }


def test_search_restaurant_photos_passes_optional_osm_place_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [sentinel.photo]
    calls = {}
    place_id = uuid4()

    def fake_search(query, session, **kwargs):
        calls.update(query=query, session=session, **kwargs)
        return expected

    monkeypatch.setattr(restaurant_discovery, "search_images_by_text", fake_search)

    response = restaurant_discovery.search_restaurant_photos(
        ImageSearchRequest(
            query="restaurant exterior",
            top_k=3,
            osm_places_only=True,
        ),
        sentinel.session,
        osm_place_id=place_id,
    )

    assert response is expected
    assert calls == {
        "query": "restaurant exterior",
        "session": sentinel.session,
        "top_k": 3,
        "osm_places_only": True,
        "city": None,
        "cuisine": None,
        "osm_place_id": place_id,
    }


def test_search_documents_use_case_delegates_to_frozen_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [sentinel.document]
    calls = {}

    def fake_search(query, session, *, top_k):
        calls.update(query=query, session=session, top_k=top_k)
        return expected

    monkeypatch.setattr(knowledge, "search_document_chunks", fake_search)

    response = knowledge.search_documents(
        DocumentSearchRequest(query="menu", top_k=3),
        sentinel.session,
    )

    assert response is expected
    assert calls == {
        "query": "menu",
        "session": sentinel.session,
        "top_k": 3,
    }


def test_answer_from_documents_use_case_delegates_to_frozen_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = sentinel.answer
    calls = {}

    def fake_answer(request, session):
        calls.update(request=request, session=session)
        return expected

    monkeypatch.setattr(knowledge, "answer_with_rag", fake_answer)

    request = RagQuestionRequest(query="question", top_k=3)
    response = knowledge.answer_from_documents(request, sentinel.session)

    assert response is expected
    assert calls == {"request": request, "session": sentinel.session}
