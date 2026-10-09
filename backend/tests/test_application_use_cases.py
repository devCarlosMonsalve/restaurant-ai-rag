from unittest.mock import sentinel
from uuid import uuid4

import pytest

from app.application import knowledge, restaurant_discovery
from app.schemas import (
    DocumentSearchRequest,
    ImageSearchResult,
    ImageSearchRequest,
    OsmRestaurantSearchRequest,
    RagAnswerResponse,
    RagPhoto,
    RagQuestionRequest,
    RagQuestionWithPhotosRequest,
    RagSource,
)


def test_search_restaurants_use_case_delegates_to_repository() -> None:
    expected = sentinel.response
    calls = {}

    class FakeRepository:
        def search_restaurants(self, query, **kwargs):
            calls.update(query=query, **kwargs)
            return expected

    response = restaurant_discovery.search_restaurants(
        OsmRestaurantSearchRequest(
            query="terrace",
            top_k=4,
            city="Madrid",
            cuisine="spanish",
        ),
        FakeRepository(),
    )

    assert response is expected
    assert calls == {
        "query": "terrace",
        "top_k": 4,
        "city": "Madrid",
        "cuisine": "spanish",
        "include_places_with_photos": False,
    }


def test_search_restaurants_use_case_can_include_indexed_photo_places() -> None:
    expected = sentinel.response
    calls = {}

    class FakeRepository:
        def search_restaurants(self, query, **kwargs):
            calls.update(query=query, **kwargs)
            return expected

    response = restaurant_discovery.search_restaurants(
        OsmRestaurantSearchRequest(
            query="mexican restaurants",
            top_k=5,
            city="Madrid",
            cuisine="mexican",
        ),
        FakeRepository(),
        include_places_with_photos=True,
    )

    assert response is expected
    assert calls == {
        "query": "mexican restaurants",
        "top_k": 5,
        "city": "Madrid",
        "cuisine": "mexican",
        "include_places_with_photos": True,
    }


def test_search_restaurant_photos_use_case_delegates_to_repository() -> None:
    expected = [sentinel.photo]
    calls = {}

    class FakeRepository:
        def search_photos(self, query, **kwargs):
            calls.update(query=query, **kwargs)
            return expected

    response = restaurant_discovery.search_restaurant_photos(
        ImageSearchRequest(
            query="terrace",
            top_k=4,
            osm_places_only=True,
            city="Madrid",
            cuisine="spanish",
        ),
        FakeRepository(),
    )

    assert response is expected
    assert calls == {
        "query": "terrace",
        "top_k": 4,
        "osm_places_only": True,
        "city": "Madrid",
        "cuisine": "spanish",
        "osm_place_id": None,
    }


def test_search_restaurant_photos_passes_optional_osm_place_filter() -> None:
    expected = [sentinel.photo]
    calls = {}
    place_id = uuid4()

    class FakeRepository:
        def search_photos(self, query, **kwargs):
            calls.update(query=query, **kwargs)
            return expected

    response = restaurant_discovery.search_restaurant_photos(
        ImageSearchRequest(
            query="restaurant exterior",
            top_k=3,
            osm_places_only=True,
        ),
        FakeRepository(),
        osm_place_id=place_id,
    )

    assert response is expected
    assert calls == {
        "query": "restaurant exterior",
        "top_k": 3,
        "osm_places_only": True,
        "city": None,
        "cuisine": None,
        "osm_place_id": place_id,
    }


def test_search_documents_use_case_delegates_to_repository() -> None:
    expected = [sentinel.document]
    calls = {}

    class FakeRepository:
        def search_documents(self, query, *, top_k):
            calls.update(query=query, top_k=top_k)
            return expected

    response = knowledge.search_documents(
        DocumentSearchRequest(query="menu", top_k=3),
        FakeRepository(),
    )

    assert response is expected
    assert calls == {
        "query": "menu",
        "top_k": 3,
    }


def test_answer_from_documents_use_case_delegates_to_repository() -> None:
    expected = sentinel.answer
    calls = {}

    class FakeRepository:
        def answer_from_documents(self, request):
            calls["request"] = request
            return expected

    request = RagQuestionRequest(query="question", top_k=3)
    response = knowledge.answer_from_documents(request, FakeRepository())

    assert response is expected
    assert calls == {"request": request}


def test_answer_from_documents_with_photos_keeps_evidence_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_answer = RagAnswerResponse(
        answer="The menu describes tomato pasta [menu.txt#0].",
        sources=[
            RagSource(
                document_id=uuid4(),
                source_name="menu.txt",
                chunk_index=0,
                similarity=0.91,
            )
        ],
    )
    photo = ImageSearchResult(
        id=uuid4(),
        source_name="pasta.jpg",
        image_path="C:/images/pasta.jpg",
        similarity=0.88,
        license_name="CC BY 4.0",
        attribution="Photographer",
    )
    request = RagQuestionWithPhotosRequest(
        query="tomato pasta",
        top_k=3,
        include_photos=True,
    )
    calls = {}

    def fake_document_answer(received_request, repository):
        assert received_request is request
        assert repository is sentinel.knowledge_repository
        return document_answer

    monkeypatch.setattr(
        knowledge,
        "answer_from_documents",
        fake_document_answer,
    )

    def fake_search_restaurant_photos(image_request, repository):
        calls["request"] = image_request
        calls["repository"] = repository
        return [photo]

    monkeypatch.setattr(
        knowledge,
        "search_restaurant_photos",
        fake_search_restaurant_photos,
    )

    response = knowledge.answer_from_documents_with_photos(
        request,
        sentinel.knowledge_repository,
        sentinel.restaurant_repository,
    )

    assert response.answer == document_answer.answer
    assert response.sources == document_answer.sources
    assert response.photos == [
        RagPhoto(
            source_name="pasta.jpg",
            image_url="/images/files/pasta.jpg",
            similarity=0.88,
            license_name="CC BY 4.0",
            attribution="Photographer",
        )
    ]
    assert "image_path" not in response.model_dump_json()
    assert calls == {
        "request": ImageSearchRequest(
            query="tomato pasta",
            top_k=3,
            osm_places_only=True,
        ),
        "repository": sentinel.restaurant_repository,
    }


def test_answer_from_documents_with_photos_skips_search_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_answer = RagAnswerResponse(answer="Text answer.", sources=[])
    request = RagQuestionWithPhotosRequest(query="tomato pasta", top_k=3)

    def fake_document_answer(received_request, repository):
        assert received_request is request
        assert repository is sentinel.knowledge_repository
        return document_answer

    monkeypatch.setattr(
        knowledge,
        "answer_from_documents",
        fake_document_answer,
    )

    def unexpected_photo_search(image_request, repository):
        assert image_request == ImageSearchRequest(
            query=request.query,
            top_k=request.top_k,
            osm_places_only=True,
        )
        assert repository is sentinel.restaurant_repository
        pytest.fail("Photo search must be opt-in")

    monkeypatch.setattr(
        knowledge,
        "search_restaurant_photos",
        unexpected_photo_search,
    )

    response = knowledge.answer_from_documents_with_photos(
        request,
        sentinel.knowledge_repository,
        sentinel.restaurant_repository,
    )

    assert response.answer == "Text answer."
    assert response.sources == []
    assert response.photos == []


def test_answer_from_documents_with_photos_surfaces_search_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document_answer = RagAnswerResponse(answer="Text answer.", sources=[])
    request = RagQuestionWithPhotosRequest(
        query="tomato pasta",
        top_k=3,
        include_photos=True,
    )

    def return_document_answer(received_request, repository):
        assert received_request is request
        assert repository is sentinel.knowledge_repository
        return document_answer

    monkeypatch.setattr(
        knowledge,
        "answer_from_documents",
        return_document_answer,
    )

    def fail_photo_search(image_request, repository):
        assert image_request == ImageSearchRequest(
            query="tomato pasta",
            top_k=3,
            osm_places_only=True,
        )
        assert repository is sentinel.restaurant_repository
        raise RuntimeError("Image retrieval failed")

    monkeypatch.setattr(
        knowledge,
        "search_restaurant_photos",
        fail_photo_search,
    )

    with pytest.raises(RuntimeError, match="Image retrieval failed"):
        knowledge.answer_from_documents_with_photos(
            request,
            sentinel.knowledge_repository,
            sentinel.restaurant_repository,
        )
