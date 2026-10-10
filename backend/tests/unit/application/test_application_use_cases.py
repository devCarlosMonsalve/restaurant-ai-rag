from unittest.mock import sentinel
from uuid import uuid4

import pytest

from app.application import (
    document_answer as document_answer_use_case,
)
from app.application import restaurant_discovery as legacy_restaurant_discovery
from app.knowledge.application import answer_question, search_documents
from app.knowledge.application.contracts import (
    DocumentSearchRequest as KnowledgeDocumentSearchRequest,
    DocumentSearchResult as KnowledgeDocumentSearchResult,
    RagAnswerResponse as KnowledgeRagAnswerResponse,
    RagQuestionRequest as KnowledgeRagQuestionRequest,
)
from app.restaurant_discovery.application import service as restaurant_discovery
from app.restaurant_discovery.application.ports import (
    RestaurantDiscoveryPort,
)
from app.restaurant_discovery.application.contracts import (
    ImageSearchRequest,
    OsmRestaurantSearchRequest,
)
from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)
from app.application.ports import RestaurantDiscoveryPort as LegacyRestaurantDiscoveryPort
from app.infrastructure.persistence.postgres.restaurant_discovery import (
    PostgresRestaurantDiscoveryAdapter as LegacyPostgresRestaurantDiscoveryAdapter,
)
from app.schemas import (
    DocumentSearchRequest as LegacyDocumentSearchRequest,
    DocumentSearchResult as LegacyDocumentSearchResult,
    DocumentSearchRequest,
    DocumentSearchResult,
    ImageSearchResult,
    ImageSearchRequest,
    OsmRestaurantSearchRequest,
    RagAnswerResponse,
    RagPhoto,
    RagQuestionRequest,
    RagQuestionWithPhotosRequest,
    RagSource,
)
from app.schemas import (
    ImageSearchRequest as LegacyImageSearchRequest,
    OsmRestaurantSearchRequest as LegacyOsmRestaurantSearchRequest,
)


def test_discovery_import_paths_are_backward_compatible() -> None:
    assert (
        legacy_restaurant_discovery.search_restaurants
        is restaurant_discovery.search_restaurants
    )
    assert (
        legacy_restaurant_discovery.search_restaurant_photos
        is restaurant_discovery.search_restaurant_photos
    )
    assert LegacyRestaurantDiscoveryPort is RestaurantDiscoveryPort
    assert LegacyPostgresRestaurantDiscoveryAdapter is PostgresRestaurantDiscoveryAdapter
    assert LegacyImageSearchRequest is ImageSearchRequest
    assert LegacyOsmRestaurantSearchRequest is OsmRestaurantSearchRequest
    assert LegacyDocumentSearchRequest is KnowledgeDocumentSearchRequest
    assert LegacyDocumentSearchResult is KnowledgeDocumentSearchResult
    assert RagAnswerResponse is KnowledgeRagAnswerResponse
    assert RagQuestionRequest is KnowledgeRagQuestionRequest


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

    response = search_documents.search_documents(
        DocumentSearchRequest(query="menu", top_k=3),
        FakeRepository(),
    )

    assert response is expected
    assert calls == {
        "query": "menu",
        "top_k": 3,
    }


def test_answer_from_documents_use_case_uses_retriever_and_generator() -> None:
    chunks = [
        DocumentSearchResult(
            document_id=uuid4(),
            source_name="menu.txt",
            chunk_index=0,
            content="The pasta is fresh.",
            similarity=0.91,
        )
    ]

    class FakeRetriever:
        def search_documents(self, query, *, top_k):
            assert query == "question"
            assert top_k == 3
            return chunks

    def fake_generator(question, context, /, *, context_chunk_count):
        assert question == "question"
        assert context == "[menu.txt#0]\nThe pasta is fresh."
        assert context_chunk_count == 1
        return "Grounded answer."

    response = answer_question.answer_from_documents(
        RagQuestionRequest(query="question", top_k=3),
        FakeRetriever(),
        fake_generator,
    )

    assert response.answer == "Grounded answer."
    assert response.sources == [
        RagSource(
            document_id=chunks[0].document_id,
            source_name="menu.txt",
            chunk_index=0,
            similarity=0.91,
        )
    ]


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

    def fake_document_answer(received_request, retriever, generate_answer):
        assert received_request is request
        assert retriever is sentinel.document_retriever
        assert generate_answer is sentinel.answer_generator
        return document_answer

    monkeypatch.setattr(
        document_answer_use_case,
        "answer_from_documents",
        fake_document_answer,
    )

    def fake_search_restaurant_photos(image_request, repository):
        calls["request"] = image_request
        calls["repository"] = repository
        return [photo]

    monkeypatch.setattr(
        document_answer_use_case,
        "search_restaurant_photos",
        fake_search_restaurant_photos,
    )

    response = document_answer_use_case.answer_from_documents_with_photos(
        request,
        sentinel.document_retriever,
        sentinel.answer_generator,
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

    def fake_document_answer(received_request, retriever, generate_answer):
        assert received_request is request
        assert retriever is sentinel.document_retriever
        assert generate_answer is sentinel.answer_generator
        return document_answer

    monkeypatch.setattr(
        document_answer_use_case,
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
        document_answer_use_case,
        "search_restaurant_photos",
        unexpected_photo_search,
    )

    response = document_answer_use_case.answer_from_documents_with_photos(
        request,
        sentinel.document_retriever,
        sentinel.answer_generator,
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

    def return_document_answer(received_request, retriever, generate_answer):
        assert received_request is request
        assert retriever is sentinel.document_retriever
        assert generate_answer is sentinel.answer_generator
        return document_answer

    monkeypatch.setattr(
        document_answer_use_case,
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
        document_answer_use_case,
        "search_restaurant_photos",
        fail_photo_search,
    )

    with pytest.raises(RuntimeError, match="Image retrieval failed"):
        document_answer_use_case.answer_from_documents_with_photos(
            request,
            sentinel.document_retriever,
            sentinel.answer_generator,
            sentinel.restaurant_repository,
        )
