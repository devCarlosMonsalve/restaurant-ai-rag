from unittest.mock import sentinel

import pytest
from pydantic import ValidationError

from app.schemas import (
    DocumentSearchResult,
    ImageSearchResult,
    RagAnswerResponse,
)
from app.tools import (
    answer_from_documents,
    search_documents,
    search_restaurant_photos,
)


def test_search_restaurant_photos_delegates_to_existing_image_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [sentinel.image_result]
    calls = {}

    def fake_search(query, session, **kwargs):
        calls.update(query=query, session=session, **kwargs)
        return expected

    monkeypatch.setattr(
        "app.tools.restaurant_photos.search_images_by_text",
        fake_search,
    )

    results = search_restaurant_photos(
        "  pasta  ",
        sentinel.session,
        top_k=3,
        city=" Madrid ",
        cuisine="italian",
    )

    assert results is expected
    assert calls == {
        "query": "pasta",
        "session": sentinel.session,
        "top_k": 3,
        "city": "Madrid",
        "cuisine": "italian",
        "osm_places_only": True,
    }


def test_search_restaurant_photos_can_preserve_sample_corpus_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake_search(query, session, **kwargs):
        calls.update(query=query, session=session, **kwargs)
        return []

    monkeypatch.setattr(
        "app.tools.restaurant_photos.search_images_by_text",
        fake_search,
    )

    search_restaurant_photos(
        "pasta",
        sentinel.session,
        osm_places_only=False,
    )

    assert calls["osm_places_only"] is False


def test_search_documents_delegates_to_existing_document_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [
        DocumentSearchResult(
            document_id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
            source_name="menu.txt",
            chunk_index=0,
            content="Fresh pasta",
            similarity=0.9,
        )
    ]
    calls = {}

    def fake_search(query, session, *, top_k):
        calls.update(query=query, session=session, top_k=top_k)
        return expected

    monkeypatch.setattr(
        "app.tools.document_search.search_document_chunks",
        fake_search,
    )

    results = search_documents("  pasta  ", sentinel.session, top_k=3)

    assert results is expected
    assert calls == {
        "query": "pasta",
        "session": sentinel.session,
        "top_k": 3,
    }


def test_answer_from_documents_delegates_to_existing_rag_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RagAnswerResponse(answer="Answer [menu.txt#0]", sources=[])
    calls = {}

    def fake_answer(request, session):
        calls.update(
            query=request.query,
            top_k=request.top_k,
            session=session,
        )
        return expected

    monkeypatch.setattr("app.tools.document_answer.answer_with_rag", fake_answer)

    response = answer_from_documents(
        "  how is pasta prepared?  ",
        sentinel.session,
        top_k=4,
    )

    assert response is expected
    assert calls == {
        "query": "how is pasta prepared?",
        "top_k": 4,
        "session": sentinel.session,
    }


@pytest.mark.parametrize(
    ("call", "field"),
    [
        (lambda: search_restaurant_photos(" ", sentinel.session), "query"),
        (lambda: search_restaurant_photos("pasta", sentinel.session, top_k=21), "top_k"),
        (lambda: search_documents("", sentinel.session), "query"),
        (lambda: search_documents("pasta", sentinel.session, top_k=0), "top_k"),
        (lambda: answer_from_documents(" ", sentinel.session), "query"),
        (lambda: answer_from_documents("question", sentinel.session, top_k=21), "top_k"),
    ],
)
def test_document_and_photo_tools_validate_arguments(call, field: str) -> None:
    with pytest.raises(ValidationError) as error:
        call()

    assert error.value.errors()[0]["loc"] == (field,)


def test_photo_result_type_includes_license_and_attribution_contract() -> None:
    result = ImageSearchResult(
        id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
        source_name="pasta.jpg",
        image_path="C:/images/pasta.jpg",
        similarity=0.9,
        source_url="https://commons.wikimedia.org/wiki/File:Pasta.jpg",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Photographer",
    )

    assert result.license_name == "CC BY 4.0"
    assert result.attribution == "Photographer"
