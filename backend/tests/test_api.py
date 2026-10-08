from fastapi.testclient import TestClient

from app.schemas import (
    DocumentSearchResult,
    OsmRestaurantSearchResult,
    RagAnswerResponse,
    RagSource,
)
from app.models.osm_place import OsmPlace


def test_home_serves_restaurant_search_page(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Saborea Madrid" in response.text
    assert 'id="search-form"' in response.text
    assert 'id="city"' in response.text
    assert 'id="cuisine"' in response.text
    assert 'id="no-photo-section"' in response.text


def test_commons_image_route_serves_only_files_from_image_directory(
    client: TestClient,
    monkeypatch,
    tmp_path,
) -> None:
    image_path = tmp_path / "commons-photo.jpg"
    image_path.write_bytes(b"test image")
    monkeypatch.setattr("app.main.COMMONS_IMAGES_DIR", tmp_path)

    response = client.get("/images/files/commons-photo.jpg")
    missing_response = client.get("/images/files/missing.jpg")

    assert response.status_code == 200
    assert response.content == b"test image"
    assert response.headers["content-type"] == "image/jpeg"
    assert missing_response.status_code == 404


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_restaurant_and_list_it(client: TestClient) -> None:
    restaurant_data = {
        "name": "La Trattoria",
        "description": "Italian restaurant in the city center",
        "cuisine": "Italian",
        "price_range": "$$",
        "location": "Madrid",
    }

    create_response = client.post("/restaurants", json=restaurant_data)

    assert create_response.status_code == 201
    created_restaurant = create_response.json()
    assert created_restaurant["name"] == restaurant_data["name"]
    assert created_restaurant["id"]

    list_response = client.get("/restaurants")

    assert list_response.status_code == 200
    assert list_response.json() == [created_restaurant]


def test_create_restaurant_rejects_invalid_input(client: TestClient) -> None:
    response = client.post(
        "/restaurants",
        json={
            "name": "",
            "description": "Invalid restaurant",
            "cuisine": "Italian",
            "price_range": "$$",
            "location": "Madrid",
        },
    )

    assert response.status_code == 422
    assert client.get("/restaurants").json() == []


def test_list_osm_places_includes_required_attribution(
    client: TestClient,
    db_session,
) -> None:
    db_session.add(
        OsmPlace(
            osm_type="node",
            osm_id=123,
            name="Example Restaurant",
            city="Madrid",
            cuisine="italian",
            location="Calle Mayor 10",
            latitude=40.4,
            longitude=-3.7,
            wikimedia_commons=None,
            source_url="https://www.openstreetmap.org/node/123",
        )
    )
    db_session.commit()

    response = client.get("/restaurants/osm")

    assert response.status_code == 200
    assert response.json()[0]["name"] == "Example Restaurant"
    assert response.json()[0]["source_url"] == (
        "https://www.openstreetmap.org/node/123"
    )
    assert response.json()[0]["attribution"] == "© OpenStreetMap contributors"
    assert response.json()[0]["attribution_url"] == (
        "https://www.openstreetmap.org/copyright"
    )
    assert response.json()[0]["wikimedia_commons"] is None
    assert response.json()[0]["has_photos"] is False
    assert response.json()[0]["features"] == []


def test_osm_restaurant_search_returns_semantic_matches(
    client: TestClient,
    monkeypatch,
) -> None:
    expected = [
        OsmRestaurantSearchResult(
            id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
            name="Honest Greens",
            city="Madrid",
            cuisine="vegetarian",
            location="Calle de la Luna",
            latitude=40.4,
            longitude=-3.7,
            source_url="https://www.openstreetmap.org/node/123",
            attribution="© OpenStreetMap contributors",
            attribution_url="https://www.openstreetmap.org/copyright",
            similarity=0.82,
        )
    ]
    calls = {}

    def fake_search(query, session, *, top_k, city, cuisine):
        calls.update(
            query=query,
            top_k=top_k,
            city=city,
            cuisine=cuisine,
        )
        return expected

    monkeypatch.setattr("app.main.search_osm_places_by_text", fake_search)

    response = client.post(
        "/restaurants/osm/search",
        json={
            "query": "vegetarian food",
            "top_k": 3,
            "city": "Madrid",
            "cuisine": "vegetarian",
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["name"] == "Honest Greens"
    assert response.json()[0]["similarity"] == 0.82
    assert calls == {
        "query": "vegetarian food",
        "top_k": 3,
        "city": "Madrid",
        "cuisine": "vegetarian",
    }


def test_document_search_returns_ranked_chunks(
    client: TestClient,
    monkeypatch,
) -> None:
    expected_results = [
        DocumentSearchResult(
            document_id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
            source_name="menu.txt",
            chunk_index=0,
            content="Fresh pasta made daily",
            similarity=0.91,
        )
    ]
    calls = {}

    def fake_search(query, session, *, top_k):
        calls["query"] = query
        calls["top_k"] = top_k
        return expected_results

    monkeypatch.setattr("app.main.search_document_chunks", fake_search)

    response = client.post(
        "/documents/search",
        json={"query": "fresh pasta", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()[0]["content"] == "Fresh pasta made daily"
    assert response.json()[0]["similarity"] == 0.91
    assert calls == {"query": "fresh pasta", "top_k": 3}


def test_document_search_validates_query_and_top_k(client: TestClient) -> None:
    empty_query_response = client.post(
        "/documents/search",
        json={"query": "   "},
    )
    excessive_top_k_response = client.post(
        "/documents/search",
        json={"query": "pasta", "top_k": 21},
    )

    assert empty_query_response.status_code == 422
    assert excessive_top_k_response.status_code == 422


def test_ask_documents_returns_answer_and_sources(
    client: TestClient,
    monkeypatch,
) -> None:
    document_id = "e4ada293-47d1-492e-b6f3-ff27aa6624d1"
    expected_response = RagAnswerResponse(
        answer="La pasta fresca se prepara cada día [menu.txt#0].",
        sources=[
            RagSource(
                document_id=document_id,
                source_name="menu.txt",
                chunk_index=0,
                similarity=0.91,
            )
        ],
    )
    calls = {}

    def fake_answer(request, session):
        calls["query"] = request.query
        calls["top_k"] = request.top_k
        return expected_response

    monkeypatch.setattr("app.main.answer_with_rag", fake_answer)

    response = client.post(
        "/documents/ask",
        json={"query": "¿Cómo se prepara la pasta?", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == expected_response.answer
    assert response.json()["sources"][0]["source_name"] == "menu.txt"
    assert calls == {"query": "¿Cómo se prepara la pasta?", "top_k": 3}
