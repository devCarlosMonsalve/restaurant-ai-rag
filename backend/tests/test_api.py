from fastapi.testclient import TestClient

from app.schemas import (
    DocumentSearchResult,
    RagAnswerResponse,
    RagSource,
)

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
