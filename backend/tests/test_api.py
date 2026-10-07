from fastapi.testclient import TestClient


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
