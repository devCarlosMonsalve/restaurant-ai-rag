import asyncio

import httpx
import pytest

from app.interfaces.a2a import restaurant_discovery_server as a2a_server
from app.agents.restaurant_search_agent import RestaurantSearchAgentError
from app.agents.schemas import (
    RestaurantSearchAgentCandidate,
    RestaurantSearchAgentPhoto,
    RestaurantSearchAgentResponse,
)
from app.itinerary_planning.infrastructure.a2a.restaurant_discovery_client import (
    RestaurantDiscoveryDelegationError,
    delegate_restaurant_discovery,
)


def delegate_with_in_process_agent(
    query: str,
) -> RestaurantSearchAgentResponse:
    async def invoke() -> RestaurantSearchAgentResponse:
        async with a2a_server.app.router.lifespan_context(a2a_server.app):
            return await delegate_restaurant_discovery(
                query,
                transport=httpx.ASGITransport(app=a2a_server.app),
            )

    return asyncio.run(invoke())


def test_planner_client_discovers_agent_and_consumes_synthetic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_queries: list[str] = []

    def fake_search(query: str) -> RestaurantSearchAgentResponse:
        received_queries.append(query)
        return RestaurantSearchAgentResponse(
            answer="Synthetic restaurant options for the itinerary.",
            restaurants=[
                RestaurantSearchAgentCandidate(
                    name="Casa Verde",
                    city="Madrid",
                    cuisine="vegetarian",
                    location="Calle Mayor 1",
                    latitude=40.4168,
                    longitude=-3.7038,
                    features=["outdoor_seating"],
                    source_url="https://www.openstreetmap.org/node/123",
                    attribution="© OpenStreetMap contributors",
                    attribution_url="https://www.openstreetmap.org/copyright",
                    similarity=0.82,
                )
            ],
            photos=[
                RestaurantSearchAgentPhoto(
                    image_url="/images/commons-photo.jpg",
                    source_url="https://commons.wikimedia.org/wiki/File:commons-photo.jpg",
                    attribution="Photographer",
                )
            ],
        )

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", fake_search)

    response = delegate_with_in_process_agent(
        "  vegetarian restaurants in Madrid  "
    )

    assert isinstance(response, RestaurantSearchAgentResponse)
    assert response.answer == "Synthetic restaurant options for the itinerary."
    assert response.restaurants[0].name == "Casa Verde"
    assert response.restaurants[0].features == ["outdoor_seating"]
    assert response.photos[0].image_url == "/images/commons-photo.jpg"
    assert response.photos[0].source_url == (
        "https://commons.wikimedia.org/wiki/File:commons-photo.jpg"
    )
    assert received_queries == ["vegetarian restaurants in Madrid"]


def test_planner_client_surfaces_failed_agent_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        raise RestaurantSearchAgentError("provider detail stays server-side")

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", failed_search)

    with pytest.raises(
        RestaurantDiscoveryDelegationError,
        match="could not complete the request",
    ):
        delegate_with_in_process_agent("synthetic query")


def test_planner_client_rejects_invalid_query_before_network_call() -> None:
    with pytest.raises(ValueError, match="between 1 and 2,000 characters"):
        asyncio.run(delegate_restaurant_discovery("  "))
