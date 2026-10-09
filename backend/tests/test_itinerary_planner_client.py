import asyncio

import httpx
import pytest

from app import a2a_server
from app.agents.restaurant_search_agent import RestaurantSearchAgentError
from app.agents.schemas import (
    RestaurantSearchAgentPhoto,
    RestaurantSearchAgentResponse,
)
from app.itinerary_planner_client import (
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
