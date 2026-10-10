import asyncio

import httpx
import pytest

from app.interfaces.a2a import (
    itinerary_planner_server,
    restaurant_discovery_server as a2a_server,
)
from app.itinerary_planning.application import planner as itinerary_planner
from app.agents.schemas import (
    RestaurantSearchAgentCandidate,
    RestaurantSearchAgentResponse,
)
from app.interfaces.a2a.itinerary_planner_agent_client import (
    ItineraryPlannerDelegationError,
    delegate_itinerary_plan,
)


def delegate_with_in_process_agents(
    query: str,
    day_count: int,
) -> itinerary_planner.ItineraryDiningDraft:
    async def invoke() -> itinerary_planner.ItineraryDiningDraft:
        async with a2a_server.app.router.lifespan_context(a2a_server.app):
            async with itinerary_planner_server.app.router.lifespan_context(
                itinerary_planner_server.app
            ):
                return await delegate_itinerary_plan(
                    query,
                    day_count,
                    transport=httpx.ASGITransport(
                        app=itinerary_planner_server.app
                    ),
                )

    return asyncio.run(invoke())


def test_client_runs_the_full_synthetic_agent_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_queries: list[str] = []

    def fake_search(query: str) -> RestaurantSearchAgentResponse:
        received_queries.append(query)
        return RestaurantSearchAgentResponse(
            answer="Synthetic restaurant evidence.",
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
        )

    async def plan_through_in_process_restaurant_agent(
        query: str,
        day_count: int,
    ) -> itinerary_planner.ItineraryDiningDraft:
        return await itinerary_planner.plan_itinerary_dining(
            query,
            day_count,
            transport=httpx.ASGITransport(app=a2a_server.app),
        )

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", fake_search)
    monkeypatch.setattr(
        itinerary_planner_server,
        "plan_itinerary_dining",
        plan_through_in_process_restaurant_agent,
    )

    draft = delegate_with_in_process_agents(
        "vegetarian restaurants in Madrid",
        2,
    )

    assert received_queries == ["vegetarian restaurants in Madrid"]
    assert draft.requested_days == 2
    assert len(draft.days) == 1
    assert draft.days[0].restaurant.name == "Casa Verde"
    assert draft.unfilled_days == 1
    assert draft.research_answer == "Synthetic restaurant evidence."


def test_client_surfaces_a_sanitized_planner_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_plan(
        _query: str,
        _day_count: int,
    ) -> itinerary_planner.ItineraryDiningDraft:
        del _query, _day_count
        raise RuntimeError("private downstream details")

    monkeypatch.setattr(
        itinerary_planner_server,
        "plan_itinerary_dining",
        failed_plan,
    )

    with pytest.raises(
        ItineraryPlannerDelegationError,
        match="could not complete the request",
    ) as error:
        delegate_with_in_process_agents("restaurants in Madrid", 1)

    assert "private downstream details" not in str(error.value)


def test_client_rejects_invalid_request_before_network_call() -> None:
    with pytest.raises(ValueError, match="non-empty restaurant query"):
        asyncio.run(delegate_itinerary_plan("  ", 1))

    with pytest.raises(ValueError, match="positive day count"):
        asyncio.run(delegate_itinerary_plan("restaurants in Madrid", 0))
