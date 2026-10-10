import asyncio

import httpx
import pytest

from app import a2a_server
from app.agents.schemas import (
    RestaurantSearchAgentCandidate,
    RestaurantSearchAgentResponse,
)
from app.itinerary_planner import ItineraryDiningDraft, plan_itinerary_dining


def make_candidate(name: str, osm_id: int) -> RestaurantSearchAgentCandidate:
    return RestaurantSearchAgentCandidate(
        name=name,
        city="Madrid",
        cuisine="vegetarian",
        location=f"Calle Mayor {osm_id}",
        latitude=40.4168,
        longitude=-3.7038,
        features=["outdoor_seating"],
        source_url=f"https://www.openstreetmap.org/node/{osm_id}",
        attribution="© OpenStreetMap contributors",
        attribution_url="https://www.openstreetmap.org/copyright",
        similarity=0.9 - osm_id / 1000,
    )


def plan_with_in_process_agent(
    query: str,
    day_count: int,
) -> ItineraryDiningDraft:
    async def invoke() -> ItineraryDiningDraft:
        async with a2a_server.app.router.lifespan_context(a2a_server.app):
            return await plan_itinerary_dining(
                query,
                day_count,
                transport=httpx.ASGITransport(app=a2a_server.app),
            )

    return asyncio.run(invoke())


def test_planner_assigns_ranked_candidates_once_per_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        make_candidate("Casa Verde", 1),
        make_candidate("Sana Vegetariana", 2),
        make_candidate("Ecocentro", 3),
    ]

    def fake_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        return RestaurantSearchAgentResponse(
            answer="Synthetic candidates from the restaurant Agent.",
            restaurants=candidates,
        )

    monkeypatch.setattr(
        a2a_server,
        "_run_restaurant_search",
        fake_search,
    )

    draft = plan_with_in_process_agent(
        "vegetarian restaurants in Madrid",
        day_count=2,
    )

    assert draft.requested_days == 2
    assert [suggestion.day_number for suggestion in draft.days] == [1, 2]
    assert [suggestion.restaurant.name for suggestion in draft.days] == [
        "Casa Verde",
        "Sana Vegetariana",
    ]
    assert [candidate.name for candidate in draft.alternatives] == ["Ecocentro"]
    assert draft.unfilled_days == 0
    assert draft.research_answer == (
        "Synthetic candidates from the restaurant Agent."
    )
    assert "routes" in draft.evidence_notice


def test_planner_leaves_days_unfilled_when_evidence_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        return RestaurantSearchAgentResponse(
            answer="One synthetic candidate.",
            restaurants=[make_candidate("Casa Verde", 1)],
        )

    monkeypatch.setattr(
        a2a_server,
        "_run_restaurant_search",
        fake_search,
    )

    draft = plan_with_in_process_agent(
        "vegetarian restaurants in Madrid",
        day_count=3,
    )

    assert [suggestion.restaurant.name for suggestion in draft.days] == [
        "Casa Verde",
    ]
    assert draft.alternatives == []
    assert draft.unfilled_days == 2


def test_planner_reports_all_days_unfilled_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        return RestaurantSearchAgentResponse(
            answer="No structured candidates were returned.",
        )

    monkeypatch.setattr(
        a2a_server,
        "_run_restaurant_search",
        fake_search,
    )

    draft = plan_with_in_process_agent(
        "vegetarian restaurants in Madrid",
        day_count=2,
    )

    assert draft.days == []
    assert draft.alternatives == []
    assert draft.unfilled_days == 2


@pytest.mark.parametrize(
    ("query", "day_count"),
    [
        ("  ", 1),
        ("valid query", 0),
    ],
)
def test_planner_rejects_invalid_request_before_network_call(
    query: str,
    day_count: int,
) -> None:
    with pytest.raises(ValueError, match="positive day count"):
        asyncio.run(plan_itinerary_dining(query, day_count))
