import asyncio

import pytest

from app.itinerary_planning.application.contracts import (
    RestaurantEvidenceCandidate,
    RestaurantSearchEvidence,
)
from app.itinerary_planning.application.planner import (
    ItineraryDiningDraft,
    plan_itinerary_dining,
)


def make_candidate(name: str, osm_id: int) -> RestaurantEvidenceCandidate:
    return RestaurantEvidenceCandidate(
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


class StubRestaurantEvidenceProvider:
    def __init__(self, evidence: RestaurantSearchEvidence) -> None:
        self.evidence = evidence
        self.queries: list[str] = []

    async def search_restaurants(self, query: str) -> RestaurantSearchEvidence:
        self.queries.append(query)
        return self.evidence


def test_planner_assigns_ranked_candidates_once_per_day() -> None:
    candidates = [
        make_candidate("Casa Verde", 1),
        make_candidate("Sana Vegetariana", 2),
        make_candidate("Ecocentro", 3),
    ]

    provider = StubRestaurantEvidenceProvider(
        RestaurantSearchEvidence(
            answer="Synthetic candidates from the restaurant Agent.",
            restaurants=candidates,
        )
    )

    draft = asyncio.run(
        plan_itinerary_dining(
            "vegetarian restaurants in Madrid",
            day_count=2,
            evidence_provider=provider,
        )
    )

    assert provider.queries == ["vegetarian restaurants in Madrid"]
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
) -> None:
    provider = StubRestaurantEvidenceProvider(
        RestaurantSearchEvidence(
            answer="One synthetic candidate.",
            restaurants=[make_candidate("Casa Verde", 1)],
        )
    )

    draft = asyncio.run(
        plan_itinerary_dining(
            "vegetarian restaurants in Madrid",
            day_count=3,
            evidence_provider=provider,
        )
    )

    assert [suggestion.restaurant.name for suggestion in draft.days] == [
        "Casa Verde",
    ]
    assert draft.alternatives == []
    assert draft.unfilled_days == 2


def test_planner_reports_all_days_unfilled_when_no_candidates(
) -> None:
    provider = StubRestaurantEvidenceProvider(
        RestaurantSearchEvidence(
            answer="No structured candidates were returned.",
        )
    )

    draft = asyncio.run(
        plan_itinerary_dining(
            "vegetarian restaurants in Madrid",
            day_count=2,
            evidence_provider=provider,
        )
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
    provider = StubRestaurantEvidenceProvider(
        RestaurantSearchEvidence(answer="No search should occur.")
    )
    with pytest.raises(ValueError, match="positive day count"):
        asyncio.run(
            plan_itinerary_dining(
                query,
                day_count,
                evidence_provider=provider,
            )
        )
    assert provider.queries == []
