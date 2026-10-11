import asyncio

import pytest
from pydantic import ValidationError

from app.itinerary_planning.application.contracts import (
    ItineraryDaySuggestion,
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


def test_itinerary_dining_draft_accepts_consistent_payload() -> None:
    first = make_candidate("Casa Verde", 1)
    second = make_candidate("Huerta Viva", 2)
    draft = ItineraryDiningDraft.model_validate(
        {
            "query": "vegetarian restaurants",
            "requested_days": 2,
            "days": [
                ItineraryDaySuggestion(day_number=1, restaurant=first).model_dump()
            ],
            "alternatives": [second.model_dump()],
            "unfilled_days": 1,
            "research_answer": "Synthetic evidence.",
        }
    )

    assert draft.requested_days == 2
    assert draft.unfilled_days == 1
    assert draft.days[0].day_number == 1


@pytest.mark.parametrize(
    "invalid_change",
    [
        ("requested_days", 0),
        ("requested_days", True),
        ("unfilled_days", -1),
        ("unfilled_days", 0),
        ("day_number", 0),
        ("day_number", 2),
        ("duplicate_candidate", True),
        ("too_many_days", True),
    ],
)
def test_itinerary_dining_draft_rejects_inconsistent_payloads(
    invalid_change: tuple[str, object],
) -> None:
    first = make_candidate("Casa Verde", 1)
    second = make_candidate("Huerta Viva", 2)
    payload: dict[str, object] = {
        "query": "vegetarian restaurants",
        "requested_days": 2,
        "days": [
            {"day_number": 1, "restaurant": first.model_dump()},
        ],
        "alternatives": [second.model_dump()],
        "unfilled_days": 1,
        "research_answer": "Synthetic evidence.",
    }
    field, value = invalid_change

    if field == "day_number":
        payload["days"] = [{"day_number": value, "restaurant": first.model_dump()}]
    elif field == "duplicate_candidate":
        payload["alternatives"] = [first.model_dump()]
    elif field == "too_many_days":
        payload["days"] = [
            {"day_number": 1, "restaurant": first.model_dump()},
            {"day_number": 2, "restaurant": second.model_dump()},
        ]
        payload["unfilled_days"] = 0
    else:
        payload[field] = value

    with pytest.raises(ValidationError):
        ItineraryDiningDraft.model_validate(payload)


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


def test_planner_deduplicates_restaurants_by_source_url_in_retrieval_order() -> None:
    candidates = [
        make_candidate("Casa Verde", 1),
        make_candidate("Duplicate Casa Verde", 1),
        make_candidate("Huerta Viva", 2),
        make_candidate("Ecocentro", 3),
    ]
    provider = StubRestaurantEvidenceProvider(
        RestaurantSearchEvidence(
            answer="Synthetic candidates.",
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

    assert [day.restaurant.name for day in draft.days] == [
        "Casa Verde",
        "Huerta Viva",
    ]
    assert [candidate.name for candidate in draft.alternatives] == ["Ecocentro"]


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
