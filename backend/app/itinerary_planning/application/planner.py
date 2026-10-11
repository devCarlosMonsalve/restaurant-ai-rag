from pydantic import ValidationError

from app.itinerary_planning.application.contracts import (
    ItineraryDaySuggestion,
    ItineraryDiningDraft,
    ItineraryPlanningRequest,
    RestaurantEvidenceCandidate,
    RestaurantSearchEvidence,
)
from app.itinerary_planning.application.ports import RestaurantEvidenceProvider


async def plan_itinerary_dining(
    query: str,
    day_count: int,
    *,
    evidence_provider: RestaurantEvidenceProvider,
) -> ItineraryDiningDraft:
    try:
        request = ItineraryPlanningRequest(query=query, day_count=day_count)
    except ValidationError:
        raise ValueError(
            "Provide a non-empty restaurant query and a positive day count."
        ) from None

    discovery = await evidence_provider.search_restaurants(request.query)
    return _assemble_dining_draft(request, discovery)


def _assemble_dining_draft(
    request: ItineraryPlanningRequest,
    discovery: RestaurantSearchEvidence,
) -> ItineraryDiningDraft:
    unique_restaurants: list[RestaurantEvidenceCandidate] = []
    seen_source_urls: set[str] = set()
    for restaurant in discovery.restaurants:
        if restaurant.source_url in seen_source_urls:
            continue
        seen_source_urls.add(restaurant.source_url)
        unique_restaurants.append(restaurant)

    selected = unique_restaurants[: request.day_count]
    alternatives = unique_restaurants[request.day_count :]
    days = [
        ItineraryDaySuggestion(
            day_number=day_number,
            restaurant=RestaurantEvidenceCandidate.model_validate(restaurant),
        )
        for day_number, restaurant in enumerate(selected, start=1)
    ]
    return ItineraryDiningDraft(
        query=request.query,
        requested_days=request.day_count,
        days=days,
        alternatives=alternatives,
        unfilled_days=request.day_count - len(days),
        research_answer=discovery.answer,
    )
