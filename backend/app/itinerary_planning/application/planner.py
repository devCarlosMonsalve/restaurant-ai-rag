import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agents.schemas import (
    RestaurantSearchAgentCandidate,
    RestaurantSearchAgentResponse,
)
from app.itinerary_planning.infrastructure.a2a.restaurant_discovery_client import (
    DEFAULT_RESTAURANT_AGENT_URL,
    delegate_restaurant_discovery,
)


class ItineraryPlanningRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    day_count: int = Field(ge=1)


class ItineraryDaySuggestion(BaseModel):
    day_number: int
    restaurant: RestaurantSearchAgentCandidate


class ItineraryDiningDraft(BaseModel):
    query: str
    requested_days: int
    days: list[ItineraryDaySuggestion]
    alternatives: list[RestaurantSearchAgentCandidate]
    unfilled_days: int
    research_answer: str
    evidence_notice: str = (
        "Suggestions follow restaurant retrieval order; similarity is not "
        "confidence. Day numbers are placeholders; routes, opening hours, "
        "availability, and reservations are not verified."
    )


async def plan_itinerary_dining(
    query: str,
    day_count: int,
    *,
    agent_url: str = DEFAULT_RESTAURANT_AGENT_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ItineraryDiningDraft:
    try:
        request = ItineraryPlanningRequest(query=query, day_count=day_count)
    except ValidationError:
        raise ValueError(
            "Provide a non-empty restaurant query and a positive day count."
        ) from None

    discovery = await delegate_restaurant_discovery(
        request.query,
        agent_url=agent_url,
        transport=transport,
    )
    return _assemble_dining_draft(request, discovery)


def _assemble_dining_draft(
    request: ItineraryPlanningRequest,
    discovery: RestaurantSearchAgentResponse,
) -> ItineraryDiningDraft:
    selected = discovery.restaurants[: request.day_count]
    alternatives = discovery.restaurants[request.day_count :]
    days = [
        ItineraryDaySuggestion(
            day_number=day_number,
            restaurant=restaurant,
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
