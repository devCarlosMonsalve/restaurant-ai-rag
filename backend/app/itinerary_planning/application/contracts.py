from pydantic import BaseModel, ConfigDict, Field


class ItineraryPlanningRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    day_count: int = Field(ge=1)


class RestaurantEvidenceCandidate(BaseModel):
    name: str
    city: str
    cuisine: str | None
    location: str | None
    latitude: float | None
    longitude: float | None
    features: list[str] = Field(default_factory=list)
    source_url: str
    attribution: str
    attribution_url: str
    similarity: float


class RestaurantSearchEvidence(BaseModel):
    answer: str
    restaurants: list[RestaurantEvidenceCandidate] = Field(default_factory=list)


class ItineraryDaySuggestion(BaseModel):
    day_number: int
    restaurant: RestaurantEvidenceCandidate


class ItineraryDiningDraft(BaseModel):
    query: str
    requested_days: int
    days: list[ItineraryDaySuggestion]
    alternatives: list[RestaurantEvidenceCandidate]
    unfilled_days: int
    research_answer: str
    evidence_notice: str = (
        "Suggestions follow restaurant retrieval order; similarity is not "
        "confidence. Day numbers are placeholders; routes, opening hours, "
        "availability, and reservations are not verified."
    )
