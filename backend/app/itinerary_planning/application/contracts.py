import math

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _coerce_integral_day_number(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("Day numbers and counts must be integers")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise ValueError("Day numbers and counts must be integers")


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
    day_number: int = Field(strict=True, ge=1)
    restaurant: RestaurantEvidenceCandidate

    @field_validator("day_number", mode="before")
    @classmethod
    def validate_day_number(cls, value: object) -> int:
        return _coerce_integral_day_number(value)


class ItineraryDiningDraft(BaseModel):
    query: str
    requested_days: int = Field(strict=True, ge=1)
    days: list[ItineraryDaySuggestion]
    alternatives: list[RestaurantEvidenceCandidate]
    unfilled_days: int = Field(strict=True, ge=0)
    research_answer: str
    evidence_notice: str = (
        "Suggestions follow restaurant retrieval order; similarity is not "
        "confidence. Day numbers are placeholders; routes, opening hours, "
        "availability, and reservations are not verified."
    )

    @field_validator(
        "requested_days",
        "unfilled_days",
        mode="before",
    )
    @classmethod
    def validate_day_counts(cls, value: object) -> int:
        return _coerce_integral_day_number(value)

    @model_validator(mode="after")
    def validate_day_assignments(self) -> "ItineraryDiningDraft":
        if len(self.days) > self.requested_days:
            raise ValueError("The number of assigned days exceeds requested_days")

        expected_day_numbers = list(range(1, len(self.days) + 1))
        actual_day_numbers = [day.day_number for day in self.days]
        if actual_day_numbers != expected_day_numbers:
            raise ValueError("Assigned day numbers must be consecutive from 1")

        if self.unfilled_days != self.requested_days - len(self.days):
            raise ValueError("unfilled_days must match the unassigned requested days")

        restaurant_urls = [
            day.restaurant.source_url for day in self.days
        ] + [candidate.source_url for candidate in self.alternatives]
        if len(restaurant_urls) != len(set(restaurant_urls)):
            raise ValueError(
                "A restaurant cannot appear more than once in a dining draft"
            )

        return self
