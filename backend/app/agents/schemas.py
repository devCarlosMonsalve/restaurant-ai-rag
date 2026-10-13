from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class RestaurantSearchAgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)


class RestaurantSearchAgentPhoto(BaseModel):
    image_url: str
    source_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    restaurant_name: str | None = None
    restaurant_location: str | None = None
    restaurant_cuisine: str | None = None
    restaurant_source_url: str | None = None
    restaurant_attribution: str | None = None
    restaurant_attribution_url: str | None = None


class RestaurantSearchAgentCandidate(BaseModel):
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


class ReviewRecommendationToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    preferences: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        min_length=1,
        max_length=8,
    )


class ReviewCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    city: str = Field(min_length=1, max_length=100)
    cuisine: str | None = Field(default=None, max_length=255)
    location: str | None = Field(default=None, max_length=512)
    features: list[str] = Field(default_factory=list, max_length=100)
    source_url: str = Field(min_length=1, max_length=2048)
    attribution: str = Field(min_length=1, max_length=255)
    attribution_url: str = Field(min_length=1, max_length=2048)


class ReviewRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    preferences: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        min_length=1,
        max_length=8,
    )
    candidates: list[ReviewCandidate] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_candidate_sources(self) -> "ReviewRecommendationRequest":
        source_urls = [candidate.source_url for candidate in self.candidates]
        if len(source_urls) != len(set(source_urls)):
            raise ValueError("Candidate source URLs must be unique")
        return self


class CandidateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    field: Literal["name", "city", "cuisine", "location", "features"]
    value: str = Field(min_length=1, max_length=512)
    preference: str = Field(min_length=1, max_length=120)


class CandidateComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_source_url: str
    assessment: Literal["matches", "partially_matches", "insufficient_evidence"]
    reasons: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(
        min_length=1,
        max_length=5,
    )
    evidence: list[CandidateEvidence] = Field(max_length=10)
    unresolved_preferences: list[
        Annotated[str, Field(min_length=1, max_length=120)]
    ] = Field(max_length=8)


class UnattributedDocumentSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_name: str = Field(min_length=1, max_length=255)
    chunk_index: int = Field(ge=0)


class ReviewRecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    recommended_candidate_source_url: str | None = None
    comparisons: list[CandidateComparison] = Field(min_length=1, max_length=5)
    unattributed_documents: list[UnattributedDocumentSource] = Field(
        default_factory=list,
        max_length=5,
    )
    limitations: list[Annotated[str, Field(min_length=1, max_length=400)]] = Field(
        min_length=1,
        max_length=8,
    )


class RestaurantSearchAgentResponse(BaseModel):
    answer: str
    photos: list[RestaurantSearchAgentPhoto] = Field(default_factory=list)
    restaurants: list[RestaurantSearchAgentCandidate] = Field(default_factory=list)
    review_recommendation: ReviewRecommendationResponse | None = None

    @model_serializer(mode="wrap")
    def omit_missing_recommendation(
        self,
        handler: Any,
    ) -> dict[str, Any]:
        serialized = handler(self)
        if self.review_recommendation is None:
            serialized.pop("review_recommendation", None)
        return serialized
