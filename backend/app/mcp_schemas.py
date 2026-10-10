from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.restaurant_discovery.application.contracts import OsmRestaurantSearchResponse
from app.knowledge.application.contracts import RagAnswerResponse


class McpRestaurantResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    city: str
    cuisine: str | None
    location: str | None
    latitude: float | None
    longitude: float | None
    features: list[str]
    source_url: str
    attribution: str
    attribution_url: str
    similarity: float


class McpRestaurantSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[McpRestaurantResult]
    evidence_status: Literal[
        "not_required",
        "verified",
        "partial",
        "no_evidence",
        "stale_evidence",
        "unverified",
    ]
    evidence_message: str | None

    @classmethod
    def from_search_response(
        cls,
        response: OsmRestaurantSearchResponse,
    ) -> "McpRestaurantSearchResponse":
        return cls(
            results=[
                McpRestaurantResult(
                    **result.model_dump(exclude={"id"})
                )
                for result in response.results
            ],
            evidence_status=response.evidence_status,
            evidence_message=response.evidence_message,
        )


class McpPhotoResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    similarity: float
    source_url: str | None
    license_name: str | None
    license_url: str | None
    attribution: str | None
    restaurant_name: str | None
    restaurant_location: str | None
    restaurant_cuisine: str | None
    restaurant_features: list[str]
    restaurant_source_url: str | None
    restaurant_attribution: str | None
    restaurant_attribution_url: str | None


class McpPhotoSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[McpPhotoResult]


class McpDocumentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    chunk_index: int
    content: str
    similarity: float


class McpDocumentSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[McpDocumentResult]


class McpRagSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str
    chunk_index: int
    similarity: float


class McpRagAnswerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: list[McpRagSource]

    @classmethod
    def from_rag_response(cls, response: RagAnswerResponse) -> "McpRagAnswerResponse":
        return cls(
            answer=response.answer,
            sources=[
                McpRagSource(
                    source_name=source.source_name,
                    chunk_index=source.chunk_index,
                    similarity=source.similarity,
                )
                for source in response.sources
            ],
        )
