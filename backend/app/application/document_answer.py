from pydantic import BaseModel, Field

from app.restaurant_discovery.application.ports import RestaurantDiscoveryPort
from app.restaurant_discovery.application.service import search_restaurant_photos
from app.presentation.image_urls import image_file_url
from app.knowledge.application.answer_question import answer_from_documents
from app.knowledge.application.ports import (
    DocumentRetriever,
    GroundedAnswerGenerator,
)
from app.infrastructure.observability import traced_span
from app.restaurant_discovery.application.contracts import ImageSearchRequest
from app.knowledge.application.contracts import (
    RagAnswerResponse,
    RagQuestionRequest,
)


class RagQuestionWithPhotosRequest(RagQuestionRequest):
    include_photos: bool = False


class RagPhoto(BaseModel):
    source_name: str
    image_url: str
    similarity: float
    metadata_match_count: int = 0
    source_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    restaurant_name: str | None = None
    restaurant_location: str | None = None
    restaurant_cuisine: str | None = None
    restaurant_features: list[str] = Field(default_factory=list)
    restaurant_source_url: str | None = None
    restaurant_attribution: str | None = None
    restaurant_attribution_url: str | None = None


class RagAnswerWithPhotosResponse(RagAnswerResponse):
    photos: list[RagPhoto] = Field(default_factory=list)


def answer_from_documents_with_photos(
    request: RagQuestionWithPhotosRequest,
    retriever: DocumentRetriever,
    generate_answer: GroundedAnswerGenerator,
    restaurant_repository: RestaurantDiscoveryPort,
) -> RagAnswerWithPhotosResponse:
    with traced_span(
        "application.answer_with_document_photos",
        {
            "retrieval.top_k": request.top_k,
            "photos.enabled": request.include_photos,
        },
    ) as span:
        with traced_span(
            "rag.answer_from_documents",
            {"retrieval.top_k": request.top_k},
        ) as rag_span:
            answer = answer_from_documents(
                request,
                retriever,
                generate_answer,
            )
            rag_span.set_attribute(
                "rag.retrieved_chunk_count",
                len(answer.sources),
            )
            if not answer.sources:
                rag_span.set_attribute("rag.result", "no_documents")
            else:
                rag_span.set_attribute("rag.source_count", len(answer.sources))
        photos = []
        if request.include_photos:
            image_results = search_restaurant_photos(
                ImageSearchRequest(
                    query=request.query,
                    top_k=request.top_k,
                    osm_places_only=True,
                ),
                restaurant_repository,
            )
            photos = [
                RagPhoto(
                    source_name=photo.source_name,
                    image_url=image_file_url(photo.image_path),
                    similarity=photo.similarity,
                    metadata_match_count=photo.metadata_match_count,
                    source_url=photo.source_url,
                    license_name=photo.license_name,
                    license_url=photo.license_url,
                    attribution=photo.attribution,
                    restaurant_name=photo.restaurant_name,
                    restaurant_location=photo.restaurant_location,
                    restaurant_cuisine=photo.restaurant_cuisine,
                    restaurant_features=photo.restaurant_features,
                    restaurant_source_url=photo.restaurant_source_url,
                    restaurant_attribution=photo.restaurant_attribution,
                    restaurant_attribution_url=photo.restaurant_attribution_url,
                )
                for photo in image_results
            ]
            span.set_attribute("photos.result_count", len(photos))
        return RagAnswerWithPhotosResponse(
            answer=answer.answer,
            sources=answer.sources,
            photos=photos,
        )
