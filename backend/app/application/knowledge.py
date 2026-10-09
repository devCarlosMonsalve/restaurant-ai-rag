from sqlalchemy.orm import Session

from app.application.restaurant_discovery import search_restaurant_photos
from app.document_search import search_document_chunks
from app.image_presentation import image_file_url
from app.observability import traced_span
from app.rag import answer_with_rag
from app.schemas import (
    DocumentSearchRequest,
    DocumentSearchResult,
    ImageSearchRequest,
    RagPhoto,
    RagAnswerResponse,
    RagAnswerWithPhotosResponse,
    RagQuestionRequest,
    RagQuestionWithPhotosRequest,
)


def search_documents(
    request: DocumentSearchRequest,
    session: Session,
) -> list[DocumentSearchResult]:
    return search_document_chunks(
        request.query,
        session,
        top_k=request.top_k,
    )


def answer_from_documents(
    request: RagQuestionRequest,
    session: Session,
) -> RagAnswerResponse:
    return answer_with_rag(request, session)


def answer_from_documents_with_photos(
    request: RagQuestionWithPhotosRequest,
    session: Session,
) -> RagAnswerWithPhotosResponse:
    with traced_span(
        "application.answer_with_document_photos",
        {
            "retrieval.top_k": request.top_k,
            "photos.enabled": request.include_photos,
        },
    ) as span:
        answer = answer_from_documents(request, session)
        photos = []
        if request.include_photos:
            image_results = search_restaurant_photos(
                ImageSearchRequest(
                    query=request.query,
                    top_k=request.top_k,
                    osm_places_only=True,
                ),
                session,
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
