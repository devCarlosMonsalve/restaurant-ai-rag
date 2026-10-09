from sqlalchemy.orm import Session

from app.application.restaurant_discovery import search_restaurant_photos
from app.document_search import search_document_chunks
from app.rag import answer_with_rag
from app.schemas import (
    DocumentSearchRequest,
    DocumentSearchResult,
    ImageSearchRequest,
    RagAnswerResponse,
    RagAnswerWithPhotosResponse,
    RagQuestionRequest,
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
    request: RagQuestionRequest,
    session: Session,
) -> RagAnswerWithPhotosResponse:
    answer = answer_from_documents(request, session)
    photos = search_restaurant_photos(
        ImageSearchRequest(
            query=request.query,
            top_k=request.top_k,
        ),
        session,
    )
    return RagAnswerWithPhotosResponse(
        answer=answer.answer,
        sources=answer.sources,
        photos=photos,
    )
