from sqlalchemy.orm import Session

from app.document_search import search_document_chunks
from app.rag import answer_with_rag
from app.schemas import (
    DocumentSearchRequest,
    DocumentSearchResult,
    RagAnswerResponse,
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
