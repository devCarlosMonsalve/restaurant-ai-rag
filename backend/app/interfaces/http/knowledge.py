from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.application.document_answer import (
    answer_from_documents_with_photos as answer_documents_use_case,
)
from app.infrastructure.persistence.postgres.database import get_db
from app.interfaces.http.schemas import (
    RagAnswerWithPhotosResponse,
    RagQuestionWithPhotosRequest,
)
from app.knowledge.application.contracts import (
    DocumentSearchRequest,
    DocumentSearchResult,
)
from app.knowledge.application.search_documents import (
    search_documents as search_documents_use_case,
)
from app.knowledge.infrastructure.generation.answer_chain import (
    generate_grounded_answer,
)
from app.knowledge.infrastructure.postgres.retriever import (
    PostgresDocumentRetriever,
)
from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)

router = APIRouter()


@router.post(
    "/documents/search",
    response_model=list[DocumentSearchResult],
    tags=["documents"],
)
def search_documents(
    request: DocumentSearchRequest,
    db: Session = Depends(get_db),
) -> list[DocumentSearchResult]:
    return search_documents_use_case(request, PostgresDocumentRetriever(db))


@router.post(
    "/documents/ask",
    response_model=RagAnswerWithPhotosResponse,
    tags=["documents"],
)
def ask_documents(
    request: RagQuestionWithPhotosRequest,
    db: Session = Depends(get_db),
) -> RagAnswerWithPhotosResponse:
    return answer_documents_use_case(
        request,
        PostgresDocumentRetriever(db),
        generate_grounded_answer,
        PostgresRestaurantDiscoveryAdapter(db),
    )
