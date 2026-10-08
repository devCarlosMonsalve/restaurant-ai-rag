from sqlalchemy.orm import Session

from app.application.knowledge import search_documents as search_documents_use_case
from app.schemas import DocumentSearchRequest, DocumentSearchResult


def search_documents(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> list[DocumentSearchResult]:
    """Retrieve ranked document chunks using the existing search service."""
    request = DocumentSearchRequest(query=query, top_k=top_k)
    return search_documents_use_case(request, session)
