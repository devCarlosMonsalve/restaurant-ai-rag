from sqlalchemy.orm import Session

from app.document_search import search_document_chunks
from app.schemas import DocumentSearchRequest, DocumentSearchResult


def search_documents(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> list[DocumentSearchResult]:
    """Retrieve ranked document chunks using the existing search service."""
    request = DocumentSearchRequest(query=query, top_k=top_k)
    return search_document_chunks(
        request.query,
        session,
        top_k=request.top_k,
    )
