from sqlalchemy.orm import Session

from app.application.knowledge import search_documents as search_documents_use_case
from app.schemas import DocumentSearchRequest, DocumentSearchResult


def search_documents(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> list[DocumentSearchResult]:
    """Retrieve relevant excerpts from the indexed document corpus.

    Use this when source excerpts are needed for inspection, evidence
    composition, or a workflow that will perform its own synthesis. This is
    retrieval only: it returns ranked chunks and does not write a natural-
    language answer. To get a generated answer with citations, use
    ``answer_from_documents`` instead.

    Results only cover documents that have been indexed. Retrieval scores do
    not certify that a chunk is complete, current, or sufficient to answer the
    question. The host injects ``session``; it is infrastructure, not an
    Agent-selected input.

    Args:
        query: Question or topic to retrieve from indexed documents.
        session: Database session supplied by the host.
        top_k: Maximum number of document chunks, from 1 to 20.

    Returns:
        Ranked document excerpts with source filename, chunk index, and
        similarity.
    """
    request = DocumentSearchRequest(query=query, top_k=top_k)
    return search_documents_use_case(request, session)
