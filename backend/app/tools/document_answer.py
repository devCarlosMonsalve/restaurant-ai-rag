from sqlalchemy.orm import Session

from app.application.knowledge import (
    answer_from_documents as answer_from_documents_use_case,
)
from app.schemas import RagAnswerResponse, RagQuestionRequest


def answer_from_documents(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> RagAnswerResponse:
    """Answer a question using retrieval-augmented generation over documents.

    Use this when the user wants a direct answer grounded in the indexed
    document corpus. The RAG service retrieves excerpts, generates an answer
    from them, and returns the answer with its source list. It is distinct from
    ``search_documents``, which returns raw ranked excerpts for inspection and
    further synthesis without generating an answer.

    The answer is limited to retrieved indexed content and can be incomplete
    or outdated if that corpus is incomplete or outdated. Sources identify
    supporting chunks but do not independently verify the documents. For
    restaurant candidates or photos, use the restaurant discovery Tools. The
    host injects ``session``; it is infrastructure, not an Agent-selected
    input.

    Args:
        query: Natural-language question to answer from indexed documents.
        session: Database session supplied by the host.
        top_k: Maximum number of source chunks considered, from 1 to 20.

    Returns:
        A generated answer and the source chunks used by the RAG service.
    """
    request = RagQuestionRequest(query=query, top_k=top_k)
    return answer_from_documents_use_case(request, session)
