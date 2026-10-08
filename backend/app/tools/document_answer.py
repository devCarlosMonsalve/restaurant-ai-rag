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
    """Answer from indexed documents using the existing grounded RAG service."""
    request = RagQuestionRequest(query=query, top_k=top_k)
    return answer_from_documents_use_case(request, session)
