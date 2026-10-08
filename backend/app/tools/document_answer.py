from sqlalchemy.orm import Session

from app.rag import answer_with_rag
from app.schemas import RagAnswerResponse, RagQuestionRequest


def answer_from_documents(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> RagAnswerResponse:
    """Answer from indexed documents using the existing grounded RAG service."""
    request = RagQuestionRequest(query=query, top_k=top_k)
    return answer_with_rag(request, session)
