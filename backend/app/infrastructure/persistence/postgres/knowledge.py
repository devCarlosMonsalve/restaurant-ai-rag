from sqlalchemy.orm import Session

from app.document_search import search_document_chunks
from app.rag import answer_with_rag
from app.schemas import DocumentSearchResult, RagAnswerResponse, RagQuestionRequest


class PostgresKnowledgeAdapter:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search_documents(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[DocumentSearchResult]:
        return search_document_chunks(query, self.session, top_k=top_k)

    def answer_from_documents(
        self,
        request: RagQuestionRequest,
    ) -> RagAnswerResponse:
        return answer_with_rag(request, self.session)
