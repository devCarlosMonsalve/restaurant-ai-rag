from typing import Protocol

from app.schemas import DocumentSearchResult


class DocumentRetriever(Protocol):
    def search_documents(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[DocumentSearchResult]: ...


class GroundedAnswerGenerator(Protocol):
    def __call__(
        self,
        question: str,
        context: str,
        /,
        *,
        context_chunk_count: int,
    ) -> str: ...
