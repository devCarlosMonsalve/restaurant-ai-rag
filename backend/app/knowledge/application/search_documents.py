from app.knowledge.application.ports import DocumentRetriever
from app.knowledge.application.contracts import (
    DocumentSearchRequest,
    DocumentSearchResult,
)


def search_documents(
    request: DocumentSearchRequest,
    retriever: DocumentRetriever,
) -> list[DocumentSearchResult]:
    return retriever.search_documents(
        request.query,
        top_k=request.top_k,
    )
