from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.embeddings.text import embed_search_query
from app.knowledge.application.ports import DocumentRetriever
from app.models.document_chunk import DocumentChunk
from app.infrastructure.observability import traced_span
from app.knowledge.application.contracts import DocumentSearchResult


class PostgresDocumentRetriever:
    def __init__(self, session: Session) -> None:
        self.session = session

    def search_documents(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[DocumentSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        with traced_span(
            "retrieval.document_chunks",
            {"retrieval.top_k": top_k},
        ) as span:
            query_embedding = embed_search_query(query)
            cosine_distance = DocumentChunk.embedding.cosine_distance(query_embedding)
            statement = (
                select(DocumentChunk, cosine_distance)
                .order_by(cosine_distance)
                .limit(top_k)
            )
            matches = self.session.execute(statement).all()
            results = [
                DocumentSearchResult(
                    document_id=chunk.document_id,
                    source_name=chunk.source_name,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    similarity=1.0 - float(distance),
                )
                for chunk, distance in matches
            ]
            span.set_attribute("retrieval.result_count", len(results))
            return results
