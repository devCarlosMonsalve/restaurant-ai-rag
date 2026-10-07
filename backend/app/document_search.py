from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import embed_search_query
from app.models.document_chunk import DocumentChunk
from app.schemas import DocumentSearchResult


def search_document_chunks(
    query: str,
    session: Session,
    *,
    top_k: int = 5,
) -> list[DocumentSearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = embed_search_query(query)
    cosine_distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(DocumentChunk, cosine_distance)
        .order_by(cosine_distance)
        .limit(top_k)
    )
    matches = session.execute(statement).all()

    return [
        DocumentSearchResult(
            document_id=chunk.document_id,
            source_name=chunk.source_name,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            similarity=1.0 - float(distance),
        )
        for chunk, distance in matches
    ]
