from sqlalchemy.orm import Session

from app.answer_generation import generate_grounded_answer
from app.document_search import search_document_chunks
from app.observability import traced_span
from app.schemas import RagAnswerResponse, RagQuestionRequest, RagSource

NO_DOCUMENTS_ANSWER = "No encontré documentos indexados para responder la pregunta."


def answer_with_rag(
    request: RagQuestionRequest,
    session: Session,
) -> RagAnswerResponse:
    with traced_span(
        "rag.answer_from_documents",
        {"retrieval.top_k": request.top_k},
    ) as span:
        chunks = search_document_chunks(
            request.query,
            session,
            top_k=request.top_k,
        )
        span.set_attribute("rag.retrieved_chunk_count", len(chunks))
        if not chunks:
            span.set_attribute("rag.result", "no_documents")
            return RagAnswerResponse(answer=NO_DOCUMENTS_ANSWER, sources=[])

        answer = generate_grounded_answer(request.query, chunks)
        sources = [
            RagSource(
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                chunk_index=chunk.chunk_index,
                similarity=chunk.similarity,
            )
            for chunk in chunks
        ]
        span.set_attribute("rag.source_count", len(sources))
        return RagAnswerResponse(answer=answer, sources=sources)
