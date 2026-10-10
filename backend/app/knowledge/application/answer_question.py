from collections.abc import Sequence

from app.knowledge.application.ports import (
    DocumentRetriever,
    GroundedAnswerGenerator,
)
from app.observability import traced_span
from app.knowledge.application.contracts import (
    DocumentSearchResult,
    RagAnswerResponse,
    RagQuestionRequest,
    RagSource,
)

NO_DOCUMENTS_ANSWER = "No encontré documentos indexados para responder la pregunta."


def build_retrieved_context(chunks: Sequence[DocumentSearchResult]) -> str:
    return "\n\n".join(
        f"[{chunk.source_name}#{chunk.chunk_index}]\n{chunk.content}"
        for chunk in chunks
    )


def answer_from_documents(
    request: RagQuestionRequest,
    retriever: DocumentRetriever,
    generate_answer: GroundedAnswerGenerator,
) -> RagAnswerResponse:
    with traced_span(
        "rag.answer_from_documents",
        {"retrieval.top_k": request.top_k},
    ) as span:
        chunks = retriever.search_documents(
            request.query,
            top_k=request.top_k,
        )
        span.set_attribute("rag.retrieved_chunk_count", len(chunks))
        if not chunks:
            span.set_attribute("rag.result", "no_documents")
            return RagAnswerResponse(answer=NO_DOCUMENTS_ANSWER, sources=[])

        answer = generate_answer(
            request.query,
            build_retrieved_context(chunks),
            context_chunk_count=len(chunks),
        )
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
