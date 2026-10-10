from collections.abc import Sequence

from app.knowledge.application.ports import (
    DocumentRetriever,
    GroundedAnswerGenerator,
)
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
    chunks = retriever.search_documents(
        request.query,
        top_k=request.top_k,
    )
    if not chunks:
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
    return RagAnswerResponse(answer=answer, sources=sources)
