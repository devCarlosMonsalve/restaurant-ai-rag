from sqlalchemy.orm import Session

from app.answer_generation import generate_grounded_answer
from app.document_search import search_document_chunks
from app.schemas import RagAnswerResponse, RagQuestionRequest, RagSource

NO_DOCUMENTS_ANSWER = "No encontré documentos indexados para responder la pregunta."


def answer_with_rag(
    request: RagQuestionRequest,
    session: Session,
) -> RagAnswerResponse:
    chunks = search_document_chunks(
        request.query,
        session,
        top_k=request.top_k,
    )
    if not chunks:
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
    return RagAnswerResponse(answer=answer, sources=sources)
