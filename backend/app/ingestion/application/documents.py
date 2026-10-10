from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.embeddings import embed_document_chunks
from app.infrastructure.filesystem.text_documents import ingest_txt
from app.models.document_chunk import DocumentChunk


def ingest_txt_to_database(
    path: str | Path,
    session: Session,
    *,
    chunk_size_words: int = 500,
    overlap_words: int = 75,
) -> tuple[UUID, int]:
    text_file = Path(path)
    if len(text_file.name) > 255:
        raise ValueError("The source filename cannot exceed 255 characters")

    chunks = ingest_txt(
        text_file,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )
    if not chunks:
        raise ValueError("The text file contains no content")

    embeddings = embed_document_chunks(chunks, document_title=text_file.stem)
    document_id = uuid4()
    session.add_all(
        DocumentChunk(
            document_id=document_id,
            source_name=text_file.name,
            chunk_index=index,
            content=chunk,
            embedding=embedding,
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    )

    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise

    return document_id, len(chunks)
