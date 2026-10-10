from pathlib import Path
from uuid import UUID, uuid4

from app.ingestion.application.ports import (
    DocumentChunkEmbedder,
    DocumentChunkRecord,
    DocumentChunkStore,
    TextChunkReader,
)


def ingest_text_document(
    path: str | Path,
    *,
    reader: TextChunkReader,
    embedder: DocumentChunkEmbedder,
    store: DocumentChunkStore,
    chunk_size_words: int = 500,
    overlap_words: int = 75,
) -> tuple[UUID, int]:
    text_file = Path(path)
    if len(text_file.name) > 255:
        raise ValueError("The source filename cannot exceed 255 characters")

    chunks = reader(
        text_file,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )
    if not chunks:
        raise ValueError("The text file contains no content")

    embeddings = embedder(chunks, document_title=text_file.stem)
    document_id = uuid4()
    records = [
        DocumentChunkRecord(
            document_id=document_id,
            source_name=text_file.name,
            chunk_index=index,
            content=chunk,
            embedding=embedding,
        )
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    store.save_all(records)
    return document_id, len(chunks)
