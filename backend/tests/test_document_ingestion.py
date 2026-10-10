from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingestion import ingest_txt_to_database
from app.models.document_chunk import DocumentChunk


def test_ingest_txt_embeds_and_stores_chunks(
    tmp_path: Path,
    db_session: Session,
    monkeypatch,
) -> None:
    text_file = tmp_path / "menu.txt"
    text_file.write_text("uno dos tres cuatro cinco seis siete", encoding="utf-8")

    monkeypatch.setattr(
        "app.ingestion.application.documents.embed_document_chunks",
        lambda chunks, document_title: [[0.1] * 768 for _ in chunks],
    )

    document_id, chunk_count = ingest_txt_to_database(
        text_file,
        db_session,
        chunk_size_words=3,
        overlap_words=1,
    )

    stored_chunks = db_session.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    ).all()

    assert chunk_count == 3
    assert [chunk.content for chunk in stored_chunks] == [
        "uno dos tres",
        "tres cuatro cinco",
        "cinco seis siete",
    ]
    assert all(len(chunk.embedding) == 768 for chunk in stored_chunks)
    assert all(chunk.source_name == "menu.txt" for chunk in stored_chunks)
