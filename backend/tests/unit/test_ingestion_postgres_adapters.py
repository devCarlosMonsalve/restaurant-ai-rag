from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.ingestion.application.ports import (
    DocumentChunkRecord,
    ImageEmbeddingRecord,
)
from app.infrastructure.persistence.postgres.ingestion import (
    PostgresDocumentChunkStore,
    PostgresImageEmbeddingStore,
)


def test_document_store_rolls_back_and_reraises_commit_errors() -> None:
    session = MagicMock()
    error = SQLAlchemyError("commit failed")
    session.commit.side_effect = error
    store = PostgresDocumentChunkStore(session)
    record = DocumentChunkRecord(
        document_id=uuid4(),
        source_name="menu.txt",
        chunk_index=0,
        content="menu",
        embedding=[0.1],
    )

    with pytest.raises(SQLAlchemyError) as raised:
        store.save_all([record])

    assert raised.value is error
    session.add_all.assert_called_once()
    session.rollback.assert_called_once_with()


def test_image_store_rolls_back_and_reraises_commit_errors(tmp_path: Path) -> None:
    session = MagicMock()
    session.scalar.return_value = None
    error = SQLAlchemyError("commit failed")
    session.commit.side_effect = error
    store = PostgresImageEmbeddingStore(session)
    record = ImageEmbeddingRecord(
        image_path=tmp_path / "photo.jpg",
        embedding=[0.1],
        metadata=None,
    )

    with pytest.raises(SQLAlchemyError) as raised:
        store.upsert(record)

    assert raised.value is error
    session.add.assert_called_once()
    session.rollback.assert_called_once_with()
