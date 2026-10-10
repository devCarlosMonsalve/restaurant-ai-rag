from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.embeddings.text import embed_document_chunks
from app.infrastructure.embeddings.image import embed_image
from app.infrastructure.filesystem.text_documents import ingest_txt
from app.infrastructure.persistence.postgres.ingestion import (
    PostgresDocumentChunkStore,
    PostgresImageEmbeddingStore,
    upsert_osm_place as postgres_upsert_osm_place,
)
from app.ingestion.application.documents_usecase import ingest_text_document
from app.ingestion.application.images_usecase import ingest_image
from app.ingestion.application.ports import ImageSourceMetadata
from app.models.osm_place import OsmPlace
from app.infrastructure.external_data.open_data_sources import OSMRestaurant


def ingest_txt_to_database(
    path: str | Path,
    session: Session,
    *,
    chunk_size_words: int = 500,
    overlap_words: int = 75,
) -> tuple[UUID, int]:
    return ingest_text_document(
        path,
        reader=ingest_txt,
        embedder=embed_document_chunks,
        store=PostgresDocumentChunkStore(session),
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )


def ingest_image_to_database(
    path: str | Path,
    session: Session,
    *,
    metadata: ImageSourceMetadata | None = None,
) -> UUID:
    return ingest_image(
        path,
        embedder=embed_image,
        store=PostgresImageEmbeddingStore(session),
        metadata=metadata,
    )


def upsert_osm_place(
    restaurant: OSMRestaurant,
    session: Session,
) -> OsmPlace:
    return postgres_upsert_osm_place(restaurant, session)


__all__ = [
    "ImageSourceMetadata",
    "ingest_image_to_database",
    "ingest_txt_to_database",
    "upsert_osm_place",
]
