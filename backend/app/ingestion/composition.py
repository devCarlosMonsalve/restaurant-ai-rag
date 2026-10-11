from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.infrastructure.embeddings.text import embed_document_chunks
from app.infrastructure.embeddings.image import embed_image
from app.infrastructure.external_data.open_data_sources import OSMRestaurant
from app.infrastructure.filesystem.text_documents import ingest_txt
from app.infrastructure.persistence.postgres.ingestion import (
    PostgresDocumentChunkStore,
    PostgresImageEmbeddingStore,
    PostgresOSMPlaceStore,
)
from app.ingestion.application.documents_usecase import ingest_text_document
from app.ingestion.application.images_usecase import ingest_image
from app.ingestion.application.osm import (
    upsert_osm_place as upsert_osm_place_usecase,
)
from app.ingestion.application.ports import ImageSourceMetadata, OSMPlaceRecord
from app.models.osm_place import OsmPlace


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
    record = OSMPlaceRecord(
        osm_type=restaurant.osm_type,
        osm_id=restaurant.osm_id,
        name=restaurant.name,
        city=restaurant.city,
        cuisine=restaurant.cuisine,
        location=restaurant.location,
        latitude=restaurant.latitude,
        longitude=restaurant.longitude,
        wikimedia_commons=restaurant.wikimedia_commons,
        features=restaurant.features,
        source_url=restaurant.source_url,
    )
    place_id = upsert_osm_place_usecase(
        record,
        store=PostgresOSMPlaceStore(session),
    )
    place = session.get(OsmPlace, place_id)
    if place is None:
        raise RuntimeError("The OSM place was not available after its upsert")
    return place


__all__ = [
    "ImageSourceMetadata",
    "ingest_image_to_database",
    "ingest_txt_to_database",
    "upsert_osm_place",
]
