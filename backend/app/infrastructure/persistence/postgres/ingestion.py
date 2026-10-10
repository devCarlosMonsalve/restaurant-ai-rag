from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.domain.osm_place import osm_searchable_metadata_changed
from app.ingestion.application.ports import (
    DocumentChunkRecord,
    ImageEmbeddingRecord,
)
from app.models.document_chunk import DocumentChunk
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.open_data_sources import OSMRestaurant


class PostgresDocumentChunkStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_all(self, records: Sequence[DocumentChunkRecord]) -> None:
        self._session.add_all(
            DocumentChunk(
                document_id=record.document_id,
                source_name=record.source_name,
                chunk_index=record.chunk_index,
                content=record.content,
                embedding=record.embedding,
            )
            for record in records
        )
        try:
            self._session.commit()
        except SQLAlchemyError:
            self._session.rollback()
            raise


class PostgresImageEmbeddingStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, record: ImageEmbeddingRecord) -> UUID:
        image_path = record.image_path
        resolved_path = str(image_path)
        metadata = record.metadata
        existing = self._session.scalar(
            select(ImageEmbedding).where(ImageEmbedding.image_path == resolved_path)
        )
        if existing is None:
            image = ImageEmbedding(
                source_name=image_path.name,
                image_path=resolved_path,
                embedding=record.embedding,
                source_url=metadata.source_url if metadata else None,
                license_name=metadata.license_name if metadata else None,
                license_url=metadata.license_url if metadata else None,
                attribution=metadata.attribution if metadata else None,
                osm_place_id=metadata.osm_place_id if metadata else None,
            )
            self._session.add(image)
        else:
            existing.source_name = image_path.name
            existing.embedding = record.embedding
            image = existing
            if metadata is not None:
                image.source_url = metadata.source_url
                image.license_name = metadata.license_name
                image.license_url = metadata.license_url
                image.attribution = metadata.attribution
                image.osm_place_id = metadata.osm_place_id

        try:
            self._session.commit()
        except SQLAlchemyError:
            self._session.rollback()
            raise

        return image.id


def upsert_osm_place(
    restaurant: OSMRestaurant,
    session: Session,
) -> OsmPlace:
    place = session.scalar(
        select(OsmPlace).where(
            OsmPlace.osm_type == restaurant.osm_type,
            OsmPlace.osm_id == restaurant.osm_id,
        )
    )
    if place is None:
        place = OsmPlace(
            osm_type=restaurant.osm_type,
            osm_id=restaurant.osm_id,
            name=restaurant.name,
            city=restaurant.city,
            cuisine=restaurant.cuisine,
            location=restaurant.location,
            latitude=restaurant.latitude,
            longitude=restaurant.longitude,
            features=list(restaurant.features),
            wikimedia_commons=restaurant.wikimedia_commons,
            source_url=restaurant.source_url,
        )
        session.add(place)
    else:
        searchable_metadata_changed = osm_searchable_metadata_changed(
            current_name=place.name,
            current_city=place.city,
            current_cuisine=place.cuisine,
            current_location=place.location,
            current_features=place.features,
            new_name=restaurant.name,
            new_city=restaurant.city,
            new_cuisine=restaurant.cuisine,
            new_location=restaurant.location,
            new_features=restaurant.features,
        )
        place.name = restaurant.name
        place.city = restaurant.city
        place.cuisine = restaurant.cuisine
        place.location = restaurant.location
        place.latitude = restaurant.latitude
        place.longitude = restaurant.longitude
        place.features = list(restaurant.features)
        place.wikimedia_commons = restaurant.wikimedia_commons
        place.source_url = restaurant.source_url
        if searchable_metadata_changed:
            place.embedding = None

    session.flush()
    return place
