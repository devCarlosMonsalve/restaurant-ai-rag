from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.image_embeddings import embed_image
from app.models.image_embedding import ImageEmbedding


@dataclass(frozen=True)
class ImageSourceMetadata:
    source_url: str
    license_name: str
    license_url: str
    attribution: str
    osm_place_id: UUID


def ingest_image_to_database(
    path: str | Path,
    session: Session,
    *,
    metadata: ImageSourceMetadata | None = None,
) -> UUID:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if len(image_path.name) > 255:
        raise ValueError("The image filename cannot exceed 255 characters")

    embedding = embed_image(image_path)
    existing = session.scalar(
        select(ImageEmbedding).where(ImageEmbedding.image_path == str(image_path))
    )
    if existing is None:
        image = ImageEmbedding(
            source_name=image_path.name,
            image_path=str(image_path),
            embedding=embedding,
            source_url=metadata.source_url if metadata else None,
            license_name=metadata.license_name if metadata else None,
            license_url=metadata.license_url if metadata else None,
            attribution=metadata.attribution if metadata else None,
            osm_place_id=metadata.osm_place_id if metadata else None,
        )
        session.add(image)
    else:
        existing.source_name = image_path.name
        existing.embedding = embedding
        image = existing
        if metadata is not None:
            image.source_url = metadata.source_url
            image.license_name = metadata.license_name
            image.license_url = metadata.license_url
            image.attribution = metadata.attribution
            image.osm_place_id = metadata.osm_place_id

    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise

    return image.id
