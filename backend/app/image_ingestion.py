from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.image_embeddings import embed_image
from app.models.image_embedding import ImageEmbedding


def ingest_image_to_database(path: str | Path, session: Session) -> UUID:
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
        )
        session.add(image)
    else:
        existing.source_name = image_path.name
        existing.embedding = embedding
        image = existing

    try:
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        raise

    return image.id
