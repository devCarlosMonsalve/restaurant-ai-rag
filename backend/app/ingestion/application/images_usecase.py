from pathlib import Path
from uuid import UUID

from app.ingestion.application.ports import (
    ImageEmbedder,
    ImageEmbeddingRecord,
    ImageEmbeddingStore,
    ImageSourceMetadata,
)


def ingest_image(
    path: str | Path,
    *,
    embedder: ImageEmbedder,
    store: ImageEmbeddingStore,
    metadata: ImageSourceMetadata | None = None,
) -> UUID:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if len(image_path.name) > 255:
        raise ValueError("The image filename cannot exceed 255 characters")

    embedding = embedder(image_path)
    return store.upsert(
        ImageEmbeddingRecord(
            image_path=image_path,
            embedding=embedding,
            metadata=metadata,
        )
    )
