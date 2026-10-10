import argparse

from sqlalchemy import select

from app.infrastructure.persistence.postgres.database import SessionLocal
from app.infrastructure.embeddings.text import embed_document_chunks
from app.models.osm_place import OsmPlace


def _embedding_text(place: OsmPlace) -> str:
    features = " ".join(place.features or [])
    return (
        f"Nombre: {place.name}. Cocina: {place.cuisine or 'no especificada'}. "
        f"Ubicación: {place.location or 'no especificada'}. Ciudad: {place.city}. "
        f"Características: {features or 'no especificadas'}."
    )


def index_missing_place_embeddings(*, batch_size: int) -> int:
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")

    indexed_count = 0
    with SessionLocal() as session:
        while True:
            places = session.scalars(
                select(OsmPlace)
                .where(OsmPlace.embedding.is_(None))
                .order_by(OsmPlace.osm_type, OsmPlace.osm_id)
                .limit(batch_size)
            ).all()
            if not places:
                break

            vectors = embed_document_chunks(
                [_embedding_text(place) for place in places],
                document_title="restaurante de OpenStreetMap",
            )
            if len(vectors) != len(places):
                raise RuntimeError("Embedding count does not match place count")
            for place, vector in zip(places, vectors, strict=True):
                place.embedding = vector
            session.commit()
            indexed_count += len(places)
            print(f"Indexed embeddings for {indexed_count} OSM restaurants.")

    return indexed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate semantic search embeddings for OSM restaurants."
    )
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")

    indexed_count = index_missing_place_embeddings(batch_size=args.batch_size)
    print(f"Indexed {indexed_count} OSM restaurant embeddings.")


if __name__ == "__main__":
    main()
