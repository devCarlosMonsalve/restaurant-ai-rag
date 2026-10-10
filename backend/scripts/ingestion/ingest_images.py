import argparse
from pathlib import Path

from app.infrastructure.persistence.postgres.database import SessionLocal
from app.infrastructure.embeddings.image import SUPPORTED_IMAGE_SUFFIXES
from app.ingestion import ingest_image_to_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and store CLIP embeddings for local images."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="An image file or a directory containing supported images",
    )
    args = parser.parse_args()
    image_path = args.path.expanduser()

    if image_path.is_dir():
        image_paths = sorted(
            path
            for path in image_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    elif image_path.is_file():
        image_paths = [image_path]
    else:
        raise FileNotFoundError(f"Image path not found: {image_path}")

    if not image_paths:
        raise ValueError(f"No supported images found at: {image_path}")

    with SessionLocal() as session:
        for path in image_paths:
            image_id = ingest_image_to_database(path, session)
            print(f"Indexed {path.name} ({image_id})")


if __name__ == "__main__":
    main()
