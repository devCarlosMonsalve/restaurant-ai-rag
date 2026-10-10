import argparse
import re
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.ingestion import ImageSourceMetadata, ingest_image_to_database, upsert_osm_place
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.open_data_sources import (
    OSMRestaurant,
    CommonsPhoto,
    download_commons_photo,
    fetch_commons_photos,
    fetch_madrid_restaurants,
)

IMAGES_DIR = Path(__file__).parent / "data" / "images" / "commons"


def _safe_filename(restaurant: OSMRestaurant, photo: CommonsPhoto) -> str:
    title = photo.title.removeprefix("File:")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(title).stem).strip("._")
    filename = f"osm-{restaurant.osm_type}-{restaurant.osm_id}-{stem}"
    return f"{filename[:240]}{photo.file_extension}"


def _validate_image(content: bytes, photo: CommonsPhoto) -> None:
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError(f"Unsupported image format returned for {photo.title}")
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Invalid image returned for {photo.title}") from error


def _store_photo(
    restaurant: OSMRestaurant,
    place: OsmPlace,
    photo: CommonsPhoto,
    session: Session,
) -> bool:
    image_path = IMAGES_DIR / _safe_filename(restaurant, photo)
    existing_image = session.scalar(
        select(ImageEmbedding).where(ImageEmbedding.image_path == str(image_path))
    )
    if existing_image is not None and image_path.is_file():
        existing_image.source_url = photo.source_url
        existing_image.license_name = photo.license_name
        existing_image.license_url = photo.license_url
        existing_image.attribution = photo.attribution
        existing_image.osm_place_id = place.id
        return False

    content = download_commons_photo(photo)
    _validate_image(content, photo)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(content)

    ingest_image_to_database(
        image_path,
        session,
        metadata=ImageSourceMetadata(
            source_url=photo.source_url,
            license_name=photo.license_name,
            license_url=photo.license_url,
            attribution=photo.attribution,
            osm_place_id=place.id,
        ),
    )
    print(
        f"Indexed {photo.title} for {restaurant.name} "
        f"({photo.license_name}; attribution: {photo.attribution or 'not required'})"
    )
    return True


def import_madrid_images(*, limit: int | None, photos_per_place: int) -> int:
    restaurants = fetch_madrid_restaurants(limit=limit)
    if not restaurants:
        print("No Madrid restaurants were found.")
        return 0

    indexed_count = 0
    commons_request_count = 0
    with SessionLocal() as session:
        for index, restaurant in enumerate(restaurants, start=1):
            if index == 1 or index % 500 == 0:
                print(f"Importing OSM restaurant {index} of {len(restaurants)}.")
            place = upsert_osm_place(restaurant, session)
            if not restaurant.wikimedia_commons:
                continue
            if not restaurant.wikimedia_commons.startswith(("Category:", "File:")):
                print("Skipped: OSM Commons reference is not a Category: or File: link.")
                continue

            if commons_request_count:
                time.sleep(1)
            commons_request_count += 1
            photos = fetch_commons_photos(
                restaurant.wikimedia_commons,
                limit=photos_per_place,
            )
            if not photos:
                print("No images with an approved reusable license were found.")
                continue

            for photo in photos:
                if _store_photo(restaurant, place, photo, session):
                    indexed_count += 1

        try:
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise

    print(
        f"Imported {len(restaurants)} OSM restaurants; "
        f"indexed {indexed_count} new Commons images."
    )
    return indexed_count


def main() -> None:
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description=(
            "Import licensed Wikimedia Commons photos linked from Madrid "
            "OpenStreetMap restaurants and index them with CLIP."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Madrid restaurants to import (default: all)",
    )
    parser.add_argument("--photos-per-place", type=int, default=2)
    args = parser.parse_args()

    if args.limit is not None and not 1 <= args.limit <= 10000:
        parser.error("--limit must be between 1 and 10000")
    if not 1 <= args.photos_per_place <= 10:
        parser.error("--photos-per-place must be between 1 and 10")

    count = import_madrid_images(
        limit=args.limit,
        photos_per_place=args.photos_per_place,
    )
    print(f"Indexed {count} Commons images.")


if __name__ == "__main__":
    main()
