import argparse
import re
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.image_ingestion import ImageSourceMetadata, ingest_image_to_database
from app.models.osm_place import OsmPlace
from app.open_data_sources import (
    OSMRestaurant,
    CommonsPhoto,
    download_commons_photo,
    fetch_commons_photos,
    fetch_madrid_restaurants,
)
from app.osm_ingestion import upsert_osm_place

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
) -> None:
    content = download_commons_photo(photo)
    _validate_image(content, photo)

    image_path = IMAGES_DIR / _safe_filename(restaurant, photo)
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


def import_madrid_images(*, limit: int, photos_per_place: int) -> int:
    restaurants = fetch_madrid_restaurants(limit=limit)
    if not restaurants:
        print("No Madrid restaurants with Wikimedia Commons references were found.")
        return 0

    indexed_count = 0
    with SessionLocal() as session:
        for index, restaurant in enumerate(restaurants):
            if index:
                time.sleep(1)

            print(
                f"OSM: {restaurant.name} | "
                f"{restaurant.wikimedia_commons} | {restaurant.source_url}"
            )
            place = upsert_osm_place(restaurant, session)
            if not restaurant.wikimedia_commons.startswith(("Category:", "File:")):
                print("Skipped: OSM Commons reference is not a Category: or File: link.")
                continue

            photos = fetch_commons_photos(
                restaurant.wikimedia_commons,
                limit=photos_per_place,
            )
            if not photos:
                print("No images with an approved reusable license were found.")
                continue

            for photo in photos:
                _store_photo(restaurant, place, photo, session)
                indexed_count += 1

        try:
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise

    return indexed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Import licensed Wikimedia Commons photos linked from Madrid "
            "OpenStreetMap restaurants and index them with CLIP."
        )
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--photos-per-place", type=int, default=2)
    args = parser.parse_args()

    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    if not 1 <= args.photos_per_place <= 10:
        parser.error("--photos-per-place must be between 1 and 10")

    count = import_madrid_images(
        limit=args.limit,
        photos_per_place=args.photos_per_place,
    )
    print(f"Indexed {count} Commons images.")


if __name__ == "__main__":
    main()
