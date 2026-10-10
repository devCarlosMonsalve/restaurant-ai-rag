from app.ingestion import (
    ImageSourceMetadata,
    ingest_image_to_database,
    ingest_txt_to_database,
    upsert_osm_place,
)
from app.ingestion.application.documents import (
    ingest_txt_to_database as documents_facade,
)
from app.ingestion.application.images import (
    ImageSourceMetadata as ImagesSourceMetadata,
    ingest_image_to_database as images_facade,
)
from app.ingestion.application.osm import upsert_osm_place as osm_facade


def test_ingestion_package_exports_compatibility_facades() -> None:
    assert ingest_txt_to_database is documents_facade
    assert ingest_image_to_database is images_facade
    assert ImageSourceMetadata is ImagesSourceMetadata
    assert upsert_osm_place is osm_facade
