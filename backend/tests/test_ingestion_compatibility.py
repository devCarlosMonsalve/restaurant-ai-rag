from app.document_ingestion import (
    ingest_txt_to_database as legacy_ingest_txt_to_database,
)
from app.image_ingestion import (
    ImageSourceMetadata as LegacyImageSourceMetadata,
    ingest_image_to_database as legacy_ingest_image_to_database,
)
from app.ingestion import (
    ImageSourceMetadata,
    ingest_image_to_database,
    ingest_txt_to_database,
    upsert_osm_place,
)
from app.ingestion.application.documents import ingest_txt_to_database as documents_use_case
from app.ingestion.application.images import (
    ingest_image_to_database as images_use_case,
)
from app.ingestion.application.osm import upsert_osm_place as osm_use_case
from app.osm_ingestion import upsert_osm_place as legacy_upsert_osm_place


def test_flat_ingestion_imports_remain_compatible() -> None:
    assert legacy_ingest_txt_to_database is ingest_txt_to_database is documents_use_case
    assert legacy_ingest_image_to_database is ingest_image_to_database is images_use_case
    assert LegacyImageSourceMetadata is ImageSourceMetadata
    assert legacy_upsert_osm_place is upsert_osm_place is osm_use_case
