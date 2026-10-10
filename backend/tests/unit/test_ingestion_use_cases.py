from pathlib import Path
from uuid import UUID

import pytest

from app.domain.osm_place import osm_searchable_metadata_changed
from app.ingestion.application.documents_usecase import ingest_text_document
from app.ingestion.application.images_usecase import ingest_image
from app.ingestion.application.ports import (
    DocumentChunkRecord,
    ImageEmbeddingRecord,
    ImageSourceMetadata,
)


def test_document_use_case_coordinates_chunking_embedding_and_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "menu.txt"
    records: list[DocumentChunkRecord] = []

    class Store:
        def save_all(self, chunks: list[DocumentChunkRecord]) -> None:
            records.extend(chunks)

    def read_chunks(
        path: Path,
        *,
        chunk_size_words: int,
        overlap_words: int,
    ) -> list[str]:
        assert path == source
        assert chunk_size_words == 500
        assert overlap_words == 75
        return ["first chunk", "second chunk"]

    def embed_chunks(
        chunks: list[str],
        *,
        document_title: str,
    ) -> list[list[float]]:
        assert chunks == ["first chunk", "second chunk"]
        assert document_title == "menu"
        return [[0.1], [0.2]]

    document_id, chunk_count = ingest_text_document(
        source,
        reader=read_chunks,
        embedder=embed_chunks,
        store=Store(),
    )

    assert isinstance(document_id, UUID)
    assert chunk_count == 2
    assert [record.document_id for record in records] == [document_id, document_id]
    assert [record.chunk_index for record in records] == [0, 1]
    assert [record.content for record in records] == ["first chunk", "second chunk"]
    assert [record.embedding for record in records] == [[0.1], [0.2]]


def test_document_use_case_does_not_store_when_embedding_fails(tmp_path: Path) -> None:
    saved: list[DocumentChunkRecord] = []

    class Store:
        def save_all(self, records: list[DocumentChunkRecord]) -> None:
            saved.extend(records)

    def fail_embedding(
        chunks: list[str],
        *,
        document_title: str,
    ) -> list[list[float]]:
        assert chunks == ["chunk"]
        assert document_title == "menu"
        raise RuntimeError("embedding failed")

    def read_chunk(
        path: Path,
        *,
        chunk_size_words: int,
        overlap_words: int,
    ) -> list[str]:
        assert path.name == "menu.txt"
        assert chunk_size_words == 500
        assert overlap_words == 75
        return ["chunk"]

    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest_text_document(
            tmp_path / "menu.txt",
            reader=read_chunk,
            embedder=fail_embedding,
            store=Store(),
        )

    assert saved == []


def test_image_use_case_resolves_path_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"image")
    expected_id = UUID("11111111-1111-1111-1111-111111111111")
    metadata = ImageSourceMetadata(
        source_url="https://example.test/photo.jpg",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="Photographer",
        osm_place_id=UUID("22222222-2222-2222-2222-222222222222"),
    )
    stored: list[ImageEmbeddingRecord] = []

    class Store:
        def upsert(self, record: ImageEmbeddingRecord) -> UUID:
            stored.append(record)
            return expected_id

    def embed(path: Path) -> list[float]:
        assert path == image_path.resolve()
        return [0.3, 0.4]

    result = ingest_image(
        image_path,
        embedder=embed,
        store=Store(),
        metadata=metadata,
    )

    assert result == expected_id
    assert stored[0].image_path == image_path.resolve()
    assert stored[0].embedding == [0.3, 0.4]
    assert stored[0].metadata == metadata


def test_image_use_case_rejects_missing_file_before_embedding(
    tmp_path: Path,
) -> None:
    embedded = False

    class Store:
        def upsert(self, record: ImageEmbeddingRecord) -> UUID:
            raise AssertionError(f"The store must not be called: {record}")

    def embed(path: Path) -> list[float]:
        nonlocal embedded
        assert path.name == "missing.jpg"
        embedded = True
        return [0.1]

    with pytest.raises(FileNotFoundError):
        ingest_image(
            tmp_path / "missing.jpg",
            embedder=embed,
            store=Store(),
        )

    assert embedded is False


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (("Name", "City", "Italian", "Center", ["pasta"]), ("Other", "City", "Italian", "Center", ["pasta"])),
        (("Name", "City", "Italian", "Center", ["pasta"]), ("Name", "Other", "Italian", "Center", ["pasta"])),
        (("Name", "City", "Italian", "Center", ["pasta"]), ("Name", "City", "Spanish", "Center", ["pasta"])),
        (("Name", "City", "Italian", "Center", ["pasta"]), ("Name", "City", "Italian", "Other", ["pasta"])),
        (("Name", "City", "Italian", "Center", ["pasta"]), ("Name", "City", "Italian", "Center", ["tapas"])),
        (("Name", "City", "Italian", "Center", ["pasta", "pizza"]), ("Name", "City", "Italian", "Center", ["pizza", "pasta"])),
    ],
)
def test_osm_searchable_metadata_changes_invalidate_embedding(
    current: tuple[str, str, str, str, list[str]],
    new: tuple[str, str, str, str, list[str]],
) -> None:
    assert osm_searchable_metadata_changed(
        current_name=current[0],
        current_city=current[1],
        current_cuisine=current[2],
        current_location=current[3],
        current_features=current[4],
        new_name=new[0],
        new_city=new[1],
        new_cuisine=new[2],
        new_location=new[3],
        new_features=new[4],
    )


def test_osm_searchable_metadata_treats_missing_features_as_empty() -> None:
    assert not osm_searchable_metadata_changed(
        current_name="Name",
        current_city="City",
        current_cuisine=None,
        current_location=None,
        current_features=None,
        new_name="Name",
        new_city="City",
        new_cuisine=None,
        new_location=None,
        new_features=[],
    )
