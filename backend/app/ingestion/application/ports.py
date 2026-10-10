from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID


@dataclass(frozen=True)
class DocumentChunkRecord:
    document_id: UUID
    source_name: str
    chunk_index: int
    content: str
    embedding: Sequence[float]


class TextChunkReader(Protocol):
    def __call__(
        self,
        path: Path,
        *,
        chunk_size_words: int,
        overlap_words: int,
    ) -> Sequence[str]: ...


class DocumentChunkEmbedder(Protocol):
    def __call__(
        self,
        chunks: Sequence[str],
        *,
        document_title: str,
    ) -> Sequence[Sequence[float]]: ...


class DocumentChunkStore(Protocol):
    def save_all(self, records: Sequence[DocumentChunkRecord]) -> None: ...


@dataclass(frozen=True)
class ImageSourceMetadata:
    source_url: str
    license_name: str
    license_url: str
    attribution: str
    osm_place_id: UUID


@dataclass(frozen=True)
class ImageEmbeddingRecord:
    image_path: Path
    embedding: Sequence[float]
    metadata: ImageSourceMetadata | None


class ImageEmbedder(Protocol):
    def __call__(self, path: Path) -> Sequence[float]: ...


class ImageEmbeddingStore(Protocol):
    def upsert(self, record: ImageEmbeddingRecord) -> UUID: ...
