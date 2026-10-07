from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.image_embeddings import (
    IMAGE_EMBEDDING_DIMENSIONS,
    embed_image,
    embed_text_for_image_search,
)
from app.image_ingestion import ingest_image_to_database
from app.models.image_embedding import ImageEmbedding
from app.schemas import ImageSearchResult


class FakeClipModel:
    def encode_image(self, image_input: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[3.0, 4.0] + [0.0] * (IMAGE_EMBEDDING_DIMENSIONS - 2)])

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[3.0, 4.0] + [0.0] * (IMAGE_EMBEDDING_DIMENSIONS - 2)])


def test_clip_embeddings_are_normalized_for_images_and_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "pasta.png"
    Image.new("RGB", (16, 16), color="red").save(image_path)
    model = FakeClipModel()
    monkeypatch.setattr(
        "app.image_embeddings._load_clip",
        lambda: (
            model,
            lambda image: torch.zeros((3, 224, 224)),
            lambda texts: torch.zeros((len(texts), 77), dtype=torch.long),
            torch.device("cpu"),
        ),
    )

    image_vector = embed_image(image_path)
    text_vector = embed_text_for_image_search("  pasta fresca  ")

    assert len(image_vector) == IMAGE_EMBEDDING_DIMENSIONS
    assert image_vector[:2] == pytest.approx([0.6, 0.8])
    assert text_vector[:2] == pytest.approx([0.6, 0.8])
    assert sum(value**2 for value in image_vector) == pytest.approx(1.0)
    assert sum(value**2 for value in text_vector) == pytest.approx(1.0)


def test_embed_image_rejects_unsupported_file_type(tmp_path: Path) -> None:
    image_path = tmp_path / "menu.txt"
    image_path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported image type"):
        embed_image(image_path)


def test_ingestion_stores_image_embedding_and_updates_existing_path(
    tmp_path: Path,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "pasta.png"
    image_path.write_bytes(b"mock image")
    monkeypatch.setattr(
        "app.image_ingestion.embed_image",
        lambda path: [0.1] * IMAGE_EMBEDDING_DIMENSIONS,
    )

    image_id = ingest_image_to_database(image_path, db_session)
    updated_id = ingest_image_to_database(image_path, db_session)
    stored_images = db_session.scalars(select(ImageEmbedding)).all()

    assert image_id == updated_id
    assert len(stored_images) == 1
    assert stored_images[0].source_name == "pasta.png"
    assert len(stored_images[0].embedding) == IMAGE_EMBEDDING_DIMENSIONS


def test_image_search_endpoint_returns_matches(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_results = [
        ImageSearchResult(
            id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
            source_name="pasta.png",
            image_path="C:/images/pasta.png",
            similarity=0.93,
        )
    ]
    calls = {}

    def fake_search(query: str, session: Session, *, top_k: int):
        calls["query"] = query
        calls["top_k"] = top_k
        return expected_results

    monkeypatch.setattr("app.main.search_images_by_text", fake_search)
    response = client.post(
        "/images/search",
        json={"query": "pasta fresca italiana", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()[0]["source_name"] == "pasta.png"
    assert response.json()[0]["similarity"] == 0.93
    assert calls == {"query": "pasta fresca italiana", "top_k": 3}


def test_image_search_endpoint_validates_query_and_top_k(
    client: TestClient,
) -> None:
    empty_query_response = client.post("/images/search", json={"query": "  "})
    excessive_top_k_response = client.post(
        "/images/search",
        json={"query": "pasta", "top_k": 21},
    )

    assert empty_query_response.status_code == 422
    assert excessive_top_k_response.status_code == 422
