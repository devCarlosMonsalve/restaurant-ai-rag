from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

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
from app.image_ingestion import ImageSourceMetadata, ingest_image_to_database
from app.image_search import (
    _metadata_match_count,
    _tokens,
    search_images_by_text,
)
from app.models.image_embedding import ImageEmbedding
from app.open_data_sources import OSMRestaurant
from app.osm_ingestion import upsert_osm_place
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


def test_ingestion_persists_open_license_and_osm_provenance(
    tmp_path: Path,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "commons-photo.jpg"
    image_path.write_bytes(b"mock image")
    monkeypatch.setattr(
        "app.image_ingestion.embed_image",
        lambda path: [0.1] * IMAGE_EMBEDDING_DIMENSIONS,
    )
    place = upsert_osm_place(
        OSMRestaurant(
            osm_type="node",
            osm_id=123,
            name="Example Restaurant",
            city="Madrid",
            cuisine="italian",
            location="Calle Mayor 10",
            latitude=40.4,
            longitude=-3.7,
            wikimedia_commons="Category:Example Restaurant",
        ),
        db_session,
    )

    image_id = ingest_image_to_database(
        image_path,
        db_session,
        metadata=ImageSourceMetadata(
            source_url="https://commons.wikimedia.org/wiki/File:commons-photo.jpg",
            license_name="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            attribution="Photo by Example",
            osm_place_id=place.id,
        ),
    )
    stored_image = db_session.get(ImageEmbedding, image_id)

    assert stored_image is not None
    assert stored_image.source_url == (
        "https://commons.wikimedia.org/wiki/File:commons-photo.jpg"
    )
    assert stored_image.license_name == "CC BY 4.0"
    assert stored_image.attribution == "Photo by Example"
    assert stored_image.osm_place.name == "Example Restaurant"


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
            source_url="https://commons.wikimedia.org/wiki/File:Pasta.jpg",
            license_name="CC BY 4.0",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            attribution="Photo by Example",
            restaurant_name="Example Restaurant",
            restaurant_location="Madrid",
            restaurant_cuisine="Italian",
            restaurant_features=["Mesas al aire libre: disponible"],
            restaurant_source_url="https://www.openstreetmap.org/node/123",
            restaurant_attribution="© OpenStreetMap contributors",
            restaurant_attribution_url="https://www.openstreetmap.org/copyright",
            metadata_match_count=2,
        )
    ]
    calls = {}

    def fake_search(
        query: str,
        session: Session,
        *,
        top_k: int,
        osm_places_only: bool = False,
        city: str | None = None,
        cuisine: str | None = None,
    ):
        calls["query"] = query
        calls["top_k"] = top_k
        calls["osm_places_only"] = osm_places_only
        calls["city"] = city
        calls["cuisine"] = cuisine
        return expected_results

    monkeypatch.setattr("app.main.search_images_by_text", fake_search)
    response = client.post(
        "/images/search",
        json={
            "query": "pasta fresca italiana",
            "top_k": 3,
            "osm_places_only": True,
            "city": "Madrid",
            "cuisine": "tapas",
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["source_name"] == "pasta.png"
    assert response.json()[0]["similarity"] == 0.93
    assert response.json()[0]["license_name"] == "CC BY 4.0"
    assert response.json()[0]["attribution"] == "Photo by Example"
    assert response.json()[0]["restaurant_name"] == "Example Restaurant"
    assert response.json()[0]["restaurant_features"] == [
        "Mesas al aire libre: disponible"
    ]
    assert response.json()[0]["restaurant_attribution"] == (
        "© OpenStreetMap contributors"
    )
    assert response.json()[0]["restaurant_attribution_url"] == (
        "https://www.openstreetmap.org/copyright"
    )
    assert response.json()[0]["metadata_match_count"] == 2
    assert calls == {
        "query": "pasta fresca italiana",
        "top_k": 3,
        "osm_places_only": True,
        "city": "Madrid",
        "cuisine": "tapas",
    }


def test_osm_image_search_requires_all_requested_feature_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        _image_search_candidate(
            "Verified Restaurant",
            [
                "Mesas al aire libre: disponible",
                "Acceso en silla de ruedas: accesible",
            ],
        ),
        _image_search_candidate(
            "Limited Access Restaurant",
            [
                "Mesas al aire libre: disponible",
                "Acceso en silla de ruedas: accesibilidad limitada",
            ],
        ),
        _image_search_candidate(
            "Missing Terrace Restaurant",
            ["Acceso en silla de ruedas: accesible"],
        ),
    ]

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, statement):
            self.calls += 1
            return [] if self.calls == 1 else candidates

    monkeypatch.setattr(
        "app.image_search.embed_text_for_image_search",
        lambda _: [0.0] * 512,
    )

    results = search_images_by_text(
        "terraza y entrada accesible sin escalones",
        FakeSession(),
        osm_places_only=True,
    )

    assert [result.restaurant_name for result in results] == ["Verified Restaurant"]


def test_osm_image_search_rejects_stale_kosher_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_today = datetime.now(timezone.utc).date().isoformat()
    candidates = [
        _image_search_candidate(
            "Current Kosher Restaurant",
            [
                "Comida kosher: disponible",
                f"Última revisión kosher: {checked_today}",
            ],
        ),
        _image_search_candidate(
            "Stale Kosher Restaurant",
            [
                "Comida kosher: disponible",
                "Última revisión kosher: 2020-01-01",
            ],
        ),
    ]

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, statement):
            self.calls += 1
            return [] if self.calls == 1 else candidates

    monkeypatch.setattr(
        "app.image_search.embed_text_for_image_search",
        lambda _: [0.0] * 512,
    )

    results = search_images_by_text(
        "cocina kosher",
        FakeSession(),
        osm_places_only=True,
    )

    assert [result.restaurant_name for result in results] == [
        "Current Kosher Restaurant"
    ]


def test_metadata_search_matches_accents_and_important_spanish_terms() -> None:
    query_tokens = _tokens("Cocido madrileño en La Bola")

    assert _metadata_match_count(
        query_tokens,
        "osm-node-2697521931-Cocido_-_panoramio.jpg",
        "La Bola",
        "Category:La Bola Taberna",
    ) == 2
    assert _tokens("el de la en Madrid") == set()
    assert _metadata_match_count(
        _tokens("restaurante mexicano"),
        "cuisine:mexican",
    ) == 1


def test_image_search_endpoint_validates_query_and_top_k(
    client: TestClient,
) -> None:
    empty_query_response = client.post("/images/search", json={"query": "  "})
    excessive_top_k_response = client.post(
        "/images/search",
        json={"query": "pasta", "top_k": 21},
    )
    excessive_city_response = client.post(
        "/images/search",
        json={"query": "pasta", "city": "M" * 101},
    )
    excessive_cuisine_response = client.post(
        "/images/search",
        json={"query": "pasta", "cuisine": "c" * 101},
    )

    assert empty_query_response.status_code == 422
    assert excessive_top_k_response.status_code == 422
    assert excessive_city_response.status_code == 422
    assert excessive_cuisine_response.status_code == 422


def _image_search_candidate(
    restaurant_name: str,
    features: list[str],
) -> tuple[SimpleNamespace, SimpleNamespace, float]:
    place = SimpleNamespace(
        id=uuid4(),
        name=restaurant_name,
        location=None,
        cuisine=None,
        features=features,
        source_url="https://www.openstreetmap.org/node/1",
    )
    image = SimpleNamespace(
        id=uuid4(),
        source_name=f"{restaurant_name}.jpg",
        image_path=f"C:/images/{restaurant_name}.jpg",
        source_url=None,
        license_name=None,
        license_url=None,
        attribution=None,
    )
    return image, place, 0.1
