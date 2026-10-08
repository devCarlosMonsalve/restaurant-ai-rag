from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app import embeddings
from app.core.config import settings


class FakeModels:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def embed_content(self, **request: object) -> SimpleNamespace:
        self.request = request
        contents = request["contents"]
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[float(index)] * embeddings.EMBEDDING_DIMENSIONS)
                for index, _ in enumerate(contents)
            ]
        )


class FakeClient:
    def __init__(self, **kwargs: object) -> None:
        self.api_key = kwargs["api_key"]
        self.models = FakeModels()
        self.closed = False

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


def test_embed_document_chunks_formats_and_returns_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient(api_key="test-api-key")
    monkeypatch.setattr(embeddings.genai, "Client", lambda **kwargs: client)

    vectors = embeddings.embed_document_chunks(
        ["Chunk one", "Chunk two"],
        document_title="Menu",
    )

    assert len(vectors) == 2
    assert len(vectors[0]) == embeddings.EMBEDDING_DIMENSIONS
    assert client.api_key == "test-api-key"
    assert client.models.request["model"] == embeddings.GEMINI_EMBEDDING_MODEL
    content_texts = [
        content.parts[0].text for content in client.models.request["contents"]
    ]
    assert content_texts == [
        "title: Menu | text: Chunk one",
        "title: Menu | text: Chunk two",
    ]
    assert client.models.request["config"].output_dimensionality == 768
    assert client.closed


def test_embed_search_query_uses_retrieval_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient(api_key="test-api-key")
    monkeypatch.setattr(embeddings.genai, "Client", lambda **kwargs: client)

    vector = embeddings.embed_search_query("quiet Italian restaurant")

    assert len(vector) == embeddings.EMBEDDING_DIMENSIONS
    assert client.models.request["contents"][0].parts[0].text == (
        "task: search result | query: quiet Italian restaurant"
    )


def test_embedding_requires_a_gemini_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        embeddings.embed_document_chunks(["Chunk one"])
