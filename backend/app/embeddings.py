from collections.abc import Sequence

from google import genai
from google.genai import types

from app.core.config import settings

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 768


def embed_document_chunks(
    chunks: Sequence[str],
    *,
    document_title: str | None = None,
) -> list[list[float]]:
    if not chunks:
        return []
    if any(not chunk.strip() for chunk in chunks):
        raise ValueError("Document chunks cannot be empty")

    title = document_title.strip() if document_title else "none"
    if not title:
        title = "none"
    contents = [f"title: {title} | text: {chunk}" for chunk in chunks]
    return _embed_contents(contents)


def embed_search_query(query: str) -> list[float]:
    if not query.strip():
        raise ValueError("Search query cannot be empty")

    contents = [f"task: search result | query: {query}"]
    return _embed_contents(contents)[0]


def _embed_contents(contents: Sequence[str]) -> list[list[float]]:
    api_key = settings.gemini_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise RuntimeError(
            "GEMINI_API_KEY must be set in the backend environment to generate embeddings"
        )

    with genai.Client(api_key=api_key.get_secret_value()) as client:
        response = client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=[
                types.UserContent(parts=[types.Part(text=content)])
                for content in contents
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )

    embeddings = response.embeddings
    if embeddings is None:
        raise RuntimeError("Gemini returned no embeddings")
    if len(embeddings) != len(contents):
        raise RuntimeError(
            f"Gemini returned {len(embeddings)} embeddings for {len(contents)} inputs"
        )

    vectors: list[list[float]] = []
    for embedding in embeddings:
        if embedding.values is None or len(embedding.values) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Gemini embeddings must contain {EMBEDDING_DIMENSIONS} values"
            )
        vectors.append([float(value) for value in embedding.values])

    return vectors
