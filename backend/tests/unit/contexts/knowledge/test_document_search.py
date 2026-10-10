from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from app.knowledge.infrastructure.postgres import retriever


class FakeRows:
    def __init__(self, rows: list[tuple[object, float]]) -> None:
        self.rows = rows

    def all(self) -> list[tuple[object, float]]:
        return self.rows


class FakeSession:
    def __init__(self, rows: list[tuple[object, float]]) -> None:
        self.rows = rows
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return FakeRows(self.rows)


def make_chunk(
    *,
    document_id=None,
    source_name: str = "menu.txt",
    chunk_index: int = 0,
    content: str = "Fresh pasta is made daily.",
) -> SimpleNamespace:
    return SimpleNamespace(
        document_id=document_id or uuid4(),
        source_name=source_name,
        chunk_index=chunk_index,
        content=content,
    )


def test_search_document_chunks_uses_cosine_order_top_k_and_maps_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_embeddings: list[str] = []
    query_vector = [0.25, 0.5, 0.75]
    first = make_chunk()
    second = make_chunk(
        source_name="prep.txt",
        chunk_index=2,
        content="The sauce uses tomatoes.",
    )
    session = FakeSession([(first, 0.13), (second, 0.42)])

    def fake_embed_search_query(query: str) -> list[float]:
        query_embeddings.append(query)
        return query_vector

    monkeypatch.setattr(
        retriever,
        "embed_search_query",
        fake_embed_search_query,
    )

    results = retriever.PostgresDocumentRetriever(session).search_documents(
        "How is the pasta made?", top_k=2
    )

    assert query_embeddings == ["How is the pasta made?"]
    assert session.statement is not None
    compiled = session.statement.compile(dialect=postgresql.dialect())
    assert "document_chunks.embedding <=> " in compiled.string
    assert "ORDER BY document_chunks.embedding <=> " in compiled.string
    assert query_vector in compiled.params.values()
    assert 2 in compiled.params.values()
    assert [result.document_id for result in results] == [
        first.document_id,
        second.document_id,
    ]
    assert [result.source_name for result in results] == ["menu.txt", "prep.txt"]
    assert [result.chunk_index for result in results] == [0, 2]
    assert [result.content for result in results] == [
        "Fresh pasta is made daily.",
        "The sauce uses tomatoes.",
    ]
    assert [result.similarity for result in results] == pytest.approx([0.87, 0.58])


def test_search_document_chunks_propagates_database_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSession:
        def execute(self, statement):
            del statement
            raise RuntimeError("database query failed")

    monkeypatch.setattr(
        retriever,
        "embed_search_query",
        lambda _query: [0.1, 0.2],
    )

    with pytest.raises(RuntimeError, match="database query failed"):
        retriever.PostgresDocumentRetriever(FailingSession()).search_documents(
            "question", top_k=3
        )
