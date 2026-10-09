from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app import answer_generation, rag
from app.core.config import settings
from app.schemas import DocumentSearchResult, RagQuestionRequest, RagSource


class FakeModels:
    def __init__(self, response_text: str | None) -> None:
        self.response_text = response_text
        self.request: dict[str, object] | None = None

    def create(self, **request: object) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace(output_text=self.response_text)


class FakeClient:
    def __init__(self, api_key: str, response_text: str | None) -> None:
        self.api_key = api_key
        self.interactions = FakeModels(response_text)
        self.closed = False

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


def make_chunk() -> DocumentSearchResult:
    return DocumentSearchResult(
        document_id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
        source_name="menu.txt",
        chunk_index=0,
        content="Fresh pasta is made daily.",
        similarity=0.91,
    )


def test_generate_grounded_answer_sends_question_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient("test-api-key", "  Pasta is made daily [menu.txt#0].  ")
    monkeypatch.setattr(answer_generation.genai, "Client", lambda **kwargs: client)

    answer = answer_generation.generate_grounded_answer(
        "How often is the pasta made?",
        [make_chunk()],
    )

    request = client.interactions.request
    assert answer == "Pasta is made daily [menu.txt#0]."
    assert request["model"] == answer_generation.GEMINI_GENERATION_MODEL
    assert request["input"] == (
        "Question:\nHow often is the pasta made?\n\nRetrieved excerpts:\n"
        "[menu.txt#0]\nFresh pasta is made daily."
    )
    assert request["system_instruction"] == answer_generation.SYSTEM_INSTRUCTION
    assert request["generation_config"]["temperature"] == 0.2
    assert request["generation_config"]["max_output_tokens"] == 512
    assert request["store"] is False
    assert client.closed


def test_generate_grounded_answer_includes_multiple_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient(
        "test-api-key",
        "Pasta is made daily [menu.txt#0], and uses tomatoes [prep.txt#2].",
    )
    monkeypatch.setattr(answer_generation.genai, "Client", lambda **kwargs: client)
    chunks = [
        make_chunk(),
        DocumentSearchResult(
            document_id="d4ada293-47d1-492e-b6f3-ff27aa6624d2",
            source_name="prep.txt",
            chunk_index=2,
            content="The sauce uses tomatoes.",
            similarity=0.87,
        ),
    ]

    answer = answer_generation.generate_grounded_answer(
        "How is the pasta served?",
        chunks,
    )

    assert answer == "Pasta is made daily [menu.txt#0], and uses tomatoes [prep.txt#2]."
    assert client.interactions.request["input"] == (
        "Question:\nHow is the pasta served?\n\nRetrieved excerpts:\n"
        "[menu.txt#0]\nFresh pasta is made daily.\n\n"
        "[prep.txt#2]\nThe sauce uses tomatoes."
    )


def test_generate_grounded_answer_rejects_empty_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient("test-api-key", " ")
    monkeypatch.setattr(answer_generation.genai, "Client", lambda **kwargs: client)

    with pytest.raises(RuntimeError, match="empty answer"):
        answer_generation.generate_grounded_answer("question", [make_chunk()])


def test_generate_grounded_answer_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        answer_generation.generate_grounded_answer("question", [make_chunk()])


def test_rag_returns_no_documents_message_without_calling_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag, "search_document_chunks", lambda *args, **kwargs: [])

    def fail_if_called(*args: object, **kwargs: object) -> str:
        raise AssertionError("The generator must not run without retrieved chunks")

    monkeypatch.setattr(rag, "generate_grounded_answer", fail_if_called)

    response = rag.answer_with_rag(
        RagQuestionRequest(query="question"),
        session=object(),
    )

    assert response.answer == rag.NO_DOCUMENTS_ANSWER
    assert response.sources == []


def test_rag_preserves_sources_for_all_retrieved_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        make_chunk(),
        DocumentSearchResult(
            document_id="d4ada293-47d1-492e-b6f3-ff27aa6624d2",
            source_name="prep.txt",
            chunk_index=2,
            content="The sauce uses tomatoes.",
            similarity=0.87,
        ),
    ]
    generator_calls: list[tuple[str, list[DocumentSearchResult]]] = []
    monkeypatch.setattr(
        rag,
        "search_document_chunks",
        lambda query, session, *, top_k: chunks,
    )
    monkeypatch.setattr(
        rag,
        "generate_grounded_answer",
        lambda query, retrieved: generator_calls.append((query, retrieved))
        or "Pasta uses tomato sauce [prep.txt#2].",
    )

    response = rag.answer_with_rag(
        RagQuestionRequest(query="How is the pasta served?", top_k=2),
        session=object(),
    )

    assert generator_calls == [("How is the pasta served?", chunks)]
    assert response.answer == "Pasta uses tomato sauce [prep.txt#2]."
    assert response.sources == [
        RagSource(
            document_id=chunks[0].document_id,
            source_name="menu.txt",
            chunk_index=0,
            similarity=0.91,
        ),
        RagSource(
            document_id=chunks[1].document_id,
            source_name="prep.txt",
            chunk_index=2,
            similarity=0.87,
        ),
    ]
