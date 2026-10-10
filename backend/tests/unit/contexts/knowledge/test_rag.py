from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

from app.core.config import settings
from app.knowledge.application import answer_question
from app.knowledge.infrastructure.generation import answer_chain
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


def make_chunk(
    *,
    source_name: str = "menu.txt",
    chunk_index: int = 0,
    content: str = "Fresh pasta is made daily.",
) -> DocumentSearchResult:
    return DocumentSearchResult(
        document_id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
        source_name=source_name,
        chunk_index=chunk_index,
        content=content,
        similarity=0.91,
    )


def test_generate_grounded_answer_sends_question_and_prepared_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient("test-api-key", "  Pasta is made daily [menu.txt#0].  ")
    monkeypatch.setattr(answer_chain.genai, "Client", lambda **kwargs: client)
    context = "[menu.txt#0]\nFresh pasta is made daily."
    prompt_messages = answer_chain._RAG_PROMPT.format_prompt(
        question="How often is the pasta made?",
        excerpts=context,
    ).to_messages()

    answer = answer_chain.generate_grounded_answer(
        "How often is the pasta made?",
        context,
        context_chunk_count=1,
    )

    assert len(prompt_messages) == 2
    assert isinstance(prompt_messages[0], SystemMessage)
    assert prompt_messages[0].content == answer_chain.SYSTEM_INSTRUCTION
    assert isinstance(prompt_messages[1], HumanMessage)
    assert prompt_messages[1].content == (
        "Question:\nHow often is the pasta made?\n\nRetrieved excerpts:\n"
        "[menu.txt#0]\nFresh pasta is made daily."
    )
    request = client.interactions.request
    assert answer == "Pasta is made daily [menu.txt#0]."
    assert request["model"] == answer_chain.GEMINI_GENERATION_MODEL
    assert request["input"] == (
        "Question:\nHow often is the pasta made?\n\nRetrieved excerpts:\n"
        "[menu.txt#0]\nFresh pasta is made daily."
    )
    assert request["system_instruction"] == answer_chain.SYSTEM_INSTRUCTION
    assert request["generation_config"]["temperature"] == 0.2
    assert request["generation_config"]["max_output_tokens"] == 512
    assert request["store"] is False
    assert client.closed


def test_generate_grounded_answer_preserves_context_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    client = FakeClient(
        "test-api-key",
        "Pasta is made daily [menu.txt#0], and uses tomatoes [prep.txt#2].",
    )
    monkeypatch.setattr(answer_chain.genai, "Client", lambda **kwargs: client)
    context = (
        "[menu.txt#0]\nFresh pasta is made daily.\n\n"
        "[prep.txt#2]\nThe sauce uses tomatoes."
    )

    answer = answer_chain.generate_grounded_answer(
        "How is the pasta served?",
        context,
        context_chunk_count=2,
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
    monkeypatch.setattr(answer_chain.genai, "Client", lambda **kwargs: client)

    with pytest.raises(RuntimeError, match="empty answer"):
        answer_chain.generate_grounded_answer(
            "question",
            "[menu.txt#0]\nEvidence.",
            context_chunk_count=1,
        )


def test_generate_grounded_answer_propagates_provider_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModels(FakeModels):
        def create(self, **request: object) -> SimpleNamespace:
            del request
            raise RuntimeError("Gemini provider unavailable")

    class FailingClient(FakeClient):
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.interactions = FailingModels(None)

    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    monkeypatch.setattr(
        answer_chain.genai,
        "Client",
        lambda **kwargs: FailingClient(kwargs["api_key"]),
    )

    with pytest.raises(RuntimeError, match="Gemini provider unavailable"):
        answer_chain.generate_grounded_answer(
            "question",
            "[menu.txt#0]\nEvidence.",
            context_chunk_count=1,
        )


def test_generate_grounded_answer_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", None)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        answer_chain.generate_grounded_answer(
            "question",
            "[menu.txt#0]\nEvidence.",
            context_chunk_count=1,
        )


def test_rag_returns_no_documents_without_calling_generator() -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.calls = []

        def search_documents(self, query: str, *, top_k: int):
            self.calls.append((query, top_k))
            return []

    def fail_if_called(question, context, /, *, context_chunk_count):
        pytest.fail("The generator must not run without retrieved chunks")

    retriever = FakeRetriever()
    request = RagQuestionRequest(query="question", top_k=4)

    response = answer_question.answer_from_documents(
        request,
        retriever,
        fail_if_called,
    )

    assert retriever.calls == [("question", 4)]
    assert response.answer == answer_question.NO_DOCUMENTS_ANSWER
    assert response.sources == []


def test_rag_builds_context_and_preserves_all_sources_in_order() -> None:
    first = make_chunk()
    second = make_chunk(
        source_name="prep.txt",
        chunk_index=2,
        content="The sauce uses tomatoes.",
    )
    chunks = [first, second]

    class FakeRetriever:
        def __init__(self) -> None:
            self.calls = []

        def search_documents(self, query: str, *, top_k: int):
            self.calls.append((query, top_k))
            return chunks

    generator_calls = []

    def fake_generate(question, context, /, *, context_chunk_count):
        generator_calls.append((question, context, context_chunk_count))
        return "Pasta uses tomato sauce [prep.txt#2]."

    retriever = FakeRetriever()
    response = answer_question.answer_from_documents(
        RagQuestionRequest(query="How is the pasta served?", top_k=2),
        retriever,
        fake_generate,
    )

    assert retriever.calls == [("How is the pasta served?", 2)]
    assert generator_calls == [
        (
            "How is the pasta served?",
            "[menu.txt#0]\nFresh pasta is made daily.\n\n"
            "[prep.txt#2]\nThe sauce uses tomatoes.",
            2,
        )
    ]
    assert response.answer == "Pasta uses tomato sauce [prep.txt#2]."
    assert response.sources == [
        RagSource(
            document_id=first.document_id,
            source_name="menu.txt",
            chunk_index=0,
            similarity=0.91,
        ),
        RagSource(
            document_id=second.document_id,
            source_name="prep.txt",
            chunk_index=2,
            similarity=0.91,
        ),
    ]
