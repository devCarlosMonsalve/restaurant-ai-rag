from types import SimpleNamespace
from typing import cast
from unittest.mock import sentinel
from uuid import uuid4

import pytest
from langchain_core.tracers.context import _tracing_v2_is_enabled
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app import answer_generation
from app import document_search, observability, rag
from app.agents import restaurant_search_agent
from app.agents.schemas import RestaurantSearchAgentResponse
from app.core.config import settings
from app.schemas import DocumentSearchResult, RagQuestionRequest
from app.workflows import restaurant_photo_search
from app.workflows.schemas import RestaurantPhotoWorkflowRequest


def test_phoenix_tracing_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "phoenix_tracing_enabled", False)

    assert observability.configure_phoenix_tracing() is None


def test_shutdown_flushes_and_closes_phoenix_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProvider:
        def force_flush(self, *, timeout_millis: int) -> bool:
            events.append(f"flush:{timeout_millis}")
            return True

        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(observability, "_provider", FakeProvider())

    assert observability.shutdown_phoenix_tracing() is True
    assert events == ["flush:5000", "shutdown"]
    assert observability._provider is None


def test_traced_span_reraises_errors_without_recording_error_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    class FakeSpan:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            del exc_info
            return False

        def set_attribute(self, key, value):
            recorded[key] = value

        def set_status(self, status):
            recorded["status"] = status

    class FakeTracer:
        def start_as_current_span(self, name, **kwargs):
            recorded["name"] = name
            recorded["initial_attributes"] = kwargs["attributes"]
            return FakeSpan()

    monkeypatch.setattr(observability, "_tracer", FakeTracer())

    with pytest.raises(RuntimeError, match="sensitive diagnostic"):
        with observability.traced_span(
            "test.operation",
            {"test.count": 1},
        ):
            raise RuntimeError("sensitive diagnostic")

    assert recorded["name"] == "test.operation"
    assert recorded["initial_attributes"] == {"test.count": 1}
    assert recorded["error.type"] == "RuntimeError"
    assert "sensitive diagnostic" not in str(recorded)


def test_rag_generation_disables_automatic_langsmith_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChain:
        def invoke(self, payload):
            assert payload["question"] == "synthetic question"
            assert _tracing_v2_is_enabled() is False
            return "grounded answer"

    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(answer_generation, "_RAG_GENERATION_CHAIN", FakeChain())

    answer = answer_generation.generate_grounded_answer(
        "synthetic question",
        [
            DocumentSearchResult(
                document_id=uuid4(),
                source_name="synthetic.txt",
                chunk_index=0,
                content="synthetic evidence",
                similarity=0.9,
            )
        ],
    )

    assert _tracing_v2_is_enabled() is True
    assert answer == "grounded answer"


def test_synthetic_rag_trace_is_hierarchical_and_excludes_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "PRIVATE_QUERY_CANARY"
    evidence = "PRIVATE_FRAGMENT_CANARY"
    source_name = "PRIVATE_SOURCE_CANARY"
    api_key_value = "PRIVATE_API_KEY_CANARY"
    document_id = uuid4()

    class FakeRows:
        def all(self):
            return [(chunk, 0.1)]

    class FakeSession:
        def execute(self, _statement):
            del _statement
            return FakeRows()

    class FakeInteractions:
        def create(self, **request):
            assert question in request["input"]
            assert evidence in request["input"]
            assert request["store"] is False
            assert _tracing_v2_is_enabled() is False
            return SimpleNamespace(output_text="synthetic answer")

    class FakeClient:
        def __init__(self) -> None:
            self.interactions = FakeInteractions()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            del exc_info
            return False

    def fake_client(*, api_key):
        assert api_key == api_key_value
        return FakeClient()

    chunk = SimpleNamespace(
        document_id=document_id,
        source_name=source_name,
        chunk_index=7,
        content=evidence,
    )
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_provider", provider)
    def fake_embed_query(_query):
        del _query
        return [0.1]

    monkeypatch.setattr(document_search, "embed_search_query", fake_embed_query)
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr(api_key_value))
    monkeypatch.setattr(
        answer_generation.genai,
        "Client",
        fake_client,
    )
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    try:
        response = rag.answer_with_rag(
            RagQuestionRequest(query=question, top_k=3),
            cast(Session, FakeSession()),
        )
        spans = exporter.get_finished_spans()
    finally:
        provider.shutdown()

    spans_by_name = {span.name: span for span in spans}
    assert response.answer == "synthetic answer"
    assert set(spans_by_name) == {
        "rag.answer_from_documents",
        "retrieval.document_chunks",
        "rag.generate_answer",
        "llm.gemini.document_answer",
    }

    root = spans_by_name["rag.answer_from_documents"]
    retrieval = spans_by_name["retrieval.document_chunks"]
    generation = spans_by_name["rag.generate_answer"]
    gemini = spans_by_name["llm.gemini.document_answer"]
    assert root.parent is None
    assert retrieval.parent is not None
    assert retrieval.parent.span_id == root.context.span_id
    assert generation.parent is not None
    assert generation.parent.span_id == root.context.span_id
    assert gemini.parent is not None
    assert gemini.parent.span_id == generation.context.span_id
    assert len({span.context.trace_id for span in spans}) == 1
    assert retrieval.attributes == {
        "retrieval.top_k": 3,
        "retrieval.result_count": 1,
    }
    assert generation.attributes == {
        "gen_ai.request.model": answer_generation.GEMINI_GENERATION_MODEL,
        "rag.context_chunk_count": 1,
    }

    serialized_spans = repr(
        [
            {
                "name": span.name,
                "attributes": dict(span.attributes or {}),
                "events": [
                    {
                        "name": event.name,
                        "attributes": dict(event.attributes or {}),
                    }
                    for event in span.events
                ],
            }
            for span in spans
        ]
    )
    for private_value in (
        question,
        evidence,
        source_name,
        api_key_value,
        str(document_id),
    ):
        assert private_value not in serialized_spans


def test_agent_graph_disables_automatic_langsmith_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = RestaurantSearchAgentResponse(answer="synthetic answer")

    def fake_run_conversation(router, query, session):
        assert query == "synthetic query"
        del router, session
        assert _tracing_v2_is_enabled() is False
        return expected

    def fake_router(api_key, *, openai_api_key=None):
        assert api_key == "synthetic-key"
        assert openai_api_key == "synthetic-openai-key"
        return object()

    monkeypatch.setattr(
        restaurant_search_agent.settings,
        "gemini_api_key",
        SecretStr("synthetic-key"),
    )
    monkeypatch.setattr(
        restaurant_search_agent.settings,
        "openai_api_key",
        SecretStr("synthetic-openai-key"),
    )
    monkeypatch.setattr(
        restaurant_search_agent,
        "create_restaurant_search_router",
        fake_router,
    )
    monkeypatch.setattr(
        restaurant_search_agent,
        "_run_conversation",
        fake_run_conversation,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "synthetic query",
        cast(Session, sentinel.session),
    )

    assert response is expected


def test_photo_workflow_graph_disables_automatic_langsmith_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGraph:
        def invoke(self, state, *, context):
            assert state["query"] == "synthetic query"
            assert context["session"] is sentinel.session
            assert _tracing_v2_is_enabled() is False
            return {
                "candidates": [],
                "evidence_status": "not_required",
                "photos": [],
                "photo_search_status": "not_started",
                "restaurants_with_photos": [],
                "candidates_without_returned_photos": [],
                "warnings": [],
            }

    monkeypatch.setattr(
        restaurant_photo_search,
        "_restaurant_photo_workflow",
        FakeGraph(),
    )

    response = restaurant_photo_search.run_restaurant_photo_workflow(
        RestaurantPhotoWorkflowRequest(query="synthetic query"),
        cast(Session, sentinel.session),
    )

    assert response.photo_search_status == "not_started"


def test_phoenix_configuration_uses_otlp_and_project_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observability, "_provider", None)
    monkeypatch.setattr(settings, "phoenix_tracing_enabled", True)
    monkeypatch.setattr(
        settings,
        "phoenix_collector_endpoint",
        "http://127.0.0.1:6006/v1/traces",
    )
    monkeypatch.setattr(settings, "phoenix_project_name", "test-project")
    captured: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, resource) -> None:
            self.resource = resource

        def add_span_processor(self, processor):
            captured["processor"] = processor

    provider = FakeProvider(sentinel.resource)

    def fake_resource_create(attributes):
        captured["resource"] = attributes
        return sentinel.resource

    def fake_exporter(*, endpoint):
        captured["endpoint"] = endpoint
        return sentinel.exporter

    def fake_tracer_provider(*, resource):
        provider.resource = resource
        return provider

    monkeypatch.setattr(observability.Resource, "create", fake_resource_create)
    monkeypatch.setattr(
        observability,
        "TracerProvider",
        fake_tracer_provider,
    )
    monkeypatch.setattr(observability, "OTLPSpanExporter", fake_exporter)
    monkeypatch.setattr(
        observability,
        "BatchSpanProcessor",
        lambda exporter: ("processor", exporter),
    )

    configured = observability.configure_phoenix_tracing()

    assert configured is provider
    assert captured["endpoint"] == "http://127.0.0.1:6006/v1/traces"
    assert captured["resource"] == {
        "service.name": "restaurant-ai-rag",
        "openinference.project.name": "test-project",
    }
    assert captured["processor"] == ("processor", sentinel.exporter)
