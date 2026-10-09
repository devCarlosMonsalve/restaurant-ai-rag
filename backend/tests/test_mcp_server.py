import asyncio
from dataclasses import replace
from unittest.mock import sentinel
from uuid import uuid4

import pytest
from mcp import Client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app import mcp_server, observability
from app.schemas import (
    DocumentSearchResult,
    ImageSearchResult,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
    RagAnswerResponse,
    RagSource,
)
from app.tools.registry import TOOL_DEFINITIONS


class FakeSession:
    def __init__(self) -> None:
        self.close_count = 0

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close_count += 1


def call_tool(name: str, arguments: dict[str, object]):
    async def invoke():
        async with Client(mcp_server.mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(invoke())


def replace_handler(monkeypatch, name: str, handler) -> None:
    definition = TOOL_DEFINITIONS[name]
    monkeypatch.setitem(TOOL_DEFINITIONS, name, replace(definition, handler=handler))


def test_protocol_lists_tools_without_hidden_or_internal_arguments():
    async def list_tools():
        async with Client(mcp_server.mcp) as client:
            return await client.list_tools()

    result = asyncio.run(list_tools())
    tools = {tool.name: tool for tool in result.tools}

    assert set(tools) == set(TOOL_DEFINITIONS)
    assert "osm_places_only" not in tools["search_restaurant_photos"].input_schema[
        "properties"
    ]
    assert all(
        "candidate_id" not in tool.input_schema["properties"]
        for tool in result.tools
    )


def test_tools_delegate_and_return_sanitized_structured_results(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(mcp_server, "SessionLocal", lambda: session)
    restaurant_id = uuid4()
    document_id = uuid4()
    image_id = uuid4()
    restaurant_result = OsmRestaurantSearchResponse(
        results=[
            OsmRestaurantSearchResult(
                id=restaurant_id,
                name="Casa Test",
                city="Madrid",
                cuisine="Spanish",
                location="Centro",
                latitude=40.4,
                longitude=-3.7,
                features=["outdoor_seating"],
                source_url="https://www.openstreetmap.org/node/1",
                attribution="© OpenStreetMap contributors",
                attribution_url="https://www.openstreetmap.org/copyright",
                similarity=0.91,
            )
        ],
        evidence_status="verified",
    )
    photo_result = ImageSearchResult(
        id=image_id,
        source_name="commons-photo.jpg",
        image_path="C:\\private\\images\\commons-photo.jpg",
        similarity=0.88,
        source_url="https://commons.wikimedia.org/wiki/File:commons-photo.jpg",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="Photographer",
        restaurant_name="Casa Test",
        restaurant_source_url="https://www.openstreetmap.org/node/1",
    )
    document_result = DocumentSearchResult(
        document_id=document_id,
        source_name="guide.pdf",
        chunk_index=2,
        content="Relevant excerpt",
        similarity=0.84,
    )
    answer_result = RagAnswerResponse(
        answer="A grounded answer.",
        sources=[
            RagSource(
                document_id=document_id,
                source_name="guide.pdf",
                chunk_index=2,
                similarity=0.84,
            )
        ],
    )
    results = {
        "search_restaurants": restaurant_result,
        "search_restaurant_photos": [photo_result],
        "search_documents": [document_result],
        "answer_from_documents": answer_result,
    }
    calls: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        def handler(_result=result, _name=name, **kwargs):
            calls[_name] = kwargs
            return _result

        replace_handler(monkeypatch, name, handler)

    responses = {
        "search_restaurants": call_tool(
            "search_restaurants",
            {"query": "Spanish in Madrid", "city": "Madrid", "cuisine": "Spanish"},
        ),
        "search_restaurant_photos": call_tool(
            "search_restaurant_photos",
            {"query": "Casa Test", "city": "Madrid"},
        ),
        "search_documents": call_tool("search_documents", {"query": "opening hours"}),
        "answer_from_documents": call_tool(
            "answer_from_documents", {"query": "What are the opening hours?"}
        ),
    }

    assert responses["search_restaurants"].structured_content["evidence_status"] == "verified"
    assert responses["search_restaurant_photos"].structured_content["results"][0][
        "restaurant_name"
    ] == "Casa Test"
    assert responses["search_documents"].structured_content["results"][0][
        "source_name"
    ] == "guide.pdf"
    assert responses["answer_from_documents"].structured_content["answer"] == (
        "A grounded answer."
    )
    assert calls["search_restaurants"]["city"] == "Madrid"
    assert calls["search_restaurants"]["cuisine"] == "Spanish"
    assert "osm_places_only" not in calls["search_restaurant_photos"]
    assert all(kwargs["session"] is session for kwargs in calls.values())
    assert session.close_count == 4

    serialized = repr(
        [response.structured_content for response in responses.values()]
    )
    assert str(restaurant_id) not in serialized
    assert str(image_id) not in serialized
    assert str(document_id) not in serialized
    assert "image_path" not in serialized
    assert "C:\\private\\images" not in serialized


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "restaurants", "top_k": 21},
    ],
)
def test_invalid_arguments_are_rejected(arguments, monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(mcp_server, "SessionLocal", lambda: session)

    def unexpected_call(**_kwargs):
        pytest.fail("Invalid arguments must not call the application Tool")

    replace_handler(monkeypatch, "search_restaurant_photos", unexpected_call)
    response = call_tool("search_restaurant_photos", arguments)

    assert response.is_error
    assert session.close_count == 0


def test_hidden_photo_filter_is_not_forwarded(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(mcp_server, "SessionLocal", lambda: session)
    received: dict[str, object] = {}

    def handler(**kwargs):
        received.update(kwargs)
        return []

    replace_handler(monkeypatch, "search_restaurant_photos", handler)
    response = call_tool(
        "search_restaurant_photos",
        {"query": "photos", "osm_places_only": True},
    )

    assert not response.is_error
    assert "osm_places_only" not in received
    assert session.close_count == 1


def test_empty_results_are_success_and_tool_errors_are_sanitized(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(mcp_server, "SessionLocal", lambda: session)
    replace_handler(monkeypatch, "search_documents", lambda **_kwargs: [])

    empty_response = call_tool("search_documents", {"query": "no matching document"})

    assert not empty_response.is_error
    assert empty_response.structured_content == {"results": []}

    def failed_handler(**_kwargs):
        raise RuntimeError("postgres://private-user:secret@db/private")

    replace_handler(monkeypatch, "search_documents", failed_handler)
    error_response = call_tool("search_documents", {"query": "database failure"})

    assert error_response.is_error
    error_text = " ".join(item.text for item in error_response.content if item.type == "text")
    assert "tool_execution_failed" in error_text
    assert "private-user" not in error_text
    assert "secret" not in error_text
    assert session.close_count == 2


def test_mcp_tool_trace_excludes_arguments_and_results(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_provider", provider)
    session = FakeSession()
    monkeypatch.setattr(mcp_server, "SessionLocal", lambda: session)

    def empty_handler(**kwargs):
        del kwargs
        return []

    replace_handler(monkeypatch, "search_documents", empty_handler)
    private_query = "PRIVATE_MCP_QUERY_CANARY"

    try:
        response = call_tool("search_documents", {"query": private_query})
        spans = exporter.get_finished_spans()
    finally:
        provider.shutdown()

    assert not response.is_error
    assert response.structured_content == {"results": []}
    assert session.close_count == 1
    assert len(spans) == 1
    assert spans[0].name == "mcp.tool.execute"
    assert spans[0].attributes == {
        "tool.name": "search_documents",
        "tool.status": "success",
    }
    assert private_query not in repr(
        (spans[0].name, dict(spans[0].attributes or {}))
    )


def test_mcp_main_configures_and_shuts_down_tracing(monkeypatch):
    events = []
    monkeypatch.setattr(
        mcp_server,
        "configure_phoenix_tracing",
        lambda: sentinel.provider,
    )
    monkeypatch.setattr(
        mcp_server.mcp,
        "run",
        lambda **kwargs: events.append(("run", kwargs)),
    )
    monkeypatch.setattr(
        mcp_server,
        "shutdown_phoenix_tracing",
        lambda: events.append("shutdown") or True,
    )
    monkeypatch.setattr(
        mcp_server.engine,
        "dispose",
        lambda: events.append("dispose"),
    )

    mcp_server.main()

    assert events == [
        ("run", {"transport": "stdio"}),
        "shutdown",
        "dispose",
    ]
