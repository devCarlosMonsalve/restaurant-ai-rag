import asyncio
import json

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import AgentCard, Role, SendMessageRequest, Task, TaskState
from google.protobuf.json_format import MessageToDict
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.interfaces.a2a import restaurant_discovery_server as a2a_server
from app.agents.restaurant_search_agent import RestaurantSearchAgentError
from app.agents.schemas import (
    RestaurantSearchAgentCandidate,
    RestaurantSearchAgentPhoto,
    RestaurantSearchAgentResponse,
)
from app.infrastructure import observability


def send_a2a_message(query: str) -> tuple[AgentCard, Task]:
    async def invoke() -> tuple[AgentCard, Task]:
        async with a2a_server.app.router.lifespan_context(a2a_server.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=a2a_server.app),
                base_url="http://testserver",
            ) as http_client:
                card = await A2ACardResolver(
                    http_client,
                    "http://testserver",
                ).get_agent_card()
                card.supported_interfaces[0].url = "http://testserver"
                client = await create_client(
                    agent=card,
                    client_config=ClientConfig(
                        streaming=False,
                        httpx_client=http_client,
                    ),
                )
                try:
                    request = SendMessageRequest(
                        message=new_text_message(
                            query,
                            role=Role.ROLE_USER,
                        )
                    )
                    responses = [
                        response async for response in client.send_message(request)
                    ]
                finally:
                    await client.close()
                assert len(responses) == 1
                return card, responses[0].task

    return asyncio.run(invoke())


def task_messages(task: Task) -> list[str]:
    messages = list(task.history)
    if task.status.HasField("message"):
        messages.append(task.status.message)
    return [
        part.text
        for message in messages
        for part in message.parts
        if part.HasField("text")
    ]


def test_agent_card_and_jsonrpc_return_structured_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_queries: list[str] = []

    def fake_search(query: str) -> RestaurantSearchAgentResponse:
        received_queries.append(query)
        return RestaurantSearchAgentResponse(
            answer="Synthetic restaurant discovery result.",
            restaurants=[
                RestaurantSearchAgentCandidate(
                    name="Casa Verde",
                    city="Madrid",
                    cuisine="vegetarian",
                    location="Calle Mayor 1",
                    latitude=40.4168,
                    longitude=-3.7038,
                    features=[],
                    source_url="https://www.openstreetmap.org/node/123",
                    attribution="© OpenStreetMap contributors",
                    attribution_url="https://www.openstreetmap.org/copyright",
                    similarity=0.82,
                )
            ],
            photos=[
                RestaurantSearchAgentPhoto(
                    image_url="/images/commons-photo.jpg",
                    source_url="https://commons.wikimedia.org/wiki/File:commons-photo.jpg",
                    attribution="Photographer",
                )
            ],
        )

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", fake_search)

    card, task = send_a2a_message("  vegetarian restaurants in Madrid  ")

    assert card.name == "Restaurant Discovery Agent"
    assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
    assert list(card.default_input_modes) == ["text/plain"]
    assert list(card.default_output_modes) == ["application/json"]
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert received_queries == ["vegetarian restaurants in Madrid"]
    assert len(task.artifacts) == 1
    artifact = task.artifacts[0]
    assert artifact.name == "restaurant_discovery_result"
    assert len(artifact.parts) == 1
    assert artifact.parts[0].media_type == "application/json"
    result = MessageToDict(artifact.parts[0].data)
    assert result["answer"] == "Synthetic restaurant discovery result."
    assert result["restaurants"][0]["name"] == "Casa Verde"
    assert "id" not in result["restaurants"][0]
    assert result["photos"][0]["image_url"] == "/images/commons-photo.jpg"
    serialized = json.dumps(result)
    assert "image_path" not in serialized
    assert "C:\\private" not in serialized
    assert "id" not in result


@pytest.mark.parametrize("query", ["  ", "x" * 2001])
def test_invalid_query_fails_without_calling_agent(
    query: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        pytest.fail("Invalid input must not call the restaurant search Agent")

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", unexpected_search)

    card, task = send_a2a_message(query)

    assert card.name == "Restaurant Discovery Agent"
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert any(
        "non-empty text request" in message
        for message in task_messages(task)
    )


@pytest.mark.parametrize(
    "failure",
    [
        RestaurantSearchAgentError("provider details must not be returned"),
        RuntimeError("private backend details must not be returned"),
    ],
)
def test_agent_failures_are_sanitized(
    failure: Exception,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        raise failure

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", failed_search)

    card, task = send_a2a_message("synthetic query")

    assert card.name == "Restaurant Discovery Agent"
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert any(
        "could not complete the request" in message
        for message in task_messages(task)
    )
    serialized = json.dumps(task_messages(task))
    assert "provider details" not in serialized
    assert "private backend details" not in serialized


def test_request_and_result_content_are_excluded_from_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "A2A_PRIVATE_QUERY_MARKER"
    answer = "A2A_PRIVATE_RESULT_MARKER"
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability, "_provider", provider)
    monkeypatch.setattr(a2a_server, "configure_phoenix_tracing", lambda: provider)
    monkeypatch.setattr(a2a_server, "shutdown_phoenix_tracing", lambda: True)

    def fake_search(_query: str) -> RestaurantSearchAgentResponse:
        del _query
        return RestaurantSearchAgentResponse(answer=answer)

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", fake_search)

    try:
        card, task = send_a2a_message(query)
        assert card.name == "Restaurant Discovery Agent"
        assert task.status.state == TaskState.TASK_STATE_COMPLETED
        span_attributes = [
            dict(span.attributes)
            for span in exporter.get_finished_spans()
        ]
    finally:
        provider.shutdown()

    serialized = json.dumps(span_attributes)
    assert query not in serialized
    assert answer not in serialized
    assert span_attributes == [
        {
            "a2a.task.status": "completed",
            "agent.photo_count": 0,
        }
    ]
