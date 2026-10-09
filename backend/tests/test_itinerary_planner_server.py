import asyncio
import json
from contextlib import AsyncExitStack
from typing import Any

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_data_message
from a2a.types import AgentCard, Role, SendMessageRequest, Task, TaskState
from google.protobuf.json_format import MessageToDict

from app import a2a_server, itinerary_planner, itinerary_planner_server
from app.agents.schemas import (
    RestaurantSearchAgentCandidate,
    RestaurantSearchAgentResponse,
)


def send_planner_request(
    payload: dict[str, Any],
    *,
    restaurant_agent: bool = False,
) -> tuple[AgentCard, Task]:
    async def invoke() -> tuple[AgentCard, Task]:
        async with AsyncExitStack() as stack:
            if restaurant_agent:
                await stack.enter_async_context(
                    a2a_server.app.router.lifespan_context(a2a_server.app)
                )
            await stack.enter_async_context(
                itinerary_planner_server.app.router.lifespan_context(
                    itinerary_planner_server.app
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=itinerary_planner_server.app
                ),
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
                        accepted_output_modes=["application/json"],
                    ),
                )
                try:
                    request = SendMessageRequest(
                        message=new_data_message(
                            payload,
                            media_type="application/json",
                            role=Role.ROLE_USER,
                        )
                    )
                    responses = [
                        response async for response in client.send_message(request)
                    ]
                finally:
                    await client.close()

        assert len(responses) == 1
        assert responses[0].WhichOneof("payload") == "task"
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


def test_planner_agent_delegates_to_restaurant_agent_and_returns_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_queries: list[str] = []

    def fake_search(query: str) -> RestaurantSearchAgentResponse:
        received_queries.append(query)
        return RestaurantSearchAgentResponse(
            answer="Synthetic options from restaurant discovery.",
            restaurants=[
                RestaurantSearchAgentCandidate(
                    name="Casa Verde",
                    city="Madrid",
                    cuisine="vegetarian",
                    location="Calle Mayor 1",
                    latitude=40.4168,
                    longitude=-3.7038,
                    features=["outdoor_seating"],
                    source_url="https://www.openstreetmap.org/node/123",
                    attribution="© OpenStreetMap contributors",
                    attribution_url="https://www.openstreetmap.org/copyright",
                    similarity=0.82,
                ),
                RestaurantSearchAgentCandidate(
                    name="Huerta Viva",
                    city="Madrid",
                    cuisine="vegetarian",
                    location="Calle Atocha 2",
                    latitude=40.412,
                    longitude=-3.699,
                    features=[],
                    source_url="https://www.openstreetmap.org/node/456",
                    attribution="© OpenStreetMap contributors",
                    attribution_url="https://www.openstreetmap.org/copyright",
                    similarity=0.77,
                ),
            ],
        )

    async def plan_through_in_process_agent(
        query: str,
        day_count: int,
    ) -> itinerary_planner.ItineraryDiningDraft:
        return await itinerary_planner.plan_itinerary_dining(
            query,
            day_count,
            transport=httpx.ASGITransport(app=a2a_server.app),
        )

    monkeypatch.setattr(a2a_server, "_run_restaurant_search", fake_search)
    monkeypatch.setattr(
        itinerary_planner_server,
        "plan_itinerary_dining",
        plan_through_in_process_agent,
    )

    card, task = send_planner_request(
        {
            "query": "vegetarian restaurants in Madrid",
            "day_count": 2,
        },
        restaurant_agent=True,
    )

    assert card.name == "Itinerary Planner Agent"
    assert list(card.default_input_modes) == ["application/json"]
    assert list(card.default_output_modes) == ["application/json"]
    assert task.status.state == TaskState.TASK_STATE_COMPLETED
    assert received_queries == ["vegetarian restaurants in Madrid"]
    assert len(task.artifacts) == 1
    artifact = task.artifacts[0]
    assert artifact.name == "itinerary_dining_draft"
    assert len(artifact.parts) == 1
    assert artifact.parts[0].media_type == "application/json"
    draft = MessageToDict(artifact.parts[0].data)
    assert [day["restaurant"]["name"] for day in draft["days"]] == [
        "Casa Verde",
        "Huerta Viva",
    ]
    assert draft["research_answer"] == (
        "Synthetic options from restaurant discovery."
    )
    assert draft["unfilled_days"] == 0
    assert all("id" not in day["restaurant"] for day in draft["days"])


@pytest.mark.parametrize(
    "payload",
    [
        {"query": " ", "day_count": 1},
        {"query": "restaurants", "day_count": 0},
        {"query": "restaurants", "day_count": True},
        {"query": "restaurants", "day_count": 1.5},
        {"query": "restaurants", "day_count": 1, "unexpected": True},
    ],
)
def test_planner_agent_rejects_invalid_json_without_delegating(
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_plan(_query: str, _day_count: int) -> None:
        del _query, _day_count
        pytest.fail("Invalid JSON requests must not invoke itinerary planning")

    monkeypatch.setattr(
        itinerary_planner_server,
        "plan_itinerary_dining",
        unexpected_plan,
    )

    card, task = send_planner_request(payload)

    assert card.name == "Itinerary Planner Agent"
    assert task.status.state == TaskState.TASK_STATE_FAILED
    assert any("application/json request" in message for message in task_messages(task))


def test_planner_agent_sanitizes_downstream_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_plan(_query: str, _day_count: int) -> None:
        del _query, _day_count
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(
        itinerary_planner_server,
        "plan_itinerary_dining",
        failed_plan,
    )

    _, task = send_planner_request(
        {
            "query": "restaurants in Madrid",
            "day_count": 1,
        }
    )

    assert task.status.state == TaskState.TASK_STATE_FAILED
    messages = task_messages(task)
    assert any("could not complete the request" in message for message in messages)
    assert "private provider detail" not in json.dumps(messages)
