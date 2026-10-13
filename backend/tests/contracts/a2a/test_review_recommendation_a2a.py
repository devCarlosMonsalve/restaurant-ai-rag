import asyncio
import json
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_data_message, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from google.genai import types
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct
from pydantic import SecretStr

from app.agents import restaurant_search_agent, review_recommendation_agent
from app.agents.review_recommendation_a2a_client import (
    ReviewRecommendationDelegationError,
    _read_recommendation_result,
    delegate_review_recommendation,
)
from app.agents.review_recommendation_agent import ReviewRecommendationAgentError
from app.agents.schemas import (
    CandidateComparison,
    CandidateEvidence,
    ReviewCandidate,
    ReviewRecommendationRequest,
    ReviewRecommendationResponse,
)
from app.core.config import settings
from app.interfaces.a2a import review_recommendation_server as review_server
from app.interfaces.a2a import restaurant_discovery_server as discovery_server
from app.restaurant_discovery.application.contracts import OsmRestaurantSearchResult


def make_request() -> ReviewRecommendationRequest:
    return ReviewRecommendationRequest(
        query="Compare vegetarian restaurants in Madrid",
        preferences=["vegetarian options"],
        candidates=[
            ReviewCandidate(
                name="Casa Verde",
                city="Madrid",
                cuisine="vegetarian",
                location="Calle Mayor 1",
                features=["Opciones vegetarianas: disponible"],
                source_url="https://www.openstreetmap.org/node/123",
                attribution="© OpenStreetMap contributors",
                attribution_url="https://www.openstreetmap.org/copyright",
            )
        ],
    )


def make_response() -> ReviewRecommendationResponse:
    request = make_request()
    candidate = request.candidates[0]
    return ReviewRecommendationResponse(
        recommended_candidate_source_url=candidate.source_url,
        comparisons=[
            CandidateComparison(
                candidate_source_url=candidate.source_url,
                assessment="matches",
                reasons=["The candidate has an indexed vegetarian feature."],
                evidence=[
                    CandidateEvidence(
                        field="features",
                        value="Opciones vegetarianas: disponible",
                        preference="vegetarian options",
                    )
                ],
                unresolved_preferences=[],
            )
        ],
        limitations=["No restaurant-linked review data is available."],
    )


def delegate_in_process(
    request: ReviewRecommendationRequest,
) -> ReviewRecommendationResponse:
    async def invoke() -> ReviewRecommendationResponse:
        async with review_server.app.router.lifespan_context(review_server.app):
            return await delegate_review_recommendation(
                request,
                agent_url=review_server.A2A_URL,
                transport=httpx.ASGITransport(app=review_server.app),
            )

    return asyncio.run(invoke())


def test_review_a2a_client_discovers_agent_and_validates_structured_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_requests: list[ReviewRecommendationRequest] = []
    expected_response = make_response()

    def fake_run(
        request: ReviewRecommendationRequest,
    ) -> ReviewRecommendationResponse:
        received_requests.append(request)
        return expected_response

    monkeypatch.setattr(review_server, "_run_recommendation", fake_run)

    result = delegate_in_process(make_request())

    assert review_server.AGENT_CARD.name == "Review & Recommendation Agent"
    assert received_requests == [make_request()]
    assert result == expected_response
    assert result.recommended_candidate_source_url == (
        "https://www.openstreetmap.org/node/123"
    )


def test_a2a_server_rejects_invalid_json_contract_without_running_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_run(_request: ReviewRecommendationRequest) -> None:
        pytest.fail("An invalid request must not run the recommendation Agent")

    monkeypatch.setattr(review_server, "_run_recommendation", unexpected_run)

    async def send_invalid_request() -> Any:
        async with review_server.app.router.lifespan_context(review_server.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=review_server.app)
            ) as http_client:
                card = await A2ACardResolver(
                    http_client,
                    review_server.A2A_URL,
                ).get_agent_card()
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
                            {"query": "Compare restaurants", "preferences": []},
                            media_type="application/json",
                            role=Role.ROLE_USER,
                        )
                    )
                    return [
                        response async for response in client.send_message(request)
                    ]
                finally:
                    await client.close()

    responses = asyncio.run(send_invalid_request())

    assert len(responses) == 1
    assert responses[0].task.status.state == TaskState.TASK_STATE_FAILED


def test_a2a_server_failure_is_reported_without_leaking_provider_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_run(
        _request: ReviewRecommendationRequest,
    ) -> ReviewRecommendationResponse:
        raise ReviewRecommendationAgentError("private provider detail")

    monkeypatch.setattr(review_server, "_run_recommendation", failed_run)

    with pytest.raises(
        ReviewRecommendationDelegationError,
        match="could not complete the request",
    ) as error:
        delegate_in_process(make_request())

    assert "private provider detail" not in str(error.value)


def test_a2a_client_converts_http_timeout_to_delegation_error() -> None:
    class TimeoutTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self,
            request: httpx.Request,
        ) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(
        ReviewRecommendationDelegationError,
        match="Could not communicate",
    ):
        asyncio.run(
            delegate_review_recommendation(
                make_request(),
                agent_url="http://review-agent.invalid",
                transport=TimeoutTransport(),
            )
        )


def test_a2a_client_rejects_malformed_json_artifact() -> None:
    payload = Struct()
    ParseDict({"comparisons": "not-a-list"}, payload)
    task = SimpleNamespace(
        status=SimpleNamespace(
            state=TaskState.TASK_STATE_COMPLETED,
            HasField=lambda _field: False,
        ),
        artifacts=[
            SimpleNamespace(
                name=review_server.REVIEW_RECOMMENDATION_ARTIFACT,
                parts=[
                    SimpleNamespace(
                        media_type="application/json",
                        data=payload,
                        WhichOneof=lambda _field: "data",
                    )
                ],
            )
        ],
    )

    with pytest.raises(
        ReviewRecommendationDelegationError,
        match="invalid result",
    ):
        _read_recommendation_result(task)


def test_discovery_agent_delegates_to_review_agent_over_a2a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = OsmRestaurantSearchResult(
        id=UUID(int=123),
        name="Casa Verde",
        city="Madrid",
        cuisine="vegetarian",
        location="Calle Mayor 1",
        latitude=40.4168,
        longitude=-3.7038,
        features=["Opciones vegetarianas: disponible"],
        source_url="https://www.openstreetmap.org/node/123",
        attribution="© OpenStreetMap contributors",
        attribution_url="https://www.openstreetmap.org/copyright",
        similarity=0.82,
    )
    recommendation = ReviewRecommendationResponse(
        recommended_candidate_source_url=candidate.source_url,
        comparisons=[
            CandidateComparison(
                candidate_source_url=candidate.source_url,
                assessment="matches",
                reasons=["The indexed cuisine matches the stated preference."],
                evidence=[
                    CandidateEvidence(
                        field="cuisine",
                        value="vegetarian",
                        preference="vegetarian cuisine",
                    )
                ],
                unresolved_preferences=[],
            )
        ],
        limitations=["No restaurant-linked review data is available."],
    )
    search_model_responses = iter(
        [
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id="search-1",
                                        name="search_restaurants",
                                        args={"query": "vegetarian restaurants"},
                                    )
                                )
                            ],
                        )
                    )
                ]
            ),
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id="recommend-1",
                                        name=restaurant_search_agent.REVIEW_RECOMMENDATION_FUNCTION,
                                        args={"preferences": ["vegetarian cuisine"]},
                                    )
                                )
                            ],
                        )
                    )
                ]
            ),
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    text=(
                                        "Casa Verde matches the stated vegetarian "
                                        "cuisine preference."
                                    )
                                )
                            ],
                        )
                    )
                ]
            ),
        ]
    )
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(
        restaurant_search_agent,
        "create_restaurant_search_router",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        review_recommendation_agent,
        "create_restaurant_search_router",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        restaurant_search_agent,
        "call_restaurant_search_model",
        lambda *_args, **_kwargs: next(search_model_responses),
    )
    monkeypatch.setattr(
        review_recommendation_agent,
        "call_restaurant_search_model",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=recommendation.model_dump_json()
        ),
    )
    monkeypatch.setattr(review_recommendation_agent, "search_documents", lambda *_a, **_k: [])
    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        lambda *_args: {
            "output": {
                "results": [candidate.model_dump(mode="json")],
                "evidence_status": "not_required",
                "evidence_message": None,
            }
        },
    )
    monkeypatch.setattr(
        review_server,
        "_run_recommendation",
        lambda request: review_recommendation_agent.run_review_recommendation_agent(
            request,
            object(),
        ),
    )
    monkeypatch.setattr(
        discovery_server,
        "_run_restaurant_search",
        lambda query: restaurant_search_agent.run_restaurant_search_agent(
            query,
            object(),
        ),
    )

    async def delegate_to_in_process_review_agent(request):
        return await delegate_review_recommendation(
            request,
            agent_url=review_server.A2A_URL,
            transport=httpx.ASGITransport(app=review_server.app),
        )

    monkeypatch.setattr(
        restaurant_search_agent,
        "delegate_review_recommendation",
        delegate_to_in_process_review_agent,
    )

    async def send_discovery_task() -> Any:
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(
                review_server.app.router.lifespan_context(review_server.app)
            )
            await stack.enter_async_context(
                discovery_server.app.router.lifespan_context(discovery_server.app)
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=discovery_server.app)
            ) as http_client:
                card = await A2ACardResolver(
                    http_client,
                    discovery_server.A2A_URL,
                ).get_agent_card()
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
                            "Compare and recommend a restaurant for vegetarian cuisine",
                            role=Role.ROLE_USER,
                        )
                    )
                    return [
                        response async for response in client.send_message(request)
                    ]
                finally:
                    await client.close()

    responses = asyncio.run(send_discovery_task())

    assert len(responses) == 1
    assert responses[0].task.status.state == TaskState.TASK_STATE_COMPLETED
    artifact = responses[0].task.artifacts[0]
    assert artifact.name == "restaurant_discovery_result"
    result = MessageToDict(artifact.parts[0].data)
    assert result["review_recommendation"]["recommended_candidate_source_url"] == (
        candidate.source_url
    )
    assert result["review_recommendation"]["limitations"]
