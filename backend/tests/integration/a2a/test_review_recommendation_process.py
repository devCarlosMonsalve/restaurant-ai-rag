import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import UUID

import httpx
import pytest
from a2a.client import A2ACardResolver
from a2a.client.errors import A2AClientError
from google.genai import types
from pydantic import SecretStr


_REVIEW_SERVER_BOOTSTRAP = """\
import uvicorn

from app.agents.schemas import (
    CandidateComparison,
    CandidateEvidence,
    ReviewRecommendationResponse,
)
from app.interfaces.a2a import review_recommendation_server as server


def deterministic_recommendation(request):
    comparisons = [
        CandidateComparison(
            candidate_source_url=candidate.source_url,
            assessment="matches",
            reasons=[
                f"cuisine={candidate.cuisine}; preference={request.preferences[0]}"
            ],
            evidence=[
                CandidateEvidence(
                    field="cuisine",
                    value=candidate.cuisine,
                    preference=request.preferences[0],
                )
            ],
            unresolved_preferences=[],
        )
        for candidate in request.candidates
    ]
    return ReviewRecommendationResponse(
        recommended_candidate_source_url=request.candidates[0].source_url,
        comparisons=comparisons,
        limitations=["Deterministic process-integration response; no model used."],
    )


server._run_recommendation = deterministic_recommendation
uvicorn.run(server.app, host=server.A2A_HOST, port=server.A2A_PORT, log_level="error")
"""


def _test_candidate():
    from app.restaurant_discovery.application.contracts import (
        OsmRestaurantSearchResult,
    )

    return OsmRestaurantSearchResult(
        id=UUID(int=123),
        name="Casa Verde",
        city="Madrid",
        cuisine="vegetarian",
        location="Calle Mayor 1",
        latitude=40.4168,
        longitude=-3.7038,
        features=["Vegetarian options: available"],
        source_url="https://www.openstreetmap.org/node/123",
        attribution="© OpenStreetMap contributors",
        attribution_url="https://www.openstreetmap.org/copyright",
        similarity=0.82,
    )


async def _resolve_card(agent_url: str):
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(1.0, connect=0.5),
        trust_env=False,
    ) as http_client:
        return await A2ACardResolver(http_client, agent_url).get_agent_card()


def _wait_for_agent_card(
    process: subprocess.Popen[str],
    agent_url: str,
):
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(
                f"Review A2A process exited with {process.returncode}: {output}"
            )
        try:
            return asyncio.run(_resolve_card(agent_url))
        except (A2AClientError, httpx.HTTPError) as error:
            last_error = error
            time.sleep(0.1)
    raise AssertionError(
        f"Review A2A Agent Card was not reachable: {last_error}"
    )


@pytest.mark.integration
def test_restaurant_search_agent_delegates_to_review_agent_in_another_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "false")

    from app.agents import restaurant_search_agent
    from app.core.config import settings
    from app.interfaces.a2a import review_recommendation_server as review_server

    host, port = "127.0.0.1", 8003
    with socket.socket() as listener:
        try:
            listener.bind((host, port))
        except OSError:
            pytest.skip("The Review A2A port 8003 is already in use.")

    backend_root = Path(__file__).resolve().parents[3]
    server_environment = os.environ.copy()
    server_environment["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"
    server_environment["PHOENIX_TRACING_ENABLED"] = "false"
    server_environment.pop("GEMINI_API_KEY", None)
    server_environment.pop("OPENAI_API_KEY", None)
    process = subprocess.Popen(
        [sys.executable, "-c", _REVIEW_SERVER_BOOTSTRAP],
        cwd=backend_root,
        env=server_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        card = _wait_for_agent_card(process, review_server.A2A_URL)
        assert card.name == "Review & Recommendation Agent"
        assert card.supported_interfaces[0].protocol_binding == "JSONRPC"
        assert card.supported_interfaces[0].url == review_server.A2A_URL
        assert card.default_input_modes == ["application/json"]
        assert card.default_output_modes == ["application/json"]

        candidate = _test_candidate()
        model_responses = iter(
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
                                            args={
                                                "preferences": [
                                                    "vegetarian cuisine"
                                                ]
                                            },
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
                                            "Casa Verde matches the vegetarian "
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

        monkeypatch.setattr(
            settings,
            "gemini_api_key",
            SecretStr("test-only-placeholder"),
        )
        monkeypatch.setattr(settings, "openai_api_key", None)
        monkeypatch.setattr(
            settings,
            "review_recommendation_agent_url",
            review_server.A2A_URL,
        )
        monkeypatch.setattr(
            restaurant_search_agent,
            "create_restaurant_search_router",
            lambda *_args, **_kwargs: object(),
        )
        monkeypatch.setattr(
            restaurant_search_agent,
            "call_restaurant_search_model",
            lambda *_args, **_kwargs: next(model_responses),
        )

        def dispatch_search(name, _arguments, _session):
            assert name == "search_restaurants"
            return {
                "output": {
                    "results": [candidate.model_dump(mode="json")],
                    "evidence_status": "not_required",
                    "evidence_message": None,
                }
            }

        monkeypatch.setattr(
            restaurant_search_agent,
            "dispatch_tool_call",
            dispatch_search,
        )

        result = restaurant_search_agent.run_restaurant_search_agent(
            "Compare and recommend a restaurant for vegetarian cuisine",
            object(),
        )

        assert result.restaurants[0].source_url == candidate.source_url
        assert result.review_recommendation is not None
        assert (
            result.review_recommendation.recommended_candidate_source_url
            == candidate.source_url
        )
        assert (
            result.review_recommendation.comparisons[0].assessment == "matches"
        )
        assert result.review_recommendation.limitations == [
            "Deterministic process-integration response; no model used."
        ]
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
