from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, NoReturn

import pytest
from google.genai import types
from litellm import Router
from litellm.types.utils import ModelResponse

import app.model_routing as model_routing
from app.model_routing import (
    ModelRouteError,
    RESTAURANT_SEARCH_LITELLM_MODEL,
    RESTAURANT_SEARCH_MODEL_ALIAS,
    RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS,
    RESTAURANT_SEARCH_OPENAI_FALLBACK_MODEL,
    RESTAURANT_SEARCH_REQUEST_TIMEOUT_SECONDS,
    RESTAURANT_SEARCH_MAX_OUTPUT_TOKENS,
    call_restaurant_search_model,
    create_restaurant_search_router,
)


def test_restaurant_search_router_has_one_gemini_route() -> None:
    router = create_restaurant_search_router("synthetic-api-key")

    assert len(router.model_list) == 1
    route = router.model_list[0]
    assert route["model_name"] == RESTAURANT_SEARCH_MODEL_ALIAS
    assert route["litellm_params"]["model"] == RESTAURANT_SEARCH_LITELLM_MODEL
    assert (
        route["litellm_params"]["timeout"]
        == RESTAURANT_SEARCH_REQUEST_TIMEOUT_SECONDS
    )
    assert router.num_retries == 0
    assert router.max_fallbacks == 1
    assert router.fallbacks is None
    assert router.cache_responses is False
    assert router.set_verbose is False


def test_restaurant_search_router_configures_openai_fallback() -> None:
    router = create_restaurant_search_router(
        "synthetic-gemini-api-key",
        openai_api_key="synthetic-openai-api-key",
    )

    assert [route["model_name"] for route in router.model_list] == [
        RESTAURANT_SEARCH_MODEL_ALIAS,
        RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS,
    ]
    fallback_route = router.model_list[1]
    assert (
        fallback_route["litellm_params"]["model"]
        == RESTAURANT_SEARCH_OPENAI_FALLBACK_MODEL
    )
    assert (
        fallback_route["litellm_params"]["api_key"]
        == "synthetic-openai-api-key"
    )
    assert router.fallbacks == [
        {
            RESTAURANT_SEARCH_MODEL_ALIAS: [
                RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS
            ]
        }
    ]


def test_model_route_failure_is_sanitized() -> None:
    class FailedRouter(Router):
        def __init__(self) -> None:
            pass

        def completion(
            self,
            model: str,
            messages: list[dict[str, str]],
            **_kwargs: Any,
        ) -> NoReturn:
            del model, messages, _kwargs
            raise RuntimeError("private provider response")

    with pytest.raises(
        ModelRouteError,
        match="could not obtain a response from any configured model route",
    ) as error:
        call_restaurant_search_model(
            FailedRouter(),
            [
                types.Content(
                    role="user",
                    parts=[types.Part(text="synthetic query")],
                )
            ],
            "synthetic system instruction",
            tools_enabled=False,
        )

    assert "private provider response" not in str(error.value)


def test_model_route_uses_openai_after_gemini_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = create_restaurant_search_router(
        "synthetic-gemini-api-key",
        openai_api_key="synthetic-openai-api-key",
    )
    attempted_routes: list[str] = []

    def synthetic_completion(**kwargs: Any) -> ModelResponse:
        route = kwargs["model"]
        attempted_routes.append(route)
        if route == RESTAURANT_SEARCH_MODEL_ALIAS:
            raise RuntimeError("synthetic Gemini failure")
        assert route == RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS
        return ModelResponse(
            id="synthetic-id",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthetic answer"},
                    "finish_reason": "stop",
                }
            ],
            model=RESTAURANT_SEARCH_OPENAI_FALLBACK_MODEL,
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        )

    monkeypatch.setattr(router, "_completion", synthetic_completion)
    response = call_restaurant_search_model(
        router,
        [
            types.Content(
                role="user",
                parts=[types.Part(text="synthetic query")],
            )
        ],
        "synthetic system instruction",
        tools_enabled=False,
    )

    assert attempted_routes == [
        RESTAURANT_SEARCH_MODEL_ALIAS,
        RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS,
    ]
    assert response.candidates[0].content.parts[0].text == "synthetic answer"


def test_model_usage_is_traced_without_conversation_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attributes: dict[str, Any] = {}

    class CapturingSpan:
        def set_attribute(self, key: str, value: int) -> None:
            attributes[key] = value

    @contextmanager
    def fake_traced_span(
        name: str,
        initial_attributes: dict[str, Any],
    ) -> Generator[CapturingSpan, None, None]:
        assert name == "gen_ai.chat"
        attributes.update(initial_attributes)
        yield CapturingSpan()

    class SuccessfulRouter(Router):
        def __init__(self) -> None:
            pass

        def completion(
            self,
            model: str,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> ModelResponse:
            assert model == RESTAURANT_SEARCH_MODEL_ALIAS
            assert messages[-1]["content"] == "synthetic query"
            assert kwargs["max_tokens"] == RESTAURANT_SEARCH_MAX_OUTPUT_TOKENS
            return ModelResponse(
                id="synthetic-id",
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthetic answer"},
                        "finish_reason": "stop",
                    }
                ],
                model=RESTAURANT_SEARCH_LITELLM_MODEL,
                usage={
                    "prompt_tokens": 11,
                    "completion_tokens": 4,
                    "total_tokens": 15,
                },
            )

    monkeypatch.setattr(model_routing, "traced_span", fake_traced_span)
    call_restaurant_search_model(
        SuccessfulRouter(),
        [
            types.Content(
                role="user",
                parts=[types.Part(text="synthetic query")],
            )
        ],
        "synthetic system instruction",
        tools_enabled=False,
    )

    assert attributes == {
        "gen_ai.request.model": RESTAURANT_SEARCH_LITELLM_MODEL,
        "gen_ai.request.max_tokens": RESTAURANT_SEARCH_MAX_OUTPUT_TOKENS,
        "gen_ai.response.model": RESTAURANT_SEARCH_LITELLM_MODEL,
        "gen_ai.usage.input_tokens": 11,
        "gen_ai.usage.output_tokens": 4,
        "gen_ai.usage.total_tokens": 15,
    }
    assert "synthetic query" not in str(attributes)
    assert "synthetic answer" not in str(attributes)
