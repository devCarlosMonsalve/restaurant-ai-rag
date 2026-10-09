from typing import Any, NoReturn

import pytest
from google.genai import types
from litellm import Router

from app.model_routing import (
    ModelRouteError,
    RESTAURANT_SEARCH_LITELLM_MODEL,
    RESTAURANT_SEARCH_MODEL_ALIAS,
    call_restaurant_search_model,
    create_restaurant_search_router,
)


def test_restaurant_search_router_has_one_gemini_route() -> None:
    router = create_restaurant_search_router("synthetic-api-key")

    assert len(router.model_list) == 1
    route = router.model_list[0]
    assert route["model_name"] == RESTAURANT_SEARCH_MODEL_ALIAS
    assert route["litellm_params"]["model"] == RESTAURANT_SEARCH_LITELLM_MODEL
    assert router.num_retries == 0
    assert router.cache_responses is False
    assert router.set_verbose is False


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

    with pytest.raises(ModelRouteError, match="Gemini could not complete") as error:
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
