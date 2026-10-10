import json
import logging
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from google.genai import types
from litellm import Router
from litellm.types.utils import ModelResponse

from app.infrastructure.observability import traced_span
from app.tools.registry import get_function_declarations

logger = logging.getLogger(__name__)

RESTAURANT_SEARCH_MODEL_ALIAS = "restaurant-search-agent"
RESTAURANT_SEARCH_LITELLM_MODEL = "gemini/gemini-3.8-flash"
RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS = "restaurant-search-openai-fallback"
RESTAURANT_SEARCH_OPENAI_FALLBACK_MODEL = "openai/gpt-4.1-mini"
RESTAURANT_SEARCH_REQUEST_TIMEOUT_SECONDS = 60
RESTAURANT_SEARCH_MAX_OUTPUT_TOKENS = 1024


class ModelRouteError(RuntimeError):
    """The configured model route could not produce a usable response."""


def create_restaurant_search_router(
    api_key: str,
    openai_api_key: str | None = None,
) -> Router:
    model_list: list[dict[str, Any]] = [
        {
            "model_name": RESTAURANT_SEARCH_MODEL_ALIAS,
            "litellm_params": {
                "model": RESTAURANT_SEARCH_LITELLM_MODEL,
                "api_key": api_key,
                "timeout": RESTAURANT_SEARCH_REQUEST_TIMEOUT_SECONDS,
            },
        }
    ]
    fallbacks: list[dict[str, list[str]]] = []
    if openai_api_key is not None:
        model_list.append(
            {
                "model_name": RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS,
                "litellm_params": {
                    "model": RESTAURANT_SEARCH_OPENAI_FALLBACK_MODEL,
                    "api_key": openai_api_key,
                    "timeout": RESTAURANT_SEARCH_REQUEST_TIMEOUT_SECONDS,
                },
            }
        )
        fallbacks.append(
            {
                RESTAURANT_SEARCH_MODEL_ALIAS: [
                    RESTAURANT_SEARCH_OPENAI_FALLBACK_ALIAS
                ]
            }
        )

    return Router(
        model_list=model_list,
        fallbacks=fallbacks,
        cache_responses=False,
        num_retries=0,
        max_fallbacks=1,
        set_verbose=False,
    )


def call_restaurant_search_model(
    router: Router,
    contents: Sequence[types.Content],
    system_instruction: str,
    *,
    tools_enabled: bool,
) -> types.GenerateContentResponse:
    request: dict[str, Any] = {
        "model": RESTAURANT_SEARCH_MODEL_ALIAS,
        "messages": _to_litellm_messages(contents, system_instruction),
        "temperature": 0.2,
        "max_tokens": RESTAURANT_SEARCH_MAX_OUTPUT_TOKENS,
        "stream": False,
    }
    if tools_enabled:
        request["tools"] = _litellm_tools()
        request["tool_choice"] = "auto"

    try:
        with traced_span(
            "gen_ai.chat",
            {
                "gen_ai.request.model": RESTAURANT_SEARCH_LITELLM_MODEL,
                "gen_ai.request.max_tokens": RESTAURANT_SEARCH_MAX_OUTPUT_TOKENS,
            },
        ) as span:
            response = router.completion(**request)
            response_model = getattr(response, "model", None)
            if isinstance(response_model, str):
                span.set_attribute("gen_ai.response.model", response_model)
            usage = getattr(response, "usage", None)
            if isinstance(response, ModelResponse) and usage is not None:
                if usage.prompt_tokens is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
                if usage.completion_tokens is not None:
                    span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        usage.completion_tokens,
                    )
                if usage.total_tokens is not None:
                    span.set_attribute("gen_ai.usage.total_tokens", usage.total_tokens)
    except Exception as error:
        logger.error(
            "Restaurant search model route failed with %s",
            type(error).__name__,
        )
        raise ModelRouteError(
            "The restaurant search Agent could not obtain a response from any configured model route."
        ) from error

    if not isinstance(response, ModelResponse):
        raise ModelRouteError("The restaurant search model route must be non-streaming.")
    return _to_google_response(response)


def _to_litellm_messages(
    contents: Sequence[types.Content],
    system_instruction: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_instruction}
    ]
    for content in contents:
        parts = content.parts or []
        if content.role == "user":
            function_responses = [
                part.function_response
                for part in parts
                if part.function_response is not None
            ]
            if function_responses:
                if len(function_responses) != len(parts):
                    raise ModelRouteError(
                        "The Agent conversation contains mixed tool and text results."
                    )
                for function_response in function_responses:
                    if not function_response.id:
                        raise ModelRouteError(
                            "A tool result is missing its function-call identifier."
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": function_response.id,
                            "content": json.dumps(
                                function_response.response,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        }
                    )
                continue

            text_parts = [part.text for part in parts if part.text is not None]
            if len(text_parts) != len(parts) or not text_parts:
                raise ModelRouteError(
                    "The Agent conversation contains unsupported user content."
                )
            messages.append({"role": "user", "content": "\n".join(text_parts)})
            continue

        if content.role == "model":
            text_parts = [part.text for part in parts if part.text is not None]
            function_calls = [
                part.function_call
                for part in parts
                if part.function_call is not None
            ]
            if len(text_parts) + len(function_calls) != len(parts):
                raise ModelRouteError(
                    "The Agent conversation contains unsupported model content."
                )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if function_calls:
                tool_calls: list[dict[str, Any]] = []
                for function_call in function_calls:
                    if not function_call.id or not function_call.name:
                        raise ModelRouteError(
                            "A model tool call is missing its identifier or name."
                        )
                    arguments = function_call.args or {}
                    if not isinstance(arguments, dict):
                        raise ModelRouteError(
                            "A model tool call returned invalid arguments."
                        )
                    tool_calls.append(
                        {
                            "id": function_call.id,
                            "type": "function",
                            "function": {
                                "name": function_call.name,
                                "arguments": json.dumps(
                                    arguments,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                            },
                        }
                    )
                assistant_message["tool_calls"] = tool_calls
            if not text_parts and not function_calls:
                raise ModelRouteError(
                    "The Agent conversation contains an empty model message."
                )
            messages.append(assistant_message)
            continue

        raise ModelRouteError("The Agent conversation contains an unknown role.")

    return messages


def _litellm_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for declaration in get_function_declarations():
        parameters = declaration.parameters_json_schema
        if not isinstance(parameters, dict):
            raise ModelRouteError(
                f"The {declaration.name} tool has no JSON Schema parameters."
            )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": declaration.name,
                    "description": declaration.description or "",
                    "parameters": parameters,
                },
            }
        )
    return tools


def _to_google_response(
    response: ModelResponse,
) -> types.GenerateContentResponse:
    if not response.choices:
        return types.GenerateContentResponse()

    message = response.choices[0].message
    if message.content is not None and not isinstance(message.content, str):
        raise ModelRouteError("Gemini returned a non-text Agent answer.")

    parts: list[types.Part] = []
    if message.content is not None:
        parts.append(types.Part(text=message.content))

    for tool_call in message.tool_calls or []:
        name = tool_call.function.name
        if not name:
            raise ModelRouteError("Gemini returned a tool call without a name.")
        try:
            arguments = json.loads(tool_call.function.arguments)
        except (TypeError, json.JSONDecodeError):
            raise ModelRouteError(
                "Gemini returned invalid JSON for a tool call."
            ) from None
        if not isinstance(arguments, dict):
            raise ModelRouteError("Gemini returned invalid arguments for a tool call.")
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=tool_call.id or uuid4().hex,
                    name=name,
                    args=arguments,
                )
            )
        )

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=parts,
                )
            )
        ]
    )