from collections.abc import Sequence
from typing import Any

import pytest
from google.genai import types
from pydantic import SecretStr

from app.agents import restaurant_search_agent
from app.core.config import settings
from app.schemas import ImageSearchResult, OsmRestaurantSearchResponse
from app.tools import registry


class FakeModels:
    def __init__(self, responses: Sequence[types.GenerateContentResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def generate_content(self, **request: Any) -> types.GenerateContentResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses: Sequence[types.GenerateContentResponse]) -> None:
        self.models = FakeModels(responses)
        self.closed = False

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


def function_call_response(
    name: str,
    arguments: dict[str, Any],
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call-1",
                                name=name,
                                args=arguments,
                            )
                        )
                    ],
                )
            )
        ]
    )


def text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=text)],
                )
            )
        ]
    )


def configure_client(
    monkeypatch: pytest.MonkeyPatch,
    *responses: types.GenerateContentResponse,
) -> FakeClient:
    client = FakeClient(responses)
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    monkeypatch.setattr(
        restaurant_search_agent.genai,
        "Client",
        lambda **kwargs: client,
    )
    return client


def test_function_declarations_hide_session_and_photo_corpus_selector() -> None:
    declarations = {
        declaration.name: declaration
        for declaration in registry.get_function_declarations()
    }

    assert set(declarations) == {
        "search_restaurants",
        "search_restaurant_photos",
        "search_documents",
        "answer_from_documents",
    }
    for declaration in declarations.values():
        schema = declaration.parameters_json_schema
        assert "session" not in schema["properties"]
    photo_properties = declarations["search_restaurant_photos"].parameters_json_schema[
        "properties"
    ]
    assert "osm_places_only" not in photo_properties


def test_dispatch_validates_arguments_and_injects_host_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    definition = registry.TOOL_DEFINITIONS["search_restaurants"]

    def fake_handler(**kwargs: Any) -> OsmRestaurantSearchResponse:
        calls.update(kwargs)
        return OsmRestaurantSearchResponse(
            results=[],
            evidence_status="not_required",
        )

    monkeypatch.setitem(
        registry.TOOL_DEFINITIONS,
        "search_restaurants",
        registry.ToolDefinition(
            name=definition.name,
            description=definition.description,
            input_model=definition.input_model,
            handler=fake_handler,
        ),
    )
    session = object()

    result = registry.dispatch_tool_call(
        "search_restaurants",
        {"query": "vegetarian food", "top_k": 3},
        session,
    )

    assert result == {
        "output": {
            "results": [],
            "evidence_status": "not_required",
            "evidence_message": None,
        }
    }
    assert calls == {
        "query": "vegetarian food",
        "top_k": 3,
        "city": None,
        "cuisine": None,
        "session": session,
    }


def test_dispatch_rejects_unknown_and_hidden_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = registry.TOOL_DEFINITIONS["search_restaurant_photos"]
    calls: list[bool] = []
    monkeypatch.setitem(
        registry.TOOL_DEFINITIONS,
        "search_restaurant_photos",
        registry.ToolDefinition(
            name=definition.name,
            description=definition.description,
            input_model=definition.input_model,
            handler=lambda **kwargs: calls.append(True),
            hidden_parameters=definition.hidden_parameters,
        ),
    )

    unknown = registry.dispatch_tool_call("not_a_tool", {}, object())
    hidden = registry.dispatch_tool_call(
        "search_restaurant_photos",
        {"query": "restaurant", "osm_places_only": False},
        object(),
    )

    assert unknown["error"]["code"] == "unknown_tool"
    assert hidden["error"]["code"] == "invalid_arguments"
    assert calls == []


def test_photo_dispatch_omits_hidden_corpus_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = registry.TOOL_DEFINITIONS["search_restaurant_photos"]
    received_arguments: dict[str, Any] = {}

    def fake_handler(
        query: str,
        session: Any,
        *,
        top_k: int = 5,
        city: str | None = None,
        cuisine: str | None = None,
    ) -> list[ImageSearchResult]:
        received_arguments.update(
            query=query,
            session=session,
            top_k=top_k,
            city=city,
            cuisine=cuisine,
        )
        return []

    monkeypatch.setitem(
        registry.TOOL_DEFINITIONS,
        "search_restaurant_photos",
        registry.ToolDefinition(
            name=definition.name,
            description=definition.description,
            input_model=definition.input_model,
            handler=fake_handler,
            hidden_parameters=definition.hidden_parameters,
        ),
    )
    session = object()

    result = registry.dispatch_tool_call(
        "search_restaurant_photos",
        {"query": "restaurant photos"},
        session,
    )

    assert result == {"output": []}
    assert received_arguments == {
        "query": "restaurant photos",
        "session": session,
        "top_k": 5,
        "city": None,
        "cuisine": None,
    }


def test_dispatch_logs_tool_failure_and_returns_explicit_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    definition = registry.TOOL_DEFINITIONS["search_restaurants"]

    def failing_handler(**kwargs: Any) -> OsmRestaurantSearchResponse:
        raise RuntimeError("internal failure")

    monkeypatch.setitem(
        registry.TOOL_DEFINITIONS,
        "search_restaurants",
        registry.ToolDefinition(
            name=definition.name,
            description=definition.description,
            input_model=definition.input_model,
            handler=failing_handler,
        ),
    )

    result = registry.dispatch_tool_call(
        "search_restaurants",
        {"query": "vegetarian food"},
        object(),
    )

    assert result == {
        "error": {
            "code": "tool_execution_failed",
            "message": "The tool failed. Do not infer or invent missing results.",
        }
    }
    assert "internal failure" not in str(result)
    assert "Tool execution failed: search_restaurants" in caplog.text


def test_photo_dispatch_never_serializes_local_image_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = registry.TOOL_DEFINITIONS["search_restaurant_photos"]
    photo = ImageSearchResult(
        id="e4ada293-47d1-492e-b6f3-ff27aa6624d1",
        source_name="commons",
        image_path="C:\\private\\photo.jpg",
        similarity=0.9,
    )
    monkeypatch.setitem(
        registry.TOOL_DEFINITIONS,
        "search_restaurant_photos",
        registry.ToolDefinition(
            name=definition.name,
            description=definition.description,
            input_model=definition.input_model,
            handler=lambda **kwargs: [photo],
            hidden_parameters=definition.hidden_parameters,
        ),
    )

    result = registry.dispatch_tool_call(
        "search_restaurant_photos",
        {"query": "restaurant"},
        object(),
    )

    assert result["output"][0]["source_name"] == "commons"
    assert "image_path" not in result["output"][0]
    assert result["output"][0]["image_url"] == "/images/files/photo.jpg"
    assert "C:\\private\\photo.jpg" not in str(result)


def test_agent_returns_photo_results_as_safe_api_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurant_photos",
            {"query": "vegetarian restaurant in Madrid", "top_k": 1},
        ),
        text_response("He encontrado una foto del candidato."),
    )
    photo_result = {
        "image_url": "/images/files/commons-photo.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:commons-photo.jpg",
        "license_name": "CC BY-SA",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Photographer",
        "restaurant_name": "Example Restaurant",
        "restaurant_location": "Madrid",
        "restaurant_cuisine": "vegetarian",
        "restaurant_source_url": "https://www.openstreetmap.org/node/123",
        "restaurant_attribution": "© OpenStreetMap contributors",
        "restaurant_attribution_url": "https://www.openstreetmap.org/copyright",
    }

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        lambda *args: {"output": [photo_result]},
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert response.photos[0].image_url == "/images/files/commons-photo.jpg"
    assert response.photos[0].restaurant_name == "Example Restaurant"
    assert response.photos[0].license_url == photo_result["license_url"]
    assert "image_path" not in str(client.models.requests[-1]["contents"])
    assert "MUST call" in client.models.requests[0]["config"].system_instruction
    function_response = client.models.requests[-1]["contents"][-1].parts[0]
    assert function_response.function_response.response["output"][0]["image_url"] == (
        "/images/files/commons-photo.jpg"
    )


def test_agent_filters_photos_to_candidates_and_deduplicates_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurants",
            {"query": "vegetarian restaurants in Madrid"},
        ),
        function_call_response(
            "search_restaurant_photos",
            {"query": "vegetarian restaurants in Madrid"},
        ),
        text_response("He encontrado una foto del candidato vegetariano."),
    )
    matching_photo = {
        "image_url": "/images/files/green-place.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:green-place.jpg",
        "restaurant_name": "Restaurante Verde",
        "restaurant_cuisine": "vegetarian",
    }
    wrong_restaurant_photo = {
        "image_url": "/images/files/casa-mingo.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:casa-mingo.jpg",
        "restaurant_name": "Casa Mingo",
        "restaurant_cuisine": "chicken",
    }
    duplicate_photo = {
        **matching_photo,
        "source_url": "https://commons.wikimedia.org/wiki/File:green-place-copy.jpg",
    }

    def fake_dispatch(name: str | None, args: Any, session: Any):
        if name == "search_restaurants":
            return {
                "output": {
                    "results": [
                        {"name": "Restaurante Verde", "cuisine": "vegetarian"}
                    ]
                }
            }
        return {
            "output": [
                matching_photo,
                wrong_restaurant_photo,
                duplicate_photo,
            ]
        }

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos de algunos.",
        object(),
    )

    assert len(response.photos) == 1
    assert response.photos[0].restaurant_name == "Restaurante Verde"
    assert response.photos[0].restaurant_cuisine == "vegetarian"
    assert "Casa Mingo" not in str(client.models.requests[-1]["contents"])
    photo_result = next(
        part.function_response.response
        for content in client.models.requests[-1]["contents"]
        for part in content.parts or []
        if part.function_response is not None
        and part.function_response.name == "search_restaurant_photos"
    )
    assert len(photo_result["output"]) == 1
    assert photo_result["output"][0]["restaurant_name"] == "Restaurante Verde"


def test_agent_forces_photo_search_when_model_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurants",
            {"query": "vegetarian restaurants in Madrid"},
        ),
        text_response("He encontrado candidatos vegetarianos."),
    )
    matching_photo = {
        "image_url": "/images/files/green-place.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:green-place.jpg",
        "restaurant_name": "Restaurante Verde",
        "restaurant_cuisine": "vegetarian",
    }
    unrelated_photo = {
        "image_url": "/images/files/casa-mingo.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:casa-mingo.jpg",
        "restaurant_name": "Casa Mingo",
        "restaurant_cuisine": "chicken",
    }
    dispatched: list[str | None] = []

    def fake_dispatch(name: str | None, args: Any, session: Any):
        dispatched.append(name)
        if name == "search_restaurants":
            return {
                "output": {
                    "results": [{"name": "Restaurante Verde"}],
                }
            }
        return {"output": [unrelated_photo, matching_photo]}

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert dispatched == ["search_restaurants", "search_restaurant_photos"]
    assert [photo.restaurant_name for photo in response.photos] == [
        "Restaurante Verde"
    ]
    assert "No he encontrado fotos" not in response.answer


def test_agent_explains_when_no_candidate_photos_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurants",
            {"query": "vegetarian restaurants in Madrid"},
        ),
        text_response("He encontrado candidatos vegetarianos."),
    )

    def fake_dispatch(name: str | None, args: Any, session: Any):
        if name == "search_restaurants":
            return {"output": {"results": [{"name": "Restaurante Verde"}]}}
        return {
            "output": [
                {
                    "image_url": "/images/files/casa-mingo.jpg",
                    "source_url": (
                        "https://commons.wikimedia.org/wiki/File:casa-mingo.jpg"
                    ),
                    "restaurant_name": "Casa Mingo",
                    "restaurant_cuisine": "chicken",
                }
            ]
        }

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert response.photos == []
    assert "No he encontrado fotos indexadas asociadas a los restaurantes candidatos." in (
        response.answer
    )


def test_agent_leads_with_unavailable_live_data_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_client(
        monkeypatch,
        text_response("Puedo buscarte candidatos según los datos disponibles."),
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Dime cuál tiene disponibilidad de mesa esta noche y cuánto cuesta el menú.",
        object(),
    )

    assert response.answer.startswith(
        "No puedo verificar la disponibilidad de mesa esta noche ni "
        "los precios actuales del menú con las herramientas disponibles."
    )


def test_agent_executes_tool_call_and_returns_model_final_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurants",
            {"query": "vegetarian food", "top_k": 3},
        ),
        text_response("Encontré candidatos vegetarianos en Madrid."),
    )
    session = object()
    dispatched: list[tuple[str | None, Any, Any]] = []

    def fake_dispatch(name: str | None, args: Any, received_session: Any):
        dispatched.append((name, args, received_session))
        return {"output": {"results": [{"name": "Example"}]}}

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos",
        session,
    )

    assert response.answer == "Encontré candidatos vegetarianos en Madrid."
    assert dispatched == [
        (
            "search_restaurants",
            {"query": "vegetarian food", "top_k": 3},
            session,
        )
    ]
    assert client.closed
    first_request, final_request = client.models.requests
    assert first_request["model"] == "gemini-3.8-flash"
    assert first_request["config"].automatic_function_calling.disable is True
    assert first_request["config"].tool_config.function_calling_config.mode == (
        types.FunctionCallingConfigMode.AUTO
    )
    assert final_request["contents"][-1].parts[0].function_response.response == {
        "output": {"results": [{"name": "Example"}]}
    }
    assert final_request["contents"][-1].parts[0].function_response.id == "call-1"
    assert all(
        repr(session) not in repr(request)
        for request in client.models.requests
    )


def test_agent_caps_tool_calls_and_asks_for_final_answer_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(
        monkeypatch,
        *[
            function_call_response("search_documents", {"query": "menus"})
            for _ in range(5)
        ],
        text_response("No puedo realizar más búsquedas en esta petición."),
    )
    dispatch_count = 0

    def fake_dispatch(name: str | None, args: Any, session: Any):
        nonlocal dispatch_count
        dispatch_count += 1
        return {"output": []}

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca documentos",
        object(),
    )

    assert response.answer == "No puedo realizar más búsquedas en esta petición."
    assert dispatch_count == restaurant_search_agent.MAX_TOOL_CALLS
    final_request = client.models.requests[-1]
    assert final_request["config"].tools is None
    assert final_request["config"].tool_config.function_calling_config.mode == (
        types.FunctionCallingConfigMode.NONE
    )
    rejected_result = client.models.requests[-1]["contents"][-1].parts[0]
    assert rejected_result.function_response.response["error"]["code"] == (
        "tool_call_limit_reached"
    )


def test_agent_surfaces_tool_failure_to_model_without_fabricating_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(
        monkeypatch,
        function_call_response("search_documents", {"query": "menus"}),
        text_response("La búsqueda documental falló; no tengo resultados."),
    )
    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        lambda *args: {
            "error": {
                "code": "tool_execution_failed",
                "message": "The tool failed.",
            }
        },
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca el menú",
        object(),
    )

    assert response.answer == "La búsqueda documental falló; no tengo resultados."
    tool_result = client.models.requests[-1]["contents"][-1].parts[0]
    assert tool_result.function_response.response["error"]["code"] == (
        "tool_execution_failed"
    )


def test_agent_rejects_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", None)

    with pytest.raises(
        restaurant_search_agent.RestaurantSearchAgentError,
        match="GEMINI_API_KEY",
    ):
        restaurant_search_agent.run_restaurant_search_agent("question", object())


def test_agent_rejects_empty_model_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(monkeypatch, text_response("  "))

    with pytest.raises(
        restaurant_search_agent.RestaurantSearchAgentError,
        match="empty Agent answer",
    ):
        restaurant_search_agent.run_restaurant_search_agent("question", object())

    assert client.closed
