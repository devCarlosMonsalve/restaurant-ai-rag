import json
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest
from google.genai import types
from litellm.types.utils import ModelResponse
from pydantic import SecretStr

from app.agents import restaurant_search_agent
from app.core.config import settings
from app.schemas import (
    ImageSearchResult,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
)
from app.tools import registry


class FakeRouter:
    def __init__(self, responses: Sequence[types.GenerateContentResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def completion(self, **request: Any) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else []
        text_parts = [part.text for part in parts or [] if part.text is not None]
        tool_calls = [
            {
                "id": part.function_call.id,
                "type": "function",
                "function": {
                    "name": part.function_call.name,
                    "arguments": json.dumps(part.function_call.args or {}),
                },
            }
            for part in parts or []
            if part.function_call is not None
        ]
        return ModelResponse(
            choices=(
                [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "\n".join(text_parts) if text_parts else None,
                            "tool_calls": tool_calls or None,
                        },
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ]
                if candidate
                else []
            )
        )


def find_tool_result(
    request: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    tool_call_names = {
        tool_call["id"]: tool_call["function"]["name"]
        for message in request["messages"]
        for tool_call in message.get("tool_calls", [])
    }
    for message in reversed(request["messages"]):
        if (
            message["role"] == "tool"
            and tool_call_names.get(message["tool_call_id"]) == tool_name
        ):
            return json.loads(message["content"])
    raise AssertionError(f"No result found for tool {tool_name}")


def find_tool_call(
    request: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    for message in reversed(request["messages"]):
        for tool_call in message.get("tool_calls", []):
            if tool_call["function"]["name"] == tool_name:
                return tool_call
    raise AssertionError(f"No call found for tool {tool_name}")


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
) -> FakeRouter:
    client = FakeRouter(responses)
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    monkeypatch.setattr(settings, "openai_api_key", None)

    def fake_router(
        api_key: str,
        *,
        openai_api_key: str | None = None,
    ) -> FakeRouter:
        del api_key
        assert openai_api_key is None
        return client

    monkeypatch.setattr(
        restaurant_search_agent,
        "create_restaurant_search_router",
        fake_router,
    )
    return client


def make_candidate(name: str, osm_id: int) -> OsmRestaurantSearchResult:
    return OsmRestaurantSearchResult(
        id=UUID(int=osm_id),
        name=name,
        city="Madrid",
        cuisine="vegetarian",
        location="Calle Mayor 1",
        latitude=40.4168,
        longitude=-3.7038,
        features=[],
        source_url=f"https://www.openstreetmap.org/node/{osm_id}",
        attribution="© OpenStreetMap contributors",
        attribution_url="https://www.openstreetmap.org/copyright",
        similarity=0.82,
    )


def make_indexed_photo(restaurant_url: str, image_id: int) -> ImageSearchResult:
    return ImageSearchResult(
        id=UUID(int=image_id),
        source_name=f"{image_id}.jpg",
        image_path=f"C:/images/{image_id}.jpg",
        similarity=0.9,
        source_url=f"https://commons.wikimedia.org/wiki/File:{image_id}.jpg",
        license_name="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        attribution="Photographer",
        restaurant_name="Casa Verde",
        restaurant_location="Calle Mayor 1",
        restaurant_cuisine="vegetarian",
        restaurant_source_url=restaurant_url,
        restaurant_attribution="© OpenStreetMap contributors",
        restaurant_attribution_url="https://www.openstreetmap.org/copyright",
    )


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
    assert "osm_place_id" not in photo_properties


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
    assert "image_path" not in str(client.requests[-1]["messages"])
    assert "MUST call" in client.requests[0]["messages"][0]["content"]
    photo_result = find_tool_result(
        client.requests[-1],
        "search_restaurant_photos",
    )
    assert photo_result["output"][0]["image_url"] == "/images/files/commons-photo.jpg"


def test_agent_scopes_photos_by_candidate_id_and_exact_osm_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_candidate = make_candidate("Casa Verde", 101)
    second_candidate = make_candidate("Casa Verde", 202)
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
    requested_place_ids: list[UUID] = []

    def fake_dispatch(name: str | None, args: Any, session: Any):
        if name == "search_restaurants":
            return {
                "output": {
                    "results": [
                        first_candidate.model_dump(mode="json"),
                        second_candidate.model_dump(mode="json"),
                    ],
                    "evidence_status": "verified",
                    "evidence_message": None,
                }
            }
        pytest.fail("The general photo Tool must not be used for known candidates")

    def fake_candidate_photo_search(
        request: Any,
        session: Any,
        *,
        osm_place_id: UUID,
    ) -> list[ImageSearchResult]:
        requested_place_ids.append(osm_place_id)
        if osm_place_id == first_candidate.id:
            return [make_indexed_photo(second_candidate.source_url, 1001)]
        return [
            make_indexed_photo(second_candidate.source_url, 2001),
            make_indexed_photo(second_candidate.source_url, 2001),
        ]

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )
    monkeypatch.setattr(
        "app.application.restaurant_discovery.search_restaurant_photos",
        fake_candidate_photo_search,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos de algunos.",
        object(),
    )

    assert requested_place_ids == [first_candidate.id, second_candidate.id]
    assert [candidate.name for candidate in response.restaurants] == [
        "Casa Verde",
        "Casa Verde",
    ]
    assert [candidate.source_url for candidate in response.restaurants] == [
        first_candidate.source_url,
        second_candidate.source_url,
    ]
    assert "id" not in response.model_dump(mode="json")["restaurants"][0]
    assert len(response.photos) == 1
    assert response.photos[0].restaurant_source_url == second_candidate.source_url
    assert response.photos[0].restaurant_source_url != first_candidate.source_url
    photo_result = find_tool_result(
        client.requests[-1],
        "search_restaurant_photos",
    )
    assert len(photo_result["output"]) == 1
    assert (
        photo_result["output"][0]["restaurant_source_url"]
        == second_candidate.source_url
    )


def test_candidate_search_replaces_photos_from_an_earlier_general_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = make_candidate("Casa Verde", 252)
    unrelated_photo = {
        "image_url": "/images/files/unrelated.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:unrelated.jpg",
        "restaurant_name": "Casa Verde",
        "restaurant_source_url": "https://www.openstreetmap.org/node/999",
    }
    configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurant_photos",
            {"query": "photos of vegetarian restaurants"},
        ),
        function_call_response(
            "search_restaurants",
            {"query": "vegetarian restaurants in Madrid"},
        ),
        text_response("Encontré un candidato."),
    )
    scoped_search_ids: list[UUID] = []

    def fake_dispatch(name: str | None, args: Any, session: Any):
        if name == "search_restaurant_photos":
            return {"output": [unrelated_photo]}
        return {
            "output": {
                "results": [candidate.model_dump(mode="json")],
                "evidence_status": "verified",
                "evidence_message": None,
            }
        }

    def fake_candidate_photo_search(
        request: Any,
        session: Any,
        *,
        osm_place_id: UUID,
    ) -> list[ImageSearchResult]:
        scoped_search_ids.append(osm_place_id)
        return [make_indexed_photo(candidate.source_url, 2521)]

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )
    monkeypatch.setattr(
        "app.application.restaurant_discovery.search_restaurant_photos",
        fake_candidate_photo_search,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert scoped_search_ids == [candidate.id]
    assert len(response.photos) == 1
    assert response.photos[0].restaurant_source_url == candidate.source_url
    assert response.photos[0].image_url != unrelated_photo["image_url"]


def test_agent_forces_photo_search_when_model_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = make_candidate("Restaurante Verde", 303)
    configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurants",
            {"query": "vegetarian restaurants in Madrid"},
        ),
        text_response("He encontrado candidatos vegetarianos."),
    )
    dispatched: list[str | None] = []
    scoped_search_ids: list[UUID] = []

    def fake_dispatch(name: str | None, args: Any, session: Any):
        dispatched.append(name)
        if name == "search_restaurants":
            return {
                "output": {
                    "results": [candidate.model_dump(mode="json")],
                    "evidence_status": "verified",
                    "evidence_message": None,
                }
            }
        pytest.fail("The Agent must use candidate-scoped photo retrieval")

    def fake_candidate_photo_search(
        request: Any,
        session: Any,
        *,
        osm_place_id: UUID,
    ) -> list[ImageSearchResult]:
        scoped_search_ids.append(osm_place_id)
        return [make_indexed_photo(candidate.source_url, 3001)]

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )
    monkeypatch.setattr(
        "app.application.restaurant_discovery.search_restaurant_photos",
        fake_candidate_photo_search,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert dispatched == ["search_restaurants"]
    assert scoped_search_ids == [candidate.id]
    assert [photo.restaurant_source_url for photo in response.photos] == [
        candidate.source_url
    ]
    assert "No he encontrado fotos" not in response.answer


def test_agent_explains_when_no_candidate_photos_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = make_candidate("Restaurante Verde", 404)
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
            return {
                "output": {
                    "results": [candidate.model_dump(mode="json")],
                    "evidence_status": "verified",
                    "evidence_message": None,
                }
            }
        pytest.fail("The Agent must use candidate-scoped photo retrieval")

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )
    monkeypatch.setattr(
        "app.application.restaurant_discovery.search_restaurant_photos",
        lambda request, session, *, osm_place_id: [],
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert response.photos == []
    assert "No he encontrado fotos indexadas asociadas a los restaurantes candidatos." in (
        response.answer
    )


def test_agent_distinguishes_candidate_photo_search_error_from_empty_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = make_candidate("Restaurante Verde", 606)
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
        text_response("No se pudieron verificar las fotos."),
    )

    def fake_dispatch(name: str | None, args: Any, session: Any):
        if name == "search_restaurants":
            return {
                "output": {
                    "results": [candidate.model_dump(mode="json")],
                    "evidence_status": "verified",
                    "evidence_message": None,
                }
            }
        pytest.fail("Candidate-scoped photos must bypass the general photo Tool")

    def fail_photo_search(request: Any, session: Any, *, osm_place_id: UUID):
        raise RuntimeError("photo index unavailable")

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )
    monkeypatch.setattr(
        "app.application.restaurant_discovery.search_restaurant_photos",
        fail_photo_search,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Busca restaurantes vegetarianos en Madrid y enséñame fotos.",
        object(),
    )

    assert response.photos == []
    assert "No he podido completar la búsqueda de fotos." in response.answer
    photo_result = find_tool_result(
        client.requests[-1],
        "search_restaurant_photos",
    )
    assert photo_result["error"]["code"] == "tool_execution_failed"


def test_general_photo_search_still_deduplicates_and_keeps_its_tool_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo = {
        "image_url": "/images/files/photo.jpg",
        "source_url": "https://commons.wikimedia.org/wiki/File:photo.jpg",
        "restaurant_name": "Casa Verde",
        "restaurant_source_url": "https://www.openstreetmap.org/node/707",
    }
    configure_client(
        monkeypatch,
        function_call_response(
            "search_restaurant_photos",
            {"query": "photos of restaurants", "top_k": 5},
        ),
        text_response("Encontré una foto."),
    )
    dispatched: list[str | None] = []

    def fake_dispatch(name: str | None, args: Any, session: Any):
        dispatched.append(name)
        return {"output": [photo, photo]}

    monkeypatch.setattr(
        restaurant_search_agent,
        "dispatch_tool_call",
        fake_dispatch,
    )

    response = restaurant_search_agent.run_restaurant_search_agent(
        "Enséñame fotos de restaurantes.",
        object(),
    )

    assert dispatched == ["search_restaurant_photos"]
    assert len(response.photos) == 1


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
    candidate = make_candidate("Example", 505)

    def fake_dispatch(name: str | None, args: Any, received_session: Any):
        dispatched.append((name, args, received_session))
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
    first_request, final_request = client.requests
    assert first_request["model"] == "restaurant-search-agent"
    assert first_request["tool_choice"] == "auto"
    assert any(
        tool["function"]["name"] == "search_restaurants"
        for tool in first_request["tools"]
    )
    assert json.loads(
        find_tool_call(final_request, "search_restaurants")["function"]["arguments"]
    ) == {"query": "vegetarian food", "top_k": 3}
    assert find_tool_result(final_request, "search_restaurants") == {
        "output": {
            "results": [candidate.model_dump(mode="json")],
            "evidence_status": "not_required",
            "evidence_message": None,
        }
    }
    assert final_request["messages"][-1]["tool_call_id"] == "call-1"
    assert all(
        repr(session) not in repr(request)
        for request in client.requests
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
    final_request = client.requests[-1]
    assert "tools" not in final_request
    assert "tool_choice" not in final_request
    rejected_result = find_tool_result(final_request, "search_documents")
    assert rejected_result["error"]["code"] == "tool_call_limit_reached"


def test_agent_graph_stops_when_model_turn_limit_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = configure_client(
        monkeypatch,
        *[
            function_call_response("search_documents", {"query": "menus"})
            for _ in range(restaurant_search_agent.MAX_MODEL_TURNS)
        ],
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

    with pytest.raises(
        restaurant_search_agent.RestaurantSearchAgentError,
        match="within the allowed model turns",
    ):
        restaurant_search_agent.run_restaurant_search_agent(
            "Busca documentos",
            object(),
        )

    assert len(client.requests) == restaurant_search_agent.MAX_MODEL_TURNS
    assert dispatch_count == restaurant_search_agent.MAX_TOOL_CALLS
    last_tool_result = find_tool_result(
        client.requests[-1],
        "search_documents",
    )
    assert last_tool_result["error"]["code"] == "tool_call_limit_reached"


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
    tool_result = find_tool_result(
        client.requests[-1],
        "search_documents",
    )
    assert tool_result["error"]["code"] == "tool_execution_failed"


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

    assert len(client.requests) == 1
