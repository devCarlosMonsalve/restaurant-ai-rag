import logging
from dataclasses import dataclass
from typing import Any, Callable

from google.genai import types
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.image_presentation import image_file_url
from app.schemas import (
    DocumentSearchRequest,
    ImageSearchRequest,
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    RagQuestionRequest,
)
from app.tools.document_answer import answer_from_documents
from app.tools.document_search import search_documents
from app.tools.restaurant_photos import search_restaurant_photos
from app.tools.restaurant_search import search_restaurants

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., BaseModel | list[BaseModel]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    hidden_parameters: frozenset[str] = frozenset()


TOOL_DEFINITIONS = {
    "search_restaurants": ToolDefinition(
        name="search_restaurants",
        description=search_restaurants.__doc__ or "",
        input_model=OsmRestaurantSearchRequest,
        handler=search_restaurants,
    ),
    "search_restaurant_photos": ToolDefinition(
        name="search_restaurant_photos",
        description=search_restaurant_photos.__doc__ or "",
        input_model=ImageSearchRequest,
        handler=search_restaurant_photos,
        hidden_parameters=frozenset({"osm_places_only"}),
    ),
    "search_documents": ToolDefinition(
        name="search_documents",
        description=search_documents.__doc__ or "",
        input_model=DocumentSearchRequest,
        handler=search_documents,
    ),
    "answer_from_documents": ToolDefinition(
        name="answer_from_documents",
        description=answer_from_documents.__doc__ or "",
        input_model=RagQuestionRequest,
        handler=answer_from_documents,
    ),
}


def get_function_declarations() -> list[types.FunctionDeclaration]:
    declarations: list[types.FunctionDeclaration] = []
    for definition in TOOL_DEFINITIONS.values():
        schema = definition.input_model.model_json_schema(mode="validation")
        properties = schema.get("properties", {})
        for parameter in definition.hidden_parameters:
            properties.pop(parameter, None)
        schema["properties"] = properties
        schema["required"] = [
            parameter
            for parameter in schema.get("required", [])
            if parameter not in definition.hidden_parameters
        ]
        schema["additionalProperties"] = False

        declarations.append(
            types.FunctionDeclaration(
                name=definition.name,
                description=definition.description,
                parameters_json_schema=schema,
            )
        )
    return declarations


def dispatch_tool_call(
    name: str | None,
    arguments: Any,
    session: Session,
) -> dict[str, Any]:
    definition = TOOL_DEFINITIONS.get(name or "")
    if definition is None:
        return _tool_error("unknown_tool", "The requested tool is not available.")
    if not isinstance(arguments, dict):
        return _tool_error("invalid_arguments", "Tool arguments must be an object.")

    allowed_parameters = (
        set(definition.input_model.model_fields) - definition.hidden_parameters
    )
    if set(arguments) - allowed_parameters:
        return _tool_error(
            "invalid_arguments",
            "Tool arguments contain unsupported fields.",
        )

    try:
        validated = definition.input_model.model_validate(arguments)
    except ValidationError:
        return _tool_error(
            "invalid_arguments",
            "Tool arguments do not match the declared parameter schema.",
        )

    try:
        result = definition.handler(
            **validated.model_dump(exclude=definition.hidden_parameters),
            session=session,
        )
        return {"output": serialize_tool_result(result)}
    except Exception:
        logger.exception("Tool execution failed: %s", definition.name)
        return _tool_error(
            "tool_execution_failed",
            "The tool failed. Do not infer or invent missing results.",
        )


def serialize_tool_result(value: Any) -> Any:
    if isinstance(value, BaseModel):
        excluded = {"image_path"} if isinstance(value, ImageSearchResult) else None
        result = value.model_dump(mode="json", exclude=excluded)
        if isinstance(value, ImageSearchResult):
            result["image_url"] = image_file_url(value.image_path)
        return result
    if isinstance(value, list):
        return [serialize_tool_result(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_tool_result(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported Tool result type: {type(value).__name__}")


def _tool_error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}
