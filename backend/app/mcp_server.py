import logging
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field, ValidationError

from app.database import SessionLocal, engine
from app.mcp_schemas import (
    McpDocumentResult,
    McpDocumentSearchResponse,
    McpPhotoResult,
    McpPhotoSearchResponse,
    McpRagAnswerResponse,
    McpRestaurantSearchResponse,
)
from app.observability import (
    configure_phoenix_tracing,
    shutdown_phoenix_tracing,
    traced_span,
)
from app.tools.registry import TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

Query = Annotated[str, Field(min_length=1, max_length=2000)]
TopK = Annotated[int, Field(ge=1, le=20)]
City = Annotated[str | None, Field(min_length=1, max_length=100)]
Cuisine = Annotated[str | None, Field(min_length=1, max_length=100)]

mcp = MCPServer("Restaurant AI RAG")


def _invoke_tool(name: str, arguments: dict[str, Any]) -> Any:
    with traced_span(
        "mcp.tool.execute",
        {"tool.name": name},
    ) as span:
        result = _invoke_tool_impl(name, arguments)
        span.set_attribute("tool.status", "success")
        return result


def _invoke_tool_impl(name: str, arguments: dict[str, Any]) -> Any:
    definition = TOOL_DEFINITIONS[name]
    allowed_parameters = (
        set(definition.input_model.model_fields) - definition.hidden_parameters
    )
    if set(arguments) - allowed_parameters:
        raise ToolError("invalid_arguments: tool arguments contain unsupported fields")

    try:
        validated = definition.input_model.model_validate(arguments)
    except ValidationError as error:
        raise ToolError("invalid_arguments: tool arguments are invalid") from error

    try:
        with SessionLocal() as session:
            return definition.handler(
                **validated.model_dump(exclude=definition.hidden_parameters),
                session=session,
            )
    except ValidationError as error:
        raise ToolError("invalid_arguments: tool arguments are invalid") from error
    except Exception as error:
        logger.exception("MCP tool execution failed: %s", name)
        raise ToolError(
            f"tool_execution_failed: {name} could not complete the request"
        ) from error


@mcp.tool()
def search_restaurants(
    query: Query,
    top_k: TopK = 12,
    city: City = None,
    cuisine: Cuisine = None,
) -> McpRestaurantSearchResponse:
    """Find restaurant candidates, including their OSM evidence status.

    Similarity is a ranking value, not a confidence score. Use the city and
    cuisine filters when the request requires exact restrictions.
    """
    response = _invoke_tool(
        "search_restaurants",
        {
            "query": query,
            "top_k": top_k,
            "city": city,
            "cuisine": cuisine,
        },
    )
    return McpRestaurantSearchResponse.from_search_response(response)


@mcp.tool()
def search_restaurant_photos(
    query: Query,
    top_k: TopK = 5,
    city: City = None,
    cuisine: Cuisine = None,
) -> McpPhotoSearchResponse:
    """Find indexed photos linked to OSM restaurants.

    Results retain Commons source, license, attribution, and restaurant source
    links. Similarity does not prove that an image is current or suitable.
    """
    results = _invoke_tool(
        "search_restaurant_photos",
        {
            "query": query,
            "top_k": top_k,
            "city": city,
            "cuisine": cuisine,
        },
    )
    return McpPhotoSearchResponse(
        results=[
            McpPhotoResult(
                source_name=result.source_name,
                similarity=result.similarity,
                source_url=result.source_url,
                license_name=result.license_name,
                license_url=result.license_url,
                attribution=result.attribution,
                restaurant_name=result.restaurant_name,
                restaurant_location=result.restaurant_location,
                restaurant_cuisine=result.restaurant_cuisine,
                restaurant_features=result.restaurant_features,
                restaurant_source_url=result.restaurant_source_url,
                restaurant_attribution=result.restaurant_attribution,
                restaurant_attribution_url=result.restaurant_attribution_url,
            )
            for result in results
        ]
    )


@mcp.tool()
def search_documents(
    query: Query,
    top_k: TopK = 5,
) -> McpDocumentSearchResponse:
    """Retrieve ranked excerpts from the indexed document corpus."""
    results = _invoke_tool(
        "search_documents",
        {
            "query": query,
            "top_k": top_k,
        },
    )
    return McpDocumentSearchResponse(
        results=[
            McpDocumentResult(
                source_name=result.source_name,
                chunk_index=result.chunk_index,
                content=result.content,
                similarity=result.similarity,
            )
            for result in results
        ]
    )


@mcp.tool()
def answer_from_documents(
    query: Query,
    top_k: TopK = 5,
) -> McpRagAnswerResponse:
    """Answer a question from indexed documents and return the cited sources."""
    response = _invoke_tool(
        "answer_from_documents",
        {
            "query": query,
            "top_k": top_k,
        },
    )
    return McpRagAnswerResponse.from_rag_response(response)


def main() -> None:
    tracer_provider = None
    try:
        tracer_provider = configure_phoenix_tracing()
        mcp.run(transport="stdio")
    finally:
        try:
            if tracer_provider is not None and not shutdown_phoenix_tracing():
                logger.warning("Phoenix spans were not fully exported at MCP shutdown")
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
