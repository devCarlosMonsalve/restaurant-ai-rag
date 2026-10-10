import logging
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from sqlalchemy.orm import Session

from app.restaurant_discovery.application.service import (
    search_restaurants as search_restaurants_use_case,
    search_verified_candidate_photos,
)
from app.image_presentation import image_file_url
from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)
from app.observability import (
    disable_automatic_langchain_tracing,
    traced_span,
)
from app.restaurant_discovery.application.contracts import (
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
)
from app.workflows.schemas import (
    RestaurantPhoto,
    RestaurantPhotoWorkflowRequest,
    RestaurantPhotoWorkflowResponse,
    RestaurantWithPhotos,
)

logger = logging.getLogger(__name__)


class RestaurantPhotoWorkflowError(RuntimeError):
    """Raised when a workflow search dependency cannot provide a valid result."""


class WorkflowContext(TypedDict):
    session: Session


class WorkflowState(TypedDict):
    query: str
    city: str | None
    cuisine: str | None
    candidate_limit: int
    photos_per_candidate: int
    candidates: list[dict[str, Any]]
    evidence_status: str | None
    photos: list[dict[str, Any]]
    photo_search_status: Literal["not_started", "completed"]
    warnings: list[str]
    restaurants_with_photos: list[dict[str, Any]]
    candidates_without_returned_photos: list[dict[str, Any]]


def search_candidates(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    try:
        result = OsmRestaurantSearchResponse.model_validate(
            search_restaurants_use_case(
                OsmRestaurantSearchRequest(
                    query=state["query"],
                    top_k=state["candidate_limit"],
                    city=state["city"],
                    cuisine=state["cuisine"],
                ),
                PostgresRestaurantDiscoveryAdapter(runtime.context["session"]),
                include_places_with_photos=True,
            )
        )
    except Exception as error:
        logger.exception("Candidate search for the photo workflow failed")
        raise RestaurantPhotoWorkflowError(
            "Restaurant candidate search failed."
        ) from error

    return {
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in result.results[: state["candidate_limit"]]
        ],
        "evidence_status": result.evidence_status,
    }


def route_after_candidate_search(state: WorkflowState) -> str:
    if state["candidates"]:
        return "search_photos_for_candidates"
    return "compose_response"


def search_photos_for_candidates(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    try:
        candidates = [
            OsmRestaurantSearchResult.model_validate(candidate)
            for candidate in state["candidates"]
        ]
        photos, photo_warnings = search_verified_candidate_photos(
            candidates,
            state["query"],
            PostgresRestaurantDiscoveryAdapter(runtime.context["session"]),
            city=state["city"],
            cuisine=state["cuisine"],
            candidate_limit=state["candidate_limit"],
            photos_per_candidate=state["photos_per_candidate"],
        )
        serialized_photos = [
            RestaurantPhoto.model_validate(_serialize_photo(photo)).model_dump(
                mode="json"
            )
            for photo in photos
        ]
    except Exception as error:
        logger.exception("Candidate-scoped photo search failed in the workflow")
        raise RestaurantPhotoWorkflowError(
            "Tool 'search_restaurant_photos' failed for a restaurant candidate."
        ) from error

    return {
        "photos": serialized_photos,
        "photo_search_status": "completed",
        "warnings": list(dict.fromkeys([*state["warnings"], *photo_warnings])),
    }


def _serialize_photo(photo: Any) -> dict[str, Any]:
    if not isinstance(photo, ImageSearchResult):
        raise RestaurantPhotoWorkflowError(
            "Photo search returned an invalid result type."
        )
    result = photo.model_dump(mode="json", exclude={"image_path"})
    result["image_url"] = image_file_url(photo.image_path)
    return result


def compose_response(state: WorkflowState) -> dict[str, Any]:
    candidates_by_url = {
        candidate["source_url"]: candidate for candidate in state["candidates"]
    }
    photos_by_restaurant: dict[str, list[RestaurantPhoto]] = {}
    for photo_data in state["photos"]:
        photo = RestaurantPhoto.model_validate(photo_data)
        restaurant_url = photo.restaurant_source_url
        if restaurant_url in candidates_by_url:
            photos_by_restaurant.setdefault(restaurant_url, []).append(photo)

    restaurants_with_photos = []
    candidates_without_returned_photos = []
    for candidate_data in state["candidates"]:
        candidate = OsmRestaurantSearchResult.model_validate(candidate_data)
        candidate_photos = photos_by_restaurant.get(candidate.source_url, [])
        if candidate_photos:
            restaurants_with_photos.append(
                RestaurantWithPhotos(
                    restaurant=candidate,
                    photos=candidate_photos,
                ).model_dump(mode="json")
            )
        else:
            candidates_without_returned_photos.append(candidate.model_dump(mode="json"))

    return {
        "restaurants_with_photos": restaurants_with_photos,
        "candidates_without_returned_photos": candidates_without_returned_photos,
    }


def _build_workflow():
    builder = StateGraph(WorkflowState, context_schema=WorkflowContext)
    builder.add_node("search_candidates", search_candidates)
    builder.add_node(
        "search_photos_for_candidates",
        search_photos_for_candidates,
    )
    builder.add_node("compose_response", compose_response)
    builder.add_edge(START, "search_candidates")
    builder.add_conditional_edges(
        "search_candidates",
        route_after_candidate_search,
        {
            "search_photos_for_candidates": "search_photos_for_candidates",
            "compose_response": "compose_response",
        },
    )
    builder.add_edge("search_photos_for_candidates", "compose_response")
    builder.add_edge("compose_response", END)
    return builder.compile()


_restaurant_photo_workflow = _build_workflow()


def run_restaurant_photo_workflow(
    request: RestaurantPhotoWorkflowRequest,
    session: Session,
) -> RestaurantPhotoWorkflowResponse:
    state: WorkflowState = {
        **request.model_dump(),
        "candidates": [],
        "evidence_status": None,
        "photos": [],
        "photo_search_status": "not_started",
        "warnings": [],
        "restaurants_with_photos": [],
        "candidates_without_returned_photos": [],
    }
    with traced_span(
        "workflow.restaurant_photo_search",
        {
            "workflow.candidate_limit": request.candidate_limit,
            "workflow.photos_per_candidate": request.photos_per_candidate,
        },
    ) as span:
        with disable_automatic_langchain_tracing():
            result = _restaurant_photo_workflow.invoke(
                state,
                context={"session": session},
            )
        span.set_attribute(
            "workflow.candidate_count",
            len(result["candidates"]),
        )
        span.set_attribute("workflow.photo_count", len(result["photos"]))
    return RestaurantPhotoWorkflowResponse(
        query=request.query,
        evidence_status=result["evidence_status"],
        photo_search_status=result["photo_search_status"],
        restaurants_with_photos=result["restaurants_with_photos"],
        candidates_without_returned_photos=(
            result["candidates_without_returned_photos"]
        ),
        warnings=result["warnings"],
    )
