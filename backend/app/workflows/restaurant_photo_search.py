import logging
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.parse import quote
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from sqlalchemy.orm import Session

from app.application.restaurant_discovery import (
    search_restaurant_photos as search_restaurant_photos_use_case,
    search_restaurants as search_restaurants_use_case,
)
from app.schemas import (
    ImageSearchRequest,
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
                runtime.context["session"],
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
    photos: list[dict[str, Any]] = []
    warnings = list(state["warnings"])
    seen_photo_keys: set[str] = set()

    for candidate in state["candidates"][: state["candidate_limit"]]:
        query_parts = [
            state["query"],
            candidate["name"],
            candidate.get("location"),
            candidate.get("city"),
        ]
        query = " ".join(part for part in query_parts if part)
        try:
            photo_results = search_restaurant_photos_use_case(
                ImageSearchRequest(
                    query=query,
                    top_k=state["photos_per_candidate"],
                    city=state["city"] or candidate.get("city"),
                    cuisine=state["cuisine"],
                    osm_places_only=True,
                ),
                runtime.context["session"],
                osm_place_id=UUID(candidate["id"]),
            )
            if not isinstance(photo_results, list):
                raise RestaurantPhotoWorkflowError(
                    "Photo search returned an invalid payload."
                )
            parsed_photos = [
                RestaurantPhoto.model_validate(_serialize_photo(photo))
                for photo in photo_results[: state["photos_per_candidate"]]
            ]
        except RestaurantPhotoWorkflowError:
            raise
        except Exception as error:
            logger.exception(
                "Candidate-scoped photo search failed for OSM place %s",
                candidate["id"],
            )
            raise RestaurantPhotoWorkflowError(
                "Tool 'search_restaurant_photos' failed for a restaurant candidate."
            ) from error

        for photo in parsed_photos:
            if not photo.restaurant_source_url:
                warning = (
                    "A photo result lacked restaurant_source_url and was not "
                    "associated with a candidate."
                )
                if warning not in warnings:
                    warnings.append(warning)
                continue
            if photo.restaurant_source_url != candidate["source_url"]:
                warning = (
                    "A photo result was excluded because its "
                    "restaurant_source_url did not match the candidate."
                )
                if warning not in warnings:
                    warnings.append(warning)
                continue

            photo_key = photo.source_url or str(photo.id)
            if photo_key in seen_photo_keys:
                continue
            seen_photo_keys.add(photo_key)
            photos.append(photo.model_dump(mode="json"))

    associated_candidate_urls = {
        photo["restaurant_source_url"] for photo in photos
    }
    if any(
        candidate["source_url"] not in associated_candidate_urls
        for candidate in state["candidates"]
    ):
        warning = (
            "A photo search returning no associated results does not establish "
            "that no photos exist."
        )
        if warning not in warnings:
            warnings.append(warning)

    return {
        "photos": photos,
        "photo_search_status": "completed",
        "warnings": warnings,
    }


def _serialize_photo(photo: Any) -> dict[str, Any]:
    if not isinstance(photo, ImageSearchResult):
        raise RestaurantPhotoWorkflowError(
            "Photo search returned an invalid result type."
        )
    filename = Path(photo.image_path.replace("\\", "/")).name
    if not filename:
        raise RestaurantPhotoWorkflowError(
            "Photo search returned an image without a usable filename."
        )
    result = photo.model_dump(mode="json", exclude={"image_path"})
    result["image_url"] = f"/images/files/{quote(filename, safe='')}"
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
    result = _restaurant_photo_workflow.invoke(
        state,
        context={"session": session},
    )
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
