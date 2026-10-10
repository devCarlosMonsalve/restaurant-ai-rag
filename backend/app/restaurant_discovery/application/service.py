import logging
from uuid import UUID

from app.restaurant_discovery.application.ports import RestaurantDiscoveryPort
from app.restaurant_discovery.domain.photo_association import (
    is_verified_photo_association,
)
from app.restaurant_discovery.application.contracts import (
    ImageSearchRequest,
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
    OsmRestaurantSearchResult,
)

logger = logging.getLogger(__name__)


class CandidatePhotoSearchError(RuntimeError):
    """A candidate-scoped photo search could not return a valid result."""


def search_restaurants(
    request: OsmRestaurantSearchRequest,
    repository: RestaurantDiscoveryPort,
    *,
    include_places_with_photos: bool = False,
) -> OsmRestaurantSearchResponse:
    return repository.search_restaurants(
        request.query,
        top_k=request.top_k,
        city=request.city,
        cuisine=request.cuisine,
        include_places_with_photos=include_places_with_photos,
    )


def search_restaurant_photos(
    request: ImageSearchRequest,
    repository: RestaurantDiscoveryPort,
    *,
    osm_place_id: UUID | None = None,
) -> list[ImageSearchResult]:
    return repository.search_photos(
        request.query,
        top_k=request.top_k,
        osm_places_only=request.osm_places_only,
        city=request.city,
        cuisine=request.cuisine,
        osm_place_id=osm_place_id,
    )


def search_verified_candidate_photos(
    candidates: list[OsmRestaurantSearchResult],
    query: str,
    repository: RestaurantDiscoveryPort,
    *,
    city: str | None = None,
    cuisine: str | None = None,
    candidate_limit: int,
    photos_per_candidate: int = 1,
) -> tuple[list[ImageSearchResult], list[str]]:
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be greater than zero")
    if photos_per_candidate <= 0:
        raise ValueError("photos_per_candidate must be greater than zero")

    photos: list[ImageSearchResult] = []
    warnings: list[str] = []
    seen_photo_keys: set[str] = set()
    candidates_without_photos = False

    for candidate in candidates[:candidate_limit]:
        candidate_query = " ".join(
            part
            for part in (query, candidate.name, candidate.location, candidate.city)
            if part
        )
        try:
            photo_results = search_restaurant_photos(
                ImageSearchRequest(
                    query=candidate_query,
                    top_k=photos_per_candidate,
                    city=city or candidate.city,
                    cuisine=cuisine,
                    osm_places_only=True,
                ),
                repository,
                osm_place_id=candidate.id,
            )
            if not isinstance(photo_results, list):
                raise CandidatePhotoSearchError(
                    "Photo search returned an invalid payload."
                )
            candidate_photos = photo_results[:photos_per_candidate]
            if not candidate_photos:
                candidates_without_photos = True
                continue
        except CandidatePhotoSearchError:
            raise
        except Exception as error:
            logger.exception(
                "Candidate-scoped photo search failed for OSM place %s",
                candidate.id,
            )
            raise CandidatePhotoSearchError(
                "Candidate-scoped photo search failed."
            ) from error

        associated_photos = 0
        for photo in candidate_photos:
            if not isinstance(photo, ImageSearchResult):
                raise CandidatePhotoSearchError(
                    "Photo search returned an invalid result type."
                )
            if not photo.restaurant_source_url:
                warning = (
                    "A photo result lacked restaurant_source_url and was not "
                    "associated with a candidate."
                )
                if warning not in warnings:
                    warnings.append(warning)
                continue
            if not is_verified_photo_association(
                candidate.source_url,
                photo.restaurant_source_url,
            ):
                warning = (
                    "A photo result was excluded because its "
                    "restaurant_source_url did not match the candidate."
                )
                if warning not in warnings:
                    warnings.append(warning)
                continue

            associated_photos += 1
            photo_key = photo.source_url or str(photo.id)
            if photo_key in seen_photo_keys:
                continue
            seen_photo_keys.add(photo_key)
            photos.append(photo)

        if not associated_photos:
            candidates_without_photos = True

    if candidates_without_photos:
        warnings.append(
            "A photo search returning no associated results does not establish "
            "that no photos exist."
        )

    return photos, warnings
