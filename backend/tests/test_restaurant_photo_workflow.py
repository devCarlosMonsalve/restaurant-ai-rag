from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.schemas import OsmRestaurantSearchResult
from app.workflows.restaurant_photo_search import (
    RestaurantPhotoWorkflowError,
    run_restaurant_photo_workflow,
)
from app.workflows.schemas import (
    RestaurantPhotoWorkflowRequest,
    RestaurantPhotoWorkflowResponse,
)


def make_candidate(
    name: str = "Casa Verde",
    source_url: str | None = None,
) -> dict[str, Any]:
    return OsmRestaurantSearchResult(
        id=uuid4(),
        name=name,
        city="Madrid",
        cuisine="vegetarian",
        location="Calle Mayor 1",
        latitude=40.4168,
        longitude=-3.7038,
        features=[],
        source_url=source_url or f"https://www.openstreetmap.org/node/{uuid4().int}",
        attribution="© OpenStreetMap contributors",
        attribution_url="https://www.openstreetmap.org/copyright",
        similarity=0.82,
    ).model_dump(mode="json")


def make_photo(
    restaurant_source_url: str | None,
    *,
    source_url: str | None = None,
    image_url: str = "/images/files/photo.jpg",
) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "source_name": "photo.jpg",
        "similarity": 0.91,
        "metadata_match_count": 1,
        "source_url": source_url,
        "license_name": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "Photographer",
        "restaurant_name": "Casa Verde",
        "restaurant_location": "Calle Mayor 1",
        "restaurant_cuisine": "vegetarian",
        "restaurant_features": [],
        "restaurant_source_url": restaurant_source_url,
        "restaurant_attribution": "© OpenStreetMap contributors",
        "restaurant_attribution_url": "https://www.openstreetmap.org/copyright",
        "image_url": image_url,
    }


def install_tool_stub(monkeypatch, candidates, photo_results=None, error=None):
    calls: list[tuple[str, dict[str, Any], Session]] = []

    def fake_dispatch(
        name: str,
        arguments: dict[str, Any],
        session: Session,
    ) -> dict[str, Any]:
        calls.append((name, arguments, session))
        if error and error[0] == name:
            return {"error": {"code": "tool_execution_failed", "message": "failed"}}
        if name == "search_restaurants":
            return {
                "output": {
                    "results": candidates,
                    "evidence_status": "verified",
                    "evidence_message": None,
                }
            }
        return {"output": photo_results or []}

    monkeypatch.setattr(
        "app.workflows.restaurant_photo_search.dispatch_tool_call",
        fake_dispatch,
    )
    return calls


def run_workflow(session: Session, **request_values):
    request = RestaurantPhotoWorkflowRequest(
        query="vegetarian restaurants in Madrid",
        city="Madrid",
        cuisine="vegetarian",
        **request_values,
    )
    return run_restaurant_photo_workflow(request, session)


def test_candidates_and_associated_photos_are_returned(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    photo = make_photo(candidate["source_url"])
    calls = install_tool_stub(monkeypatch, [candidate], [photo])

    response = run_workflow(db_session)

    assert response.photo_search_status == "completed"
    assert len(response.restaurants_with_photos) == 1
    assert response.restaurants_with_photos[0].restaurant.source_url == candidate["source_url"]
    assert len(response.restaurants_with_photos[0].photos) == 1
    assert response.candidates_without_returned_photos == []
    assert [call[0] for call in calls] == [
        "search_restaurants",
        "search_restaurant_photos",
    ]
    assert all(call[2] is db_session for call in calls)


def test_candidate_without_returned_photo_is_not_claimed_to_have_no_photos(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    install_tool_stub(monkeypatch, [candidate], [])

    response = run_workflow(db_session)

    assert response.restaurants_with_photos == []
    assert [item.source_url for item in response.candidates_without_returned_photos] == [
        candidate["source_url"]
    ]
    assert any("does not establish" in warning for warning in response.warnings)


def test_no_candidates_skips_photo_search(
    db_session: Session,
    monkeypatch,
) -> None:
    calls = install_tool_stub(monkeypatch, [], [])

    response = run_workflow(db_session)

    assert response.photo_search_status == "not_started"
    assert response.restaurants_with_photos == []
    assert response.candidates_without_returned_photos == []
    assert [call[0] for call in calls] == ["search_restaurants"]


def test_photo_without_restaurant_source_url_is_not_associated(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    install_tool_stub(monkeypatch, [candidate], [make_photo(None)])

    response = run_workflow(db_session)

    assert response.restaurants_with_photos == []
    assert response.candidates_without_returned_photos[0].source_url == candidate["source_url"]
    assert any("lacked restaurant_source_url" in warning for warning in response.warnings)


def test_photo_is_associated_by_exact_source_url(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    install_tool_stub(
        monkeypatch,
        [candidate],
        [make_photo(candidate["source_url"])],
    )

    response = run_workflow(db_session)

    assert response.restaurants_with_photos[0].restaurant.source_url == (
        response.restaurants_with_photos[0].photos[0].restaurant_source_url
    )


def test_photo_for_different_candidate_is_rejected(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    other_candidate = make_candidate("Other Place")
    install_tool_stub(
        monkeypatch,
        [candidate],
        [make_photo(other_candidate["source_url"])],
    )

    response = run_workflow(db_session)

    assert response.restaurants_with_photos == []
    assert response.candidates_without_returned_photos[0].source_url == candidate["source_url"]
    assert any("did not match the candidate" in warning for warning in response.warnings)


def test_multiple_photos_for_one_restaurant_are_returned(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    photos = [
        make_photo(candidate["source_url"], source_url="https://commons.test/one"),
        make_photo(candidate["source_url"], source_url="https://commons.test/two"),
    ]
    install_tool_stub(monkeypatch, [candidate], photos)

    response = run_workflow(db_session, photos_per_candidate=2)

    assert len(response.restaurants_with_photos[0].photos) == 2


def test_duplicate_photos_are_removed(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    duplicate = make_photo(
        candidate["source_url"],
        source_url="https://commons.test/same-photo",
    )
    install_tool_stub(monkeypatch, [candidate], [duplicate, duplicate])

    response = run_workflow(db_session, photos_per_candidate=2)

    assert len(response.restaurants_with_photos[0].photos) == 1


def test_candidate_limit_caps_processing(
    db_session: Session,
    monkeypatch,
) -> None:
    candidates = [make_candidate(f"Restaurant {index}") for index in range(3)]
    calls = install_tool_stub(monkeypatch, candidates, [])

    response = run_workflow(db_session, candidate_limit=2)

    photo_calls = [call for call in calls if call[0] == "search_restaurant_photos"]
    assert len(photo_calls) == 2
    assert len(response.candidates_without_returned_photos) == 2
    search_call = calls[0]
    assert search_call[1]["top_k"] == 2


def test_photo_limit_is_applied_per_candidate(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    photos = [
        make_photo(candidate["source_url"], source_url="https://commons.test/first"),
        make_photo(candidate["source_url"], source_url="https://commons.test/second"),
    ]
    calls = install_tool_stub(monkeypatch, [candidate], photos)

    response = run_workflow(db_session, photos_per_candidate=1)

    assert len(response.restaurants_with_photos[0].photos) == 1
    photo_call = calls[1]
    assert photo_call[1]["top_k"] == 1


@pytest.mark.parametrize(
    ("failing_tool", "candidate_results"),
    [
        ("search_restaurants", []),
        ("search_restaurant_photos", [make_candidate()]),
    ],
)
def test_tool_error_raises_instead_of_returning_empty_success(
    db_session: Session,
    monkeypatch,
    failing_tool: str,
    candidate_results: list[dict[str, Any]],
) -> None:
    install_tool_stub(
        monkeypatch,
        candidate_results,
        [],
        error=(failing_tool, "tool_execution_failed"),
    )

    with pytest.raises(RestaurantPhotoWorkflowError, match=failing_tool):
        run_workflow(db_session)


def test_public_response_never_exposes_image_path(
    db_session: Session,
    monkeypatch,
) -> None:
    candidate = make_candidate()
    install_tool_stub(
        monkeypatch,
        [candidate],
        [make_photo(candidate["source_url"])],
    )

    response = run_workflow(db_session)

    assert "image_path" not in response.model_dump_json()


def test_workflow_endpoint_injects_session_and_returns_result(
    client: TestClient,
    monkeypatch,
) -> None:
    calls: dict[str, Any] = {}
    expected = RestaurantPhotoWorkflowResponse(
        query="vegetarian restaurants in Madrid",
        evidence_status="verified",
        photo_search_status="completed",
        restaurants_with_photos=[],
        candidates_without_returned_photos=[],
    )

    def fake_workflow(request, session):
        calls["request"] = request
        calls["session"] = session
        return expected

    monkeypatch.setattr("app.main.run_restaurant_photo_workflow", fake_workflow)
    response = client.post(
        "/workflows/restaurant-photo-search",
        json={
            "query": "vegetarian restaurants in Madrid",
            "city": "Madrid",
            "cuisine": "vegetarian",
        },
    )

    assert response.status_code == 200
    assert response.json()["photo_search_status"] == "completed"
    assert calls["request"].candidate_limit == 5
    assert calls["request"].photos_per_candidate == 1
    assert isinstance(calls["session"], Session)
