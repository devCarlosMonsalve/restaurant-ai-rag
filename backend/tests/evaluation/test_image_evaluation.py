from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from types import SimpleNamespace

import pytest

from scripts.evaluation import evaluate_image_search
from app.schemas import ImageSearchResult


MADRID_CASES_PATH = (
    evaluate_image_search.EVALUATION_DIR / "madrid_cases.json"
)


def test_image_evaluation_cases_reference_existing_images() -> None:
    cases = evaluate_image_search.load_cases()
    image_names = {
        path.name for path in evaluate_image_search.IMAGES_DIR.glob("*")
    }
    answerable_cases = [case for case in cases if not case["should_abstain"]]
    abstention_cases = [case for case in cases if case["should_abstain"]]

    assert len(answerable_cases) == 12
    assert len(abstention_cases) == 12
    assert {case["language"] for case in cases} == {"en", "es"}
    assert all(
        case["expected_image"] in image_names for case in answerable_cases
    )
    assert all(case["expected_image"] is None for case in abstention_cases)
    assert sum(case["language"] == "en" for case in answerable_cases) == 6
    assert sum(case["language"] == "es" for case in answerable_cases) == 6
    assert sum(case["language"] == "en" for case in abstention_cases) == 6
    assert sum(case["language"] == "es" for case in abstention_cases) == 6


def test_madrid_evaluation_cases_reference_imported_photos() -> None:
    cases = evaluate_image_search.load_cases(MADRID_CASES_PATH)
    image_names = {
        path.name
        for path in evaluate_image_search.IMAGES_DIR.rglob("*")
        if path.is_file()
    }

    assert len(cases) == 18
    assert {case["language"] for case in cases} == {"en", "es"}
    image_cases = [
        case for case in cases if case["expected_image"] is not None
    ]
    restaurant_cases = [
        case for case in cases if case.get("expected_restaurant") is not None
    ]
    abstention_cases = [
        case for case in cases if case["should_abstain"]
    ]
    assert len(abstention_cases) == 6
    assert all(not case["should_abstain"] for case in image_cases + restaurant_cases)
    assert all(case["expected_image"] in image_names for case in image_cases)
    assert {case["expected_restaurant"] for case in restaurant_cases} == {"Xamach"}
    assert len(image_cases) == 10
    assert len(restaurant_cases) == 2
    assert all(case.get("manual_review") for case in restaurant_cases)
    assert sum(case["language"] == "en" for case in abstention_cases) == 3
    assert sum(case["language"] == "es" for case in abstention_cases) == 3
    assert sum(case["language"] == "en" for case in cases) == 9
    assert sum(case["language"] == "es" for case in cases) == 9


def test_attribution_cases_use_unique_osm_ids_and_exact_urls() -> None:
    cases = evaluate_image_search.load_attribution_cases()

    assert len(cases) == 3
    assert len({(case["osm_type"], case["osm_id"]) for case in cases}) == 3
    assert all(
        case["expected_source_url"]
        == f"https://www.openstreetmap.org/{case['osm_type']}/{case['osm_id']}"
        for case in cases
    )


def test_candidate_attribution_evaluation_checks_id_url_and_empty_index(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = SimpleNamespace(
        id=uuid4(),
        osm_type="node",
        osm_id=13481913634,
        name="Los Mandilones",
        source_url="https://www.openstreetmap.org/node/13481913634",
    )
    empty_candidate = SimpleNamespace(
        id=uuid4(),
        osm_type="node",
        osm_id=4596525105,
        name="Los Nachos de Oro",
        source_url="https://www.openstreetmap.org/node/4596525105",
    )
    photo_ids = [uuid4(), uuid4()]
    cases = [
        {
            "query": "Los Mandilones Madrid",
            "osm_type": "node",
            "osm_id": 13481913634,
            "expected_name": "Los Mandilones",
            "expected_source_url": target.source_url,
        },
        {
            "query": "Los Nachos de Oro Madrid",
            "osm_type": "node",
            "osm_id": 4596525105,
            "expected_name": "Los Nachos de Oro",
            "expected_source_url": empty_candidate.source_url,
        },
    ]

    class FakeSession:
        scalar_call = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def scalar(self, statement):
            self.scalar_call += 1
            return {
                1: target,
                2: 2,
                3: empty_candidate,
                4: 0,
            }[self.scalar_call]

        def execute(self, statement):
            return [
                (photo_ids[0], target.id, target.id, target.source_url),
                (photo_ids[1], target.id, target.id, target.source_url),
            ]

    monkeypatch.setattr(
        evaluate_image_search,
        "load_attribution_cases",
        lambda _: cases,
    )
    monkeypatch.setattr(evaluate_image_search, "SessionLocal", FakeSession)
    monkeypatch.setattr(
        evaluate_image_search,
        "search_images_by_text",
        lambda query, session, **kwargs: (
            [
                ImageSearchResult(
                    id=photo_id,
                    source_name=f"{photo_id}.jpg",
                    image_path=f"C:/{photo_id}.jpg",
                    similarity=0.9,
                    restaurant_source_url=target.source_url,
                )
                for photo_id in photo_ids
            ]
            if kwargs["osm_place_id"] == target.id
            else []
        ),
    )

    evaluate_image_search.run_attribution_evaluation(top_k=2)
    output = capsys.readouterr().out

    assert "Candidate photo ID+URL integrity: 2/2 (100%)" in output
    assert "Empty results when index has no candidate photos: 1/1 (100%)" in output
    assert "Photo-presence agreement with index: 2/2 (100%)" in output


def test_image_evaluation_reports_hit_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {
            "query": "pasta with cherry tomatoes and basil",
            "expected_image": "pasta-primavera.png",
            "should_abstain": False,
            "language": "en",
        },
        {
            "query": "pasta con tomates cherry y albahaca",
            "expected_image": "pasta-primavera.png",
            "should_abstain": False,
            "language": "es",
        },
        {
            "query": "a restaurant named La Bola",
            "expected_image": None,
            "expected_restaurant": "La Bola",
            "should_abstain": False,
            "language": "es",
        },
        {
            "query": "a sailboat on the ocean",
            "expected_image": None,
            "should_abstain": True,
            "language": "en",
        },
    ]
    (tmp_path / "pasta-primavera.png").touch()
    monkeypatch.setattr(
        evaluate_image_search,
        "load_cases",
        lambda cases_path=evaluate_image_search.CASES_PATH: cases,
    )
    monkeypatch.setattr(evaluate_image_search, "IMAGES_DIR", tmp_path)

    class FakeScalarResult:
        def all(self) -> list[str]:
            return ["pasta-primavera.png"]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def scalars(self, statement) -> FakeScalarResult:
            return FakeScalarResult()

    monkeypatch.setattr(
        evaluate_image_search,
        "SessionLocal",
        lambda: FakeSession(),
    )
    monkeypatch.setattr(
        evaluate_image_search,
        "search_images_by_text",
        lambda query,
        session,
        *,
        top_k,
        osm_places_only,
        sample_images_only: [
            SimpleNamespace(
                source_name="pasta-primavera.png",
                restaurant_name="La Bola",
                similarity=0.19,
            )
        ],
    )

    evaluate_image_search.run_evaluation(top_k=3)
    output = capsys.readouterr().out

    assert "Overall Hit@1: 3/3 (100%)" in output
    assert "Overall Hit@3: 3/3 (100%)" in output
    assert "English Hit@1: 1/1 (100%)" in output
    assert "Spanish Hit@1: 2/2 (100%)" in output
    assert "ABSTENTION REVIEW | top_similarity=0.19" in output
    assert "Negative top-similarity range: 0.190-0.190" in output


def test_image_evaluation_fails_if_expected_images_are_not_indexed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cases = [
        {
            "query": "pasta with cherry tomatoes and basil",
            "expected_image": "pasta-primavera.png",
            "should_abstain": False,
            "language": "en",
        }
    ]
    (tmp_path / "pasta-primavera.png").touch()
    monkeypatch.setattr(
        evaluate_image_search,
        "load_cases",
        lambda cases_path=evaluate_image_search.CASES_PATH: cases,
    )
    monkeypatch.setattr(evaluate_image_search, "IMAGES_DIR", tmp_path)

    class FakeScalarResult:
        def all(self) -> list[str]:
            return []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def scalars(self, statement) -> FakeScalarResult:
            return FakeScalarResult()

    monkeypatch.setattr(
        evaluate_image_search,
        "SessionLocal",
        lambda: FakeSession(),
    )

    with pytest.raises(ValueError, match="not indexed"):
        evaluate_image_search.run_evaluation(top_k=3)
