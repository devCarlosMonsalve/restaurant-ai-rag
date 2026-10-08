import json
from pathlib import Path
from typing import cast

from evaluate_restaurant_search import (
    RestaurantEvaluationCase,
    _matches_expected,
    load_cases,
)


def test_madrid_restaurant_evaluation_cases_are_valid() -> None:
    cases = load_cases()

    assert len(cases) == 17
    assert any(case.get("review_only") for case in cases)


def test_holdout_queries_are_valid_and_all_for_manual_review() -> None:
    cases = load_cases(
        Path(__file__).parent.parent
        / "data"
        / "restaurant_evaluation"
        / "madrid_holdout_queries.json"
    )

    assert len(cases) == 20
    assert all(case.get("review_only") for case in cases)


def test_holdout_judgments_cover_each_manual_query() -> None:
    data_dir = Path(__file__).parent.parent / "data" / "restaurant_evaluation"
    holdout_cases = load_cases(data_dir / "madrid_holdout_queries.json")
    judgments_data = json.loads(
        (data_dir / "madrid_holdout_judgments.json").read_text(encoding="utf-8")
    )
    judgments = judgments_data["judgments"]

    assert len(judgments) == len(holdout_cases) == 20
    assert {judgment["query"] for judgment in judgments} == {
        case["query"] for case in holdout_cases
    }
    assert len({judgment["query"] for judgment in judgments}) == len(judgments)
    assert {
        judgment["judgment"] for judgment in judgments
    } <= {"relevant", "partial", "no_evidence", "unverified"}


def test_evaluation_matches_a_feature_prefix() -> None:
    case = cast(
        RestaurantEvaluationCase,
        {
            "query": "mesas al aire libre",
            "city": "Madrid",
            "category": "feature",
            "expected_feature_prefix": "mesas al aire libre:",
        },
    )

    assert _matches_expected(
        "El Valle",
        None,
        ["Mesas al aire libre: disponible"],
        case,
    )


def test_evaluation_matches_exact_cuisine_tag() -> None:
    case = cast(
        RestaurantEvaluationCase,
        {
            "query": "tapas",
            "city": "Madrid",
            "category": "cuisine",
            "expected_cuisine": "tapas",
        },
    )

    assert _matches_expected("Vinos y Tapas", "spanish;tapas", [], case)
    assert not _matches_expected("Another place", "spanish;local", [], case)


def test_evaluation_matches_name_substring() -> None:
    case = cast(
        RestaurantEvaluationCase,
        {
            "query": "cocido",
            "city": "Madrid",
            "category": "dish",
            "expected_name_contains": "Cocido",
        },
    )

    assert _matches_expected("Cocido en barro", "spanish", [], case)


def test_evaluation_rejects_cases_with_multiple_expectations(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        '[{"query":"test","city":"Madrid","category":"feature",'
        '"expected_feature_prefix":"terraza","expected_cuisine":"spanish"}]',
        encoding="utf-8",
    )

    try:
        load_cases(cases_path)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("Expected malformed evaluation case to be rejected")
