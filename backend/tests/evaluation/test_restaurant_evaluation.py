import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.evaluation import evaluate_restaurant_search
from scripts.evaluation.evaluate_restaurant_search import (
    RestaurantEvaluationCase,
    _matches_expected,
    load_cases,
)


def test_madrid_restaurant_evaluation_cases_are_valid() -> None:
    cases = load_cases()

    assert len(cases) == 18
    assert any(case.get("review_only") for case in cases)
    assert [
        case for case in cases if case.get("expected_empty")
    ] == [
        {
            "query": "cocina kosher",
            "city": "Madrid",
            "category": "evidence",
            "expected_empty": True,
            "expected_evidence_status": "no_evidence",
            "label_source": "madrid_holdout_judgments.json",
        }
    ]


def test_evaluation_reports_filter_compliance_and_expected_empty_rate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = [
        {
            "query": "mexican food",
            "city": "Madrid",
            "cuisine": "mexican",
            "category": "cuisine",
            "expected_cuisine": "mexican",
        },
        {
            "query": "kosher",
            "city": "Madrid",
            "category": "evidence",
            "expected_empty": True,
            "expected_evidence_status": "no_evidence",
        },
    ]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

    monkeypatch.setattr(
        evaluate_restaurant_search,
        "load_cases",
        lambda _: cases,
    )
    monkeypatch.setattr(
        evaluate_restaurant_search,
        "SessionLocal",
        FakeSession,
    )

    def fake_search(query, session, **kwargs):
        if query == "kosher":
            return SimpleNamespace(results=[], evidence_status="no_evidence")
        return SimpleNamespace(
            results=[
                SimpleNamespace(
                    id="restaurant-id",
                    name="Wrong city and cuisine",
                    city="Barcelona",
                    cuisine="italian",
                    features=[],
                    similarity=0.9,
                )
            ],
            evidence_status="not_required",
        )

    monkeypatch.setattr(
        evaluate_restaurant_search,
        "search_osm_places_by_text",
        fake_search,
    )

    evaluate_restaurant_search.run_evaluation(top_k=3)
    output = capsys.readouterr().out

    assert "Expected empty | PASS | evidence=no_evidence" in output
    assert "Unexpected empty-query rate: 0/1 (0%)" in output
    assert "Constraint compliance: 0/1 queries (0%)" in output
    assert "0/1 returned results (0%)" in output


def test_holdout_queries_are_valid_and_all_for_manual_review() -> None:
    cases = load_cases(
        Path(__file__).parents[2]
        / "data"
        / "restaurant_evaluation"
        / "madrid_holdout_queries.json"
    )

    assert len(cases) == 20
    assert all(case.get("review_only") for case in cases)


def test_holdout_judgments_cover_each_manual_query() -> None:
    data_dir = Path(__file__).parents[2] / "data" / "restaurant_evaluation"
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
