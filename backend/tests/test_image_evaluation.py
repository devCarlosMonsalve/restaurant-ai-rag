from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_image_search


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
            "query": "a sailboat on the ocean",
            "expected_image": None,
            "should_abstain": True,
            "language": "en",
        },
    ]
    (tmp_path / "pasta-primavera.png").touch()
    monkeypatch.setattr(evaluate_image_search, "load_cases", lambda: cases)
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
        lambda query, session, *, top_k: [
            SimpleNamespace(source_name="pasta-primavera.png", similarity=0.19)
        ],
    )

    evaluate_image_search.run_evaluation(top_k=3)
    output = capsys.readouterr().out

    assert "Overall Hit@1: 2/2 (100%)" in output
    assert "Overall Hit@3: 2/2 (100%)" in output
    assert "English Hit@1: 1/1 (100%)" in output
    assert "Spanish Hit@1: 1/1 (100%)" in output
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
    monkeypatch.setattr(evaluate_image_search, "load_cases", lambda: cases)
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
