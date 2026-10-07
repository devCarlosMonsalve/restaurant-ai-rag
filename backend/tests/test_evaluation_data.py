import json
from pathlib import Path


def test_evaluation_cases_reference_existing_documents() -> None:
    evaluation_dir = Path(__file__).parents[1] / "data" / "evaluation"
    cases = json.loads((evaluation_dir / "cases.json").read_text(encoding="utf-8"))
    document_names = {
        path.name for path in (evaluation_dir / "documents").glob("*.txt")
    }

    answerable_cases = [case for case in cases if case["expected_source"] is not None]
    abstention_cases = [case for case in cases if case["should_abstain"]]

    assert len(answerable_cases) == 10
    assert abstention_cases
    assert all(case["expected_source"] in document_names for case in answerable_cases)
    assert all(case["expected_source"] is None for case in abstention_cases)
