import json
from pathlib import Path


def test_evaluation_cases_reference_existing_documents() -> None:
    evaluation_dir = Path(__file__).parents[1] / "data" / "evaluation"
    cases = json.loads((evaluation_dir / "cases.json").read_text(encoding="utf-8"))
    document_names = {
        path.name for path in (evaluation_dir / "documents").glob("*.txt")
    }

    answerable_cases = [
        case
        for case in cases
        if case["expected_source"] is not None
        or case.get("expected_sources")
    ]
    abstention_cases = [case for case in cases if case["should_abstain"]]

    assert len(answerable_cases) == 11
    assert abstention_cases
    assert all(
        source in document_names
        for case in answerable_cases
        for source in (
            case.get("expected_sources")
            or [case["expected_source"]]
        )
    )
    assert all(
        case["expected_source"] is None
        and not case.get("expected_sources")
        for case in abstention_cases
    )
    assert {
        case["case_type"] for case in abstention_cases
    } == {"ambiguous", "unsupported"}
    multi_source_cases = [
        case for case in cases if case.get("expected_sources")
    ]
    assert len(multi_source_cases) == 1
    assert len(multi_source_cases[0]["expected_sources"]) == 2
