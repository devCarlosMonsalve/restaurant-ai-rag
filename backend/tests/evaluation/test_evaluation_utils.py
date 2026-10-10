from pathlib import Path

from evaluation.utils import (
    build_run_metadata,
    fingerprint_rows,
    write_evaluation_report,
)


def test_corpus_fingerprint_is_stable_and_changes_with_rows() -> None:
    first = fingerprint_rows([("id-1", "name"), ("id-2", "other")])
    second = fingerprint_rows([("id-1", "name"), ("id-2", "other")])
    changed = fingerprint_rows([("id-1", "changed"), ("id-2", "other")])

    assert first == second
    assert first[1] == 2
    assert first[0] != changed[0]


def test_write_evaluation_report_creates_parent_directory(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "nested" / "report.json"
    write_evaluation_report(report_path, {"metrics": {"hit_at_1": 1.0}})

    assert '"hit_at_1": 1.0' in report_path.read_text(encoding="utf-8")


def test_run_metadata_records_dataset_corpus_and_git_version(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text('[{"query":"test"}]', encoding="utf-8")

    metadata = build_run_metadata(
        evaluator="test",
        cases_path=cases_path,
        configuration={"top_k": 3},
        corpus_name="test corpus",
        corpus_fingerprint="corpus-sha",
        corpus_row_count=2,
    )

    assert metadata["git_commit"]
    assert metadata["dataset"]["sha256"]
    assert metadata["corpus"]["sha256"] == "corpus-sha"
    assert metadata["corpus"]["row_count"] == 2
    assert metadata["configuration"] == {"top_k": 3}
