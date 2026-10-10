from pathlib import Path
from uuid import uuid4

import pytest

from scripts.evaluation import evaluate_rag
from app.schemas import DocumentSearchResult


def test_rag_evaluation_does_not_generate_answers_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    cases = [
        {
            "query": "What is in the menu?",
            "expected_source": "menu.txt",
            "should_abstain": False,
        },
        {
            "query": "What is the phone number?",
            "expected_source": None,
            "case_type": "unsupported",
            "should_abstain": True,
        },
    ]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def scalars(self, statement):
            return self

        def all(self):
            return ["menu.txt"]

    class FakeRetriever:
        def __init__(self, session) -> None:
            del session

        def search_documents(self, query: str, *, top_k: int):
            del top_k
            return (
                [
                    DocumentSearchResult(
                        document_id=uuid4(),
                        source_name="menu.txt",
                        chunk_index=0,
                        content="The soup uses tomato.",
                        similarity=0.9,
                    )
                ]
                if query == "What is in the menu?"
                else []
            )

    monkeypatch.setattr(evaluate_rag, "load_cases", lambda _: cases)
    monkeypatch.setattr(evaluate_rag, "SessionLocal", FakeSession)
    monkeypatch.setattr(
        evaluate_rag,
        "PostgresDocumentRetriever",
        FakeRetriever,
    )

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("Answer generation must be opt-in")

    monkeypatch.setattr(
        evaluate_rag,
        "generate_grounded_answer",
        unexpected_generation,
    )

    evaluate_rag.run_evaluation(
        top_k=2,
        index_missing=False,
        cases_path=tmp_path / "cases.json",
    )
    output = capsys.readouterr().out

    assert "Recall@2: 1/1 (100%)" in output
    assert "Precision@2: 50%" in output
    assert "Answer fidelity, reference correctness, and abstention are not scored" in output


def test_rag_evaluation_generates_only_with_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    cases = [
        {
            "query": "Question?",
            "expected_source": "menu.txt",
            "reference_answer": "Tomato soup.",
            "should_abstain": False,
        }
    ]

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            return None

        def scalars(self, statement):
            return self

        def all(self):
            return ["menu.txt"]

    class FakeRetriever:
        def __init__(self, session) -> None:
            del session

        def search_documents(self, query: str, *, top_k: int):
            del query, top_k
            return [
                DocumentSearchResult(
                    document_id=uuid4(),
                    source_name="menu.txt",
                    chunk_index=0,
                    content="The soup uses tomato.",
                    similarity=0.9,
                )
            ]

    monkeypatch.setattr(evaluate_rag, "load_cases", lambda _: cases)
    monkeypatch.setattr(evaluate_rag, "SessionLocal", FakeSession)
    monkeypatch.setattr(
        evaluate_rag,
        "PostgresDocumentRetriever",
        FakeRetriever,
    )
    calls = []
    monkeypatch.setattr(
        evaluate_rag,
        "generate_grounded_answer",
        lambda query, context, *, context_chunk_count: (
            calls.append((query, context, context_chunk_count))
            or "Tomato soup. [menu.txt#0]"
        ),
    )

    evaluate_rag.run_evaluation(
        top_k=2,
        index_missing=False,
        cases_path=tmp_path / "cases.json",
        generate_answers=True,
    )

    assert calls == [
        (
            "Question?",
            "[menu.txt#0]\nThe soup uses tomato.",
            1,
        )
    ]
    assert "answer=Tomato soup." in capsys.readouterr().out
