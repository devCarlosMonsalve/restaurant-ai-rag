import argparse
import json
from pathlib import Path
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.answer_generation import generate_grounded_answer
from app.database import SessionLocal
from app.document_ingestion import ingest_txt_to_database
from app.document_search import search_document_chunks
from app.models.document_chunk import DocumentChunk
from app.rag import NO_DOCUMENTS_ANSWER

EVALUATION_DIR = Path(__file__).parent / "data" / "evaluation"
DOCUMENTS_DIR = EVALUATION_DIR / "documents"
CASES_PATH = EVALUATION_DIR / "cases.json"


class EvaluationCase(TypedDict):
    query: str
    expected_source: str | None
    should_abstain: bool


def load_cases() -> list[EvaluationCase]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def index_missing_documents(session: Session) -> None:
    indexed_sources = set(
        session.scalars(select(DocumentChunk.source_name).distinct()).all()
    )
    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        if path.name in indexed_sources:
            print(f"Already indexed: {path.name}")
            continue

        document_id, chunk_count = ingest_txt_to_database(path, session)
        indexed_sources.add(path.name)
        print(f"Indexed {path.name}: {chunk_count} chunks ({document_id})")


def run_evaluation(top_k: int, index_missing: bool) -> None:
    cases = load_cases()
    answerable_cases = [case for case in cases if case["expected_source"] is not None]
    if not answerable_cases:
        raise ValueError("Evaluation data must include answerable retrieval cases")

    hits = 0
    with SessionLocal() as session:
        if index_missing:
            index_missing_documents(session)

        for case in cases:
            results = search_document_chunks(case["query"], session, top_k=top_k)
            sources = [result.source_name for result in results]
            expected_source = case["expected_source"]

            if expected_source is not None:
                hit = expected_source in sources
                hits += int(hit)
                outcome = "HIT" if hit else "MISS"
                print(
                    f"{outcome} | expected={expected_source} | "
                    f"retrieved={sources} | query={case['query']}"
                )
                continue

            if results:
                answer = generate_grounded_answer(case["query"], results)
            else:
                answer = NO_DOCUMENTS_ANSWER
            print(
                "MANUAL ABSTENTION REVIEW | "
                f"should_abstain={case['should_abstain']} | "
                f"retrieved={sources} | answer={answer}"
            )

    total = len(answerable_cases)
    print(f"Hit@{top_k}: {hits}/{total} ({hits / total:.0%})")
    if hits != total:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval against a small restaurant document set."
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--index-missing",
        action="store_true",
        help="Embed and store evaluation documents that are not in PostgreSQL yet.",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    run_evaluation(args.top_k, args.index_missing)


if __name__ == "__main__":
    main()
