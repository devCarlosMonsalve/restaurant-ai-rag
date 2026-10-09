import argparse
import json
from pathlib import Path
from typing import NotRequired, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.answer_generation import (
    GEMINI_GENERATION_MODEL,
    generate_grounded_answer,
)
from app.database import SessionLocal
from app.document_ingestion import ingest_txt_to_database
from app.document_search import search_document_chunks
from app.models.document_chunk import DocumentChunk
from app.rag import NO_DOCUMENTS_ANSWER
from app.embeddings import EMBEDDING_DIMENSIONS, GEMINI_EMBEDDING_MODEL
from evaluation_utils import (
    build_run_metadata,
    fingerprint_rows,
    write_evaluation_report,
)

EVALUATION_DIR = Path(__file__).parent / "data" / "evaluation"
DOCUMENTS_DIR = EVALUATION_DIR / "documents"
CASES_PATH = EVALUATION_DIR / "cases.json"


class EvaluationCase(TypedDict):
    query: str
    expected_source: str | None
    expected_sources: NotRequired[list[str]]
    reference_answer: NotRequired[str]
    case_type: NotRequired[str]
    should_abstain: bool


def _expected_sources(case: EvaluationCase) -> list[str]:
    if sources := case.get("expected_sources"):
        return sources
    if source := case.get("expected_source"):
        return [source]
    return []


def load_cases(cases_path: Path = CASES_PATH) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = json.loads(
        cases_path.read_text(encoding="utf-8")
    )
    if not cases:
        raise ValueError("RAG evaluation data must contain cases")
    for case in cases:
        if case.get("expected_source") and case.get("expected_sources"):
            raise ValueError(
                "Use expected_source or expected_sources, not both"
            )
        if case["should_abstain"] == bool(_expected_sources(case)):
            raise ValueError(
                "Answerable cases need expected sources; abstention cases must not"
            )
    return cases


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


def run_evaluation(
    top_k: int,
    index_missing: bool,
    *,
    cases_path: Path = CASES_PATH,
    generate_answers: bool = False,
    report_path: Path | None = None,
) -> None:
    cases = load_cases(cases_path)
    answerable_cases = [case for case in cases if _expected_sources(case)]
    if not answerable_cases:
        raise ValueError("Evaluation data must include answerable retrieval cases")

    relevant_source_count = 0
    retrieved_relevant_source_count = 0
    relevant_chunk_count = 0
    reciprocal_ranks: list[float] = []
    case_reports: list[dict[str, object]] = []
    with SessionLocal() as session:
        if index_missing:
            index_missing_documents(session)

        indexed_sources = set(
            session.scalars(
                select(DocumentChunk.source_name).distinct()
            ).all()
        )
        expected_sources = {
            source_name
            for case in answerable_cases
            for source_name in _expected_sources(case)
        }
        missing_sources = sorted(expected_sources - indexed_sources)
        if missing_sources:
            raise ValueError(
                "Labeled document sources are missing from PostgreSQL: "
                + ", ".join(missing_sources)
                + ". Verify the corpus or explicitly use --index-missing."
            )

        if report_path:
            corpus_rows = session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.document_id,
                    DocumentChunk.source_name,
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    DocumentChunk.embedding,
                ).order_by(
                    DocumentChunk.document_id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.id,
                )
            )
            corpus_fingerprint, corpus_row_count = fingerprint_rows(corpus_rows)
        else:
            corpus_fingerprint = ""
            corpus_row_count = 0

        for case in cases:
            results = search_document_chunks(case["query"], session, top_k=top_k)
            sources = [result.source_name for result in results]
            expected_sources = set(_expected_sources(case))
            answer: str | None = None
            if generate_answers:
                if results:
                    answer = generate_grounded_answer(case["query"], results)
                else:
                    answer = NO_DOCUMENTS_ANSWER

            if expected_sources:
                relevant_ranks = [
                    rank
                    for rank, result in enumerate(results, start=1)
                    if result.source_name in expected_sources
                ]
                retrieved_sources = set(sources) & expected_sources
                retrieved_relevant_source_count += len(retrieved_sources)
                relevant_source_count += len(expected_sources)
                relevant_chunk_count += len(relevant_ranks)
                first_relevant_rank = (
                    min(relevant_ranks) if relevant_ranks else None
                )
                reciprocal_ranks.append(
                    1 / first_relevant_rank if first_relevant_rank else 0
                )
                print(
                    f"{'HIT' if retrieved_sources else 'MISS'} | "
                    f"expected={sorted(expected_sources)} | "
                    f"retrieved={sources} | query={case['query']}"
                    + (f" | answer={answer}" if answer is not None else "")
                )
            else:
                first_relevant_rank = None
                print(
                    "ABSTENTION REVIEW | "
                    f"should_abstain={case['should_abstain']} | "
                    f"case_type={case.get('case_type', 'unlabeled')} | "
                    f"retrieved={sources}"
                    + (f" | answer={answer}" if answer is not None else "")
                )

            case_report: dict[str, object] = {
                "query": case["query"],
                "expected_sources": sorted(expected_sources),
                "retrieved_sources": sources,
                "retrieved_chunks": [
                    {
                        "source_name": result.source_name,
                        "chunk_index": result.chunk_index,
                        "similarity": result.similarity,
                        "content": result.content,
                    }
                    for result in results
                ],
                "should_abstain": case["should_abstain"],
                "case_type": case.get("case_type"),
                "first_relevant_rank": first_relevant_rank,
            }
            if reference_answer := case.get("reference_answer"):
                case_report["reference_answer"] = reference_answer
            if answer is not None:
                case_report["generated_answer"] = answer
                case_report["human_review"] = {
                    "faithful_to_retrieved_context": None,
                    "matches_reference_answer": None,
                    "abstains_when_evidence_is_insufficient": None,
                }
            case_reports.append(case_report)

    total = len(answerable_cases)
    recall_at_k = retrieved_relevant_source_count / relevant_source_count
    precision_at_k = relevant_chunk_count / (total * top_k)
    mrr_at_k = sum(reciprocal_ranks) / total
    hit_at_k = sum(rank > 0 for rank in reciprocal_ranks)
    print(f"Hit@{top_k}: {hit_at_k}/{total} ({hit_at_k / total:.0%})")
    print(
        f"Recall@{top_k}: {retrieved_relevant_source_count}/"
        f"{relevant_source_count} ({recall_at_k:.0%}); "
        f"Precision@{top_k}: {precision_at_k:.0%}; "
        f"MRR@{top_k}: {mrr_at_k:.3f}"
    )
    print(
        "Answer fidelity, reference correctness, and abstention are not scored "
        "automatically; generated answers require human review."
    )
    if report_path:
        metadata = build_run_metadata(
            evaluator="document_rag",
            cases_path=cases_path,
            configuration={
                "top_k": top_k,
                "embedding_model": GEMINI_EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "generation_model": (
                    GEMINI_GENERATION_MODEL if generate_answers else None
                ),
                "generation_opt_in": generate_answers,
                "index_missing_documents": index_missing,
                "ranking": "cosine_distance",
            },
            corpus_name="document_chunks",
            corpus_fingerprint=corpus_fingerprint,
            corpus_row_count=corpus_row_count,
        )
        report = {
            "metadata": metadata,
            "metrics": {
                f"hit_at_{top_k}": hit_at_k / total,
                f"recall_at_{top_k}": recall_at_k,
                f"precision_at_{top_k}": precision_at_k,
                f"mrr_at_{top_k}": mrr_at_k,
                "answer_fidelity": None,
                "reference_correctness": None,
                "unsupported_answer_abstention": None,
            },
            "cases": case_reports,
            "review_note": (
                "Retrieval metrics use labeled source files. Answer fidelity, "
                "factual correctness, and abstention remain unscored until a "
                "human reviews generated answers against the retrieved chunks "
                "and reference answers."
            ),
        }
        write_evaluation_report(report_path, report)
        print(f"JSON report written to {report_path}")

    if retrieved_relevant_source_count != relevant_source_count:
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
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="Path to a JSON evaluation case file.",
    )
    parser.add_argument(
        "--generate-answers",
        action="store_true",
        help="Opt in to Gemini answer generation for manual fidelity/correctness review.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a JSON report with dataset, corpus, code, and configuration fingerprints.",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    run_evaluation(
        args.top_k,
        args.index_missing,
        cases_path=args.cases,
        generate_answers=args.generate_answers,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
