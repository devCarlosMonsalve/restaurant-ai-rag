import argparse
from itertools import chain
import json
from pathlib import Path
import sys
from typing import Literal, NotRequired, TypedDict

from sqlalchemy import select

from app.infrastructure.persistence.postgres.database import SessionLocal
from app.infrastructure.embeddings.text import EMBEDDING_DIMENSIONS, GEMINI_EMBEDDING_MODEL
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.infrastructure.persistence.postgres.restaurant_queries import search_osm_places_by_text
from evaluation.utils import (
    build_run_metadata,
    fingerprint_rows,
    write_evaluation_report,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = (
    BACKEND_ROOT
    / "data"
    / "restaurant_evaluation"
    / "madrid_cases.json"
)
EXPECTED_FIELDS = {
    "expected_feature_prefix",
    "expected_cuisine",
    "expected_name_contains",
    "expected_empty",
}


class RestaurantEvaluationCase(TypedDict):
    query: str
    city: str
    category: Literal[
        "feature",
        "dish",
        "cuisine",
        "name",
        "evidence",
        "manual_review",
    ]
    cuisine: NotRequired[str]
    intent: NotRequired[str]
    expected_feature_prefix: NotRequired[str]
    expected_cuisine: NotRequired[str]
    expected_name_contains: NotRequired[str]
    expected_empty: NotRequired[bool]
    expected_evidence_status: NotRequired[str]
    review_only: NotRequired[bool]


def load_cases(
    cases_path: Path = CASES_PATH,
) -> list[RestaurantEvaluationCase]:
    cases: list[RestaurantEvaluationCase] = json.loads(
        cases_path.read_text(encoding="utf-8")
    )
    if not cases:
        raise ValueError("Restaurant evaluation data must contain cases")

    for case in cases:
        criteria = EXPECTED_FIELDS.intersection(case)
        if case.get("review_only"):
            if criteria:
                raise ValueError("Manual-review cases cannot define expected results")
        elif len(criteria) != 1:
            raise ValueError(
                "Each scored case must define exactly one expected result criterion"
            )
        elif "expected_empty" in criteria and case["expected_empty"] is not True:
            raise ValueError("expected_empty must be true when present")
        if case.get("expected_evidence_status") and "expected_empty" not in case:
            raise ValueError(
                "expected_evidence_status is only supported for expected-empty cases"
            )
    return cases


def _matches_expected(
    name: str,
    cuisine: str | None,
    features: list[str],
    case: RestaurantEvaluationCase,
) -> bool:
    if expected_feature := case.get("expected_feature_prefix"):
        return any(
            feature.casefold().startswith(expected_feature.casefold())
            for feature in features
        )
    if expected_cuisine := case.get("expected_cuisine"):
        return expected_cuisine.casefold() in {
            value.strip().casefold() for value in (cuisine or "").split(";")
        }
    if expected_name := case.get("expected_name_contains"):
        return expected_name.casefold() in name.casefold()
    return False


def run_evaluation(
    *,
    top_k: int,
    cases_path: Path = CASES_PATH,
    report_path: Path | None = None,
) -> None:
    cases = load_cases(cases_path)
    scored_cases = [
        case
        for case in cases
        if not case.get("review_only") and "expected_empty" not in case
    ]
    empty_cases = [
        case for case in cases if case.get("expected_empty") is True
    ]
    review_cases = [case for case in cases if case.get("review_only")]
    if not scored_cases and not empty_cases and not review_cases:
        raise ValueError("Restaurant evaluation data must include cases")

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    relevant_at_3 = 0
    feature_cases = [case for case in scored_cases if case["category"] == "feature"]
    feature_relevant_at_3 = 0
    constrained_queries = 0
    compliant_queries = 0
    constrained_results = 0
    compliant_results = 0
    unexpected_empty_queries = 0
    case_reports: list[dict[str, object]] = []
    empty_case_results: list[bool] = []

    with SessionLocal() as session:
        if report_path:
            corpus_rows = chain(
                session.execute(
                    select(
                        OsmPlace.id,
                        OsmPlace.osm_type,
                        OsmPlace.osm_id,
                        OsmPlace.name,
                        OsmPlace.city,
                        OsmPlace.cuisine,
                        OsmPlace.features,
                        OsmPlace.source_url,
                        OsmPlace.embedding,
                    ).order_by(OsmPlace.id)
                ),
                session.execute(
                    select(
                        ImageEmbedding.id,
                        ImageEmbedding.osm_place_id,
                    ).order_by(ImageEmbedding.id)
                ),
            )
            corpus_fingerprint, corpus_row_count = fingerprint_rows(corpus_rows)
        else:
            corpus_fingerprint = ""
            corpus_row_count = 0

        for case in empty_cases:
            search_response = search_osm_places_by_text(
                case["query"],
                session,
                top_k=top_k,
                city=case["city"],
                cuisine=case.get("cuisine"),
            )
            no_results = not search_response.results
            evidence_matches = (
                search_response.evidence_status
                == case.get("expected_evidence_status")
                if case.get("expected_evidence_status")
                else True
            )
            passed = no_results and evidence_matches
            empty_case_results.append(passed)
            case_reports.append(
                {
                    "query": case["query"],
                    "expected_empty": True,
                    "result_count": len(search_response.results),
                    "passed": passed,
                    "evidence_status": search_response.evidence_status,
                    "expected_evidence_status": case.get(
                        "expected_evidence_status"
                    ),
                    "label_source": case.get("label_source"),
                }
            )
            print(
                f"Expected empty | {'PASS' if passed else 'FAIL'} | "
                f"evidence={search_response.evidence_status} | "
                f"query={case['query']}"
            )

        for case in scored_cases:
            search_response = search_osm_places_by_text(
                case["query"],
                session,
                top_k=top_k,
                city=case["city"],
                cuisine=case.get("cuisine"),
            )
            matches = search_response.results
            if not matches:
                unexpected_empty_queries += 1
            relevance = [
                _matches_expected(
                    match.name,
                    match.cuisine,
                    match.features,
                    case,
                )
                for match in matches
            ]
            first_relevant_rank = next(
                (rank for rank, is_relevant in enumerate(relevance, start=1) if is_relevant),
                None,
            )
            hit_at_1 = first_relevant_rank == 1
            hit_at_3 = first_relevant_rank is not None and first_relevant_rank <= 3
            hits_at_1 += hit_at_1
            hits_at_3 += hit_at_3
            reciprocal_ranks.append(
                1 / first_relevant_rank if first_relevant_rank is not None else 0
            )
            relevant_count = sum(relevance[:3])
            relevant_at_3 += relevant_count
            if case["category"] == "feature":
                feature_relevant_at_3 += relevant_count

            has_constraints = bool(case.get("city") or case.get("cuisine"))
            query_compliant = True
            if has_constraints:
                constrained_queries += 1
                for match in matches:
                    result_compliant = True
                    if case.get("city"):
                        result_compliant &= (
                            match.city.casefold() == case["city"].strip().casefold()
                        )
                    if case.get("cuisine"):
                        result_compliant &= case["cuisine"].casefold() in {
                            value.strip().casefold()
                            for value in (match.cuisine or "").split(";")
                        }
                    constrained_results += 1
                    compliant_results += int(result_compliant)
                    query_compliant &= result_compliant
                compliant_queries += int(query_compliant)

            print(
                f"{case['category']} | "
                f"Hit@1={'yes' if hit_at_1 else 'no'} | "
                f"Hit@3={'yes' if hit_at_3 else 'no'} | "
                f"expected={next(iter(EXPECTED_FIELDS.intersection(case)))} | "
                f"query={case['query']} | "
                f"top3={[match.name for match in matches[:3]]}"
            )
            case_reports.append(
                {
                    "query": case["query"],
                    "category": case["category"],
                    "expected": next(iter(EXPECTED_FIELDS.intersection(case))),
                    "result_count": len(matches),
                    "evidence_status": search_response.evidence_status,
                    "first_relevant_rank": first_relevant_rank,
                    "hit_at_1": hit_at_1,
                    "hit_at_3": hit_at_3,
                    "constraint_compliant": (
                        query_compliant if has_constraints else None
                    ),
                    "results": [
                        {
                            "id": match.id,
                            "name": match.name,
                            "city": match.city,
                            "cuisine": match.cuisine,
                            "similarity": match.similarity,
                        }
                        for match in matches[:top_k]
                    ],
                }
            )

        for case in review_cases:
            search_response = search_osm_places_by_text(
                case["query"],
                session,
                top_k=top_k,
                city=case["city"],
                cuisine=case.get("cuisine"),
            )
            print(
                f"MANUAL REVIEW | intent={case.get('intent', '')} | "
                f"evidence={search_response.evidence_status} | "
                f"evidence_message={search_response.evidence_message!r} | "
                f"query={case['query']} | top3="
                + str(
                    [
                        {
                            "name": match.name,
                            "cuisine": match.cuisine,
                            "features": match.features,
                            "similarity": round(match.similarity, 3),
                        }
                        for match in search_response.results[:3]
                    ]
                )
            )
            case_reports.append(
                {
                    "query": case["query"],
                    "review_only": True,
                    "evidence_status": search_response.evidence_status,
                    "result_count": len(search_response.results),
                    "results": [
                        {
                            "name": match.name,
                            "cuisine": match.cuisine,
                            "features": match.features,
                            "similarity": round(match.similarity, 3),
                        }
                        for match in search_response.results[:3]
                    ],
                }
            )

    if scored_cases:
        total = len(scored_cases)
        hit_at_3_precision = relevant_at_3 / (total * 3)
        print(
            f"Overall: Hit@1={hits_at_1}/{total} ({hits_at_1 / total:.0%}); "
            f"Hit@3={hits_at_3}/{total} ({hits_at_3 / total:.0%}); "
            f"MRR@{top_k}={sum(reciprocal_ranks) / total:.3f}; "
            f"Precision@3={hit_at_3_precision:.0%}"
        )
        if feature_cases:
            feature_precision_at_3 = feature_relevant_at_3 / (
                len(feature_cases) * 3
            )
            print(
                f"Feature Precision@3={feature_precision_at_3:.0%} "
                f"({feature_relevant_at_3}/{len(feature_cases) * 3} results)"
            )
    if empty_cases:
        print(
            f"Expected-empty accuracy: {sum(empty_case_results)}/"
            f"{len(empty_case_results)} "
            f"({sum(empty_case_results) / len(empty_case_results):.0%})"
        )
    if scored_cases:
        print(
            "Unexpected empty-query rate: "
            f"{unexpected_empty_queries}/{len(scored_cases)} "
            f"({unexpected_empty_queries / len(scored_cases):.0%})"
        )
    constraint_compliance = (
        compliant_queries / constrained_queries if constrained_queries else None
    )
    result_compliance = (
        compliant_results / constrained_results if constrained_results else None
    )
    if constrained_queries:
        print(
            "Constraint compliance: "
            f"{compliant_queries}/{constrained_queries} queries "
            f"({constraint_compliance:.0%}); "
            f"{compliant_results}/{constrained_results} returned results "
            f"({result_compliance:.0%})"
        )
    if review_cases:
        print(
            "Manual-review queries are exploratory; inspect the listed OSM "
            "attributes. No similarity threshold is applied."
        )
    if report_path:
        metadata = build_run_metadata(
            evaluator="restaurant_search",
            cases_path=cases_path,
            configuration={
                "top_k": top_k,
                "embedding_model": GEMINI_EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "ranking": "cosine_distance",
                "default_excludes_places_with_photos": True,
            },
            corpus_name="osm_places plus image_embeddings association rows",
            corpus_fingerprint=corpus_fingerprint,
            corpus_row_count=corpus_row_count,
        )
        report = {
            "metadata": metadata,
            "metrics": {
                "hit_at_1": hits_at_1 / len(scored_cases) if scored_cases else None,
                "hit_at_3": hits_at_3 / len(scored_cases) if scored_cases else None,
                "mrr_at_k": (
                    sum(reciprocal_ranks) / len(scored_cases)
                    if scored_cases
                    else None
                ),
                "precision_at_3": (
                    relevant_at_3 / (len(scored_cases) * 3)
                    if scored_cases
                    else None
                ),
                "expected_empty_accuracy": (
                    sum(empty_case_results) / len(empty_case_results)
                    if empty_case_results
                    else None
                ),
                "unexpected_empty_query_rate": (
                    unexpected_empty_queries / len(scored_cases)
                    if scored_cases
                    else None
                ),
                "constraint_compliant_query_rate": constraint_compliance,
                "constraint_compliant_result_rate": result_compliance,
            },
            "cases": case_reports,
        }
        write_evaluation_report(report_path, report)
        print(f"JSON report written to {report_path}")


def main() -> None:
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="Evaluate semantic search over Madrid OSM restaurant metadata."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum ranking depth for MRR (default: 5; must be at least 3).",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="Path to a JSON evaluation case file.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a JSON report with dataset, corpus, code, and configuration fingerprints.",
    )
    args = parser.parse_args()
    if not 3 <= args.top_k <= 20:
        parser.error("--top-k must be between 3 and 20")

    run_evaluation(
        top_k=args.top_k,
        cases_path=args.cases,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
