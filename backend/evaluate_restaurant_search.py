import argparse
import json
from pathlib import Path
import sys
from typing import Literal, NotRequired, TypedDict

from app.database import SessionLocal
from app.restaurant_search import search_osm_places_by_text

CASES_PATH = (
    Path(__file__).parent
    / "data"
    / "restaurant_evaluation"
    / "madrid_cases.json"
)
EXPECTED_FIELDS = {
    "expected_feature_prefix",
    "expected_cuisine",
    "expected_name_contains",
}


class RestaurantEvaluationCase(TypedDict):
    query: str
    city: str
    category: Literal["feature", "dish", "cuisine", "name", "manual_review"]
    cuisine: NotRequired[str]
    intent: NotRequired[str]
    expected_feature_prefix: NotRequired[str]
    expected_cuisine: NotRequired[str]
    expected_name_contains: NotRequired[str]
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
) -> None:
    cases = load_cases(cases_path)
    scored_cases = [case for case in cases if not case.get("review_only")]
    review_cases = [case for case in cases if case.get("review_only")]
    if not scored_cases and not review_cases:
        raise ValueError("Restaurant evaluation data must include cases")

    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_ranks = []
    relevant_at_3 = 0
    feature_cases = [case for case in scored_cases if case["category"] == "feature"]
    feature_relevant_at_3 = 0

    with SessionLocal() as session:
        for case in scored_cases:
            search_response = search_osm_places_by_text(
                case["query"],
                session,
                top_k=top_k,
                city=case["city"],
                cuisine=case.get("cuisine"),
            )
            matches = search_response.results
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

            print(
                f"{case['category']} | "
                f"Hit@1={'yes' if hit_at_1 else 'no'} | "
                f"Hit@3={'yes' if hit_at_3 else 'no'} | "
                f"expected={next(iter(EXPECTED_FIELDS.intersection(case)))} | "
                f"query={case['query']} | "
                f"top3={[match.name for match in matches[:3]]}"
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
    if review_cases:
        print(
            "Manual-review queries are exploratory; inspect the listed OSM "
            "attributes. No similarity threshold is applied."
        )


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
    args = parser.parse_args()
    if not 3 <= args.top_k <= 20:
        parser.error("--top-k must be between 3 and 20")

    run_evaluation(top_k=args.top_k, cases_path=args.cases)


if __name__ == "__main__":
    main()
