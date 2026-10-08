import argparse
import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.image_search import search_images_by_text
from app.models.image_embedding import ImageEmbedding

EVALUATION_DIR = Path(__file__).parent / "data" / "image_evaluation"
CASES_PATH = EVALUATION_DIR / "cases.json"
IMAGES_DIR = Path(__file__).parent / "data" / "images"


class ImageEvaluationCase(TypedDict):
    query: str
    expected_image: str | None
    expected_restaurant: NotRequired[str | None]
    should_abstain: bool
    language: Literal["en", "es"]


def _expected_result(case: ImageEvaluationCase) -> str | None:
    return case.get("expected_restaurant") or case["expected_image"]


def load_cases(cases_path: Path = CASES_PATH) -> list[ImageEvaluationCase]:
    cases: list[ImageEvaluationCase] = json.loads(
        cases_path.read_text(encoding="utf-8")
    )
    if not cases:
        raise ValueError("Image evaluation data must contain at least one case")
    if any(
        case["should_abstain"] == (_expected_result(case) is not None)
        or (
            case["expected_image"] is not None
            and case.get("expected_restaurant") is not None
        )
        for case in cases
    ):
        raise ValueError(
            "Abstention cases must have no expected result; answerable cases "
            "must specify exactly one expected image or restaurant"
        )
    return cases


def _hit_count(
    evaluations: list[tuple[ImageEvaluationCase, list[str]]],
    *,
    cutoff: int,
) -> int:
    return sum(
        _expected_result(case) in retrieved_images[:cutoff]
        for case, retrieved_images in evaluations
    )


def _print_metrics(
    label: str,
    evaluations: list[tuple[ImageEvaluationCase, list[str]]],
    *,
    top_k: int,
) -> None:
    total = len(evaluations)
    hit_at_1 = _hit_count(evaluations, cutoff=1)
    hit_at_k = _hit_count(evaluations, cutoff=top_k)
    print(f"{label} Hit@1: {hit_at_1}/{total} ({hit_at_1 / total:.0%})")
    print(f"{label} Hit@{top_k}: {hit_at_k}/{total} ({hit_at_k / total:.0%})")


def run_evaluation(
    top_k: int,
    *,
    cases_path: Path = CASES_PATH,
    osm_places_only: bool = False,
) -> None:
    cases = load_cases(cases_path)
    answerable_cases = [
        case for case in cases if not case["should_abstain"]
    ]
    if not answerable_cases:
        raise ValueError("Image evaluation data must include answerable cases")

    expected_images = {
        case["expected_image"]
        for case in answerable_cases
        if case["expected_image"] is not None
    }
    missing_files = sorted(
        image_name
        for image_name in expected_images
        if not any(IMAGES_DIR.rglob(image_name))
    )
    if missing_files:
        raise ValueError(
            "Evaluation image files are missing from data/images: "
            + ", ".join(missing_files)
        )

    evaluations: list[tuple[ImageEvaluationCase, list[str]]] = []
    positive_top_scores: list[float] = []
    negative_top_scores: list[float] = []
    with SessionLocal() as session:
        indexed_images = set(
            session.scalars(select(ImageEmbedding.source_name).distinct()).all()
        )
        missing_indexed = sorted(expected_images - indexed_images)
        if missing_indexed:
            raise ValueError(
                "Evaluation images are not indexed in PostgreSQL: "
                + ", ".join(missing_indexed)
                + ". Run `python ingest_images.py .\\data\\images` first."
            )

        for case in cases:
            matches = search_images_by_text(
                case["query"],
                session,
                top_k=top_k,
                osm_places_only=osm_places_only,
                sample_images_only=not osm_places_only,
            )
            expected_restaurant = case.get("expected_restaurant")
            retrieved_images = [
                (match.restaurant_name or "")
                if expected_restaurant
                else match.source_name
                for match in matches
            ]
            top_similarity = matches[0].similarity if matches else None

            if case["should_abstain"]:
                if top_similarity is not None:
                    negative_top_scores.append(top_similarity)
                print(
                    "ABSTENTION REVIEW | "
                    f"top_similarity={top_similarity} | "
                    f"retrieved={retrieved_images} | query={case['query']}"
                )
                continue

            expected_result = _expected_result(case)
            if expected_result is None:
                raise ValueError("Answerable evaluation cases need an expected result")
            evaluations.append((case, retrieved_images))
            if top_similarity is not None:
                positive_top_scores.append(top_similarity)

            hit_at_1 = bool(retrieved_images) and (
                retrieved_images[0] == expected_result
            )
            hit_at_k = expected_result in retrieved_images
            expected_label = (
                "expected_restaurant"
                if expected_restaurant
                else "expected_image"
            )
            print(
                f"{'HIT' if hit_at_1 else 'MISS'}@1 | "
                f"{'HIT' if hit_at_k else 'MISS'}@{top_k} | "
                f"{expected_label}={expected_result} | "
                f"top_similarity={top_similarity} | "
                f"retrieved={retrieved_images} | query={case['query']}"
            )

    _print_metrics("Overall", evaluations, top_k=top_k)
    for language, label in (("en", "English"), ("es", "Spanish")):
        language_evaluations = [
            evaluation
            for evaluation in evaluations
            if evaluation[0]["language"] == language
        ]
        if language_evaluations:
            _print_metrics(label, language_evaluations, top_k=top_k)

    if positive_top_scores:
        print(
            "Positive top-similarity range: "
            f"{min(positive_top_scores):.3f}-{max(positive_top_scores):.3f} "
            f"(mean={sum(positive_top_scores) / len(positive_top_scores):.3f})"
        )
    if negative_top_scores:
        print(
            "Negative top-similarity range: "
            f"{min(negative_top_scores):.3f}-{max(negative_top_scores):.3f} "
            f"(mean={sum(negative_top_scores) / len(negative_top_scores):.3f})"
        )
        print("Negative cases are for threshold calibration; no threshold is applied.")

    if _hit_count(evaluations, cutoff=top_k) != len(evaluations):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bilingual text-to-image retrieval with CLIP."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Ranking cutoff for Hit@K (default: 3).",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=CASES_PATH,
        help="Path to a JSON evaluation case file.",
    )
    parser.add_argument(
        "--osm-places-only",
        action="store_true",
        help="Restrict retrieval to photos linked to OpenStreetMap places.",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    run_evaluation(
        args.top_k,
        cases_path=args.cases,
        osm_places_only=args.osm_places_only,
    )


if __name__ == "__main__":
    main()
