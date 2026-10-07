import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

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
    expected_image: str
    language: Literal["en", "es"]


def load_cases() -> list[ImageEvaluationCase]:
    cases: list[ImageEvaluationCase] = json.loads(
        CASES_PATH.read_text(encoding="utf-8")
    )
    if not cases:
        raise ValueError("Image evaluation data must contain at least one case")
    return cases


def _hit_count(
    evaluations: list[tuple[ImageEvaluationCase, list[str]]],
    *,
    cutoff: int,
) -> int:
    return sum(
        case["expected_image"] in retrieved_images[:cutoff]
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


def run_evaluation(top_k: int) -> None:
    cases = load_cases()
    expected_images = {case["expected_image"] for case in cases}
    missing_files = sorted(
        image_name
        for image_name in expected_images
        if not (IMAGES_DIR / image_name).is_file()
    )
    if missing_files:
        raise ValueError(
            "Evaluation image files are missing from data/images: "
            + ", ".join(missing_files)
        )

    evaluations: list[tuple[ImageEvaluationCase, list[str]]] = []
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
            matches = search_images_by_text(case["query"], session, top_k=top_k)
            retrieved_images = [match.source_name for match in matches]
            evaluations.append((case, retrieved_images))

            hit_at_1 = bool(retrieved_images) and (
                retrieved_images[0] == case["expected_image"]
            )
            hit_at_k = case["expected_image"] in retrieved_images
            print(
                f"{'HIT' if hit_at_1 else 'MISS'}@1 | "
                f"{'HIT' if hit_at_k else 'MISS'}@{top_k} | "
                f"expected={case['expected_image']} | "
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
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    run_evaluation(args.top_k)


if __name__ == "__main__":
    main()
