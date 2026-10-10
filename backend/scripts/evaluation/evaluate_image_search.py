import argparse
import json
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.infrastructure.embeddings.image import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    IMAGE_EMBEDDING_DIMENSIONS,
)
from app.infrastructure.persistence.postgres.database import SessionLocal
from app.infrastructure.persistence.postgres.image_queries import search_images_by_text
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from evaluation.utils import (
    build_run_metadata,
    fingerprint_rows,
    write_evaluation_report,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = BACKEND_ROOT / "data" / "image_evaluation"
CASES_PATH = EVALUATION_DIR / "cases.json"
ATTRIBUTION_CASES_PATH = EVALUATION_DIR / "attribution_cases.json"
IMAGES_DIR = BACKEND_ROOT / "data" / "images"


class ImageEvaluationCase(TypedDict):
    query: str
    expected_image: str | None
    expected_restaurant: NotRequired[str | None]
    should_abstain: bool
    manual_review: NotRequired[bool]
    language: Literal["en", "es"]


class ImageAttributionCase(TypedDict):
    query: str
    osm_type: Literal["node", "way", "relation"]
    osm_id: int
    expected_name: str
    expected_source_url: str


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
    if not any(
        not case["should_abstain"] and not case.get("manual_review")
        for case in cases
    ):
        raise ValueError("Image evaluation data must include scored positive cases")
    return cases


def load_attribution_cases(
    cases_path: Path = ATTRIBUTION_CASES_PATH,
) -> list[ImageAttributionCase]:
    cases: list[ImageAttributionCase] = json.loads(
        cases_path.read_text(encoding="utf-8")
    )
    if not cases:
        raise ValueError("Image attribution evaluation data must contain cases")
    if len({(case["osm_type"], case["osm_id"]) for case in cases}) != len(cases):
        raise ValueError("Image attribution cases must use unique OSM places")
    if any(
        case["osm_id"] <= 0
        or case["expected_source_url"]
        != (
            "https://www.openstreetmap.org/"
            f"{case['osm_type']}/{case['osm_id']}"
        )
        for case in cases
    ):
        raise ValueError("Image attribution cases need canonical OSM source URLs")
    return cases


def _image_corpus_fingerprint(session: Session) -> tuple[str, int]:
    rows = session.execute(
        select(
            ImageEmbedding.id,
            ImageEmbedding.source_name,
            ImageEmbedding.embedding,
            ImageEmbedding.source_url,
            ImageEmbedding.osm_place_id,
            OsmPlace.osm_type,
            OsmPlace.osm_id,
            OsmPlace.name,
            OsmPlace.source_url,
        )
        .outerjoin(OsmPlace, ImageEmbedding.osm_place_id == OsmPlace.id)
        .order_by(ImageEmbedding.id)
    )
    return fingerprint_rows(rows)


def run_attribution_evaluation(
    top_k: int,
    *,
    cases_path: Path = ATTRIBUTION_CASES_PATH,
    report_path: Path | None = None,
) -> None:
    cases = load_attribution_cases(cases_path)
    case_reports: list[dict[str, object]] = []
    linked_photo_count = 0
    valid_photo_count = 0
    candidates_without_indexed_photos = 0
    empty_results_for_candidates_without_photos = 0
    photo_presence_matches = 0

    with SessionLocal() as session:
        if report_path:
            corpus_fingerprint, corpus_row_count = _image_corpus_fingerprint(
                session
            )
        else:
            corpus_fingerprint = ""
            corpus_row_count = 0

        for case in cases:
            place = session.scalar(
                select(OsmPlace).where(
                    OsmPlace.osm_type == case["osm_type"],
                    OsmPlace.osm_id == case["osm_id"],
                )
            )
            if place is None:
                raise ValueError(
                    "Stale image attribution case: "
                    f"{case['osm_type']}/{case['osm_id']} is not indexed"
                )
            if (
                place.name.casefold() != case["expected_name"].casefold()
                or place.source_url != case["expected_source_url"]
            ):
                raise ValueError(
                    "Stale image attribution label for "
                    f"{case['osm_type']}/{case['osm_id']}: "
                    "the indexed name or exact source_url changed"
                )

            indexed_photo_count = session.scalar(
                select(func.count(ImageEmbedding.id)).where(
                    ImageEmbedding.osm_place_id == place.id
                )
            )
            indexed_photo_count = int(indexed_photo_count or 0)
            matches = search_images_by_text(
                case["query"],
                session,
                top_k=top_k,
                osm_places_only=True,
                osm_place_id=place.id,
            )
            association_rows = (
                session.execute(
                    select(
                        ImageEmbedding.id,
                        ImageEmbedding.osm_place_id,
                        OsmPlace.id,
                        OsmPlace.source_url,
                    )
                    .outerjoin(
                        OsmPlace,
                        ImageEmbedding.osm_place_id == OsmPlace.id,
                    )
                    .where(
                        ImageEmbedding.id.in_(
                            [match.id for match in matches]
                        )
                    )
                )
                if matches
                else []
            )
            associations = {row[0]: row[1:] for row in association_rows}
            valid_for_case = 0
            for match in matches:
                association = associations.get(match.id)
                is_valid = bool(
                    association
                    and association[0] == place.id
                    and association[1] == place.id
                    and association[2] == place.source_url
                    and match.restaurant_source_url == place.source_url
                )
                linked_photo_count += 1
                valid_photo_count += int(is_valid)
                valid_for_case += int(is_valid)

            no_photos = indexed_photo_count == 0
            if no_photos:
                candidates_without_indexed_photos += 1
                empty_results_for_candidates_without_photos += int(not matches)
            presence_matches = bool(matches) == bool(indexed_photo_count)
            photo_presence_matches += int(presence_matches)
            report_case = {
                "query": case["query"],
                "candidate": {
                    "osm_type": place.osm_type,
                    "osm_id": place.osm_id,
                    "name": place.name,
                    "source_url": place.source_url,
                },
                "indexed_photo_count": indexed_photo_count,
                "returned_photo_count": len(matches),
                "returned_ids_and_urls_valid": valid_for_case == len(matches),
                "photo_presence_matches_index": presence_matches,
                "photos": [
                    {
                        "id": match.id,
                        "source_name": match.source_name,
                        "restaurant_name": match.restaurant_name,
                        "restaurant_source_url": match.restaurant_source_url,
                        "similarity": match.similarity,
                    }
                    for match in matches
                ],
            }
            case_reports.append(report_case)
            print(
                f"{place.osm_type}/{place.osm_id} {place.name} | "
                f"indexed={indexed_photo_count} | returned={len(matches)} | "
                f"ID+URL={'PASS' if valid_for_case == len(matches) else 'FAIL'}"
            )

    id_url_accuracy = (
        valid_photo_count / linked_photo_count if linked_photo_count else None
    )
    no_photo_empty_rate = (
        empty_results_for_candidates_without_photos
        / candidates_without_indexed_photos
        if candidates_without_indexed_photos
        else None
    )
    presence_accuracy = photo_presence_matches / len(cases)
    print(
        f"Candidate photo ID+URL integrity: "
        f"{valid_photo_count}/{linked_photo_count} "
        f"({id_url_accuracy:.0%})"
        if linked_photo_count
        else "Candidate photo ID+URL integrity: no returned photos to score"
    )
    print(
        "Empty results when index has no candidate photos: "
        f"{empty_results_for_candidates_without_photos}/"
        f"{candidates_without_indexed_photos}"
        + (
            f" ({no_photo_empty_rate:.0%})"
            if no_photo_empty_rate is not None
            else " (no candidates without indexed photos)"
        )
    )
    print(
        f"Photo-presence agreement with index: "
        f"{photo_presence_matches}/{len(cases)} ({presence_accuracy:.0%})"
    )
    print(
        "This checks attribution and index presence only; visual relevance "
        "requires separate human labels."
    )

    if report_path:
        metadata = build_run_metadata(
            evaluator="image_candidate_attribution",
            cases_path=cases_path,
            configuration={
                "top_k": top_k,
                "image_text_model": CLIP_MODEL_NAME,
                "image_text_pretrained": CLIP_PRETRAINED,
                "embedding_dimensions": IMAGE_EMBEDDING_DIMENSIONS,
                "candidate_filter": "exact OsmPlace.id / ImageEmbedding.osm_place_id",
                "url_validation": "exact source_url equality",
            },
            corpus_name="image_embeddings joined with osm_places",
            corpus_fingerprint=corpus_fingerprint,
            corpus_row_count=corpus_row_count,
        )
        write_evaluation_report(
            report_path,
            {
                "metadata": metadata,
                "metrics": {
                    "candidate_photo_id_url_integrity": id_url_accuracy,
                    "empty_result_rate_for_candidates_without_photos": (
                        no_photo_empty_rate
                    ),
                    "photo_presence_agreement_with_index": presence_accuracy,
                    "returned_linked_photo_count": linked_photo_count,
                },
                "cases": case_reports,
                "review_note": (
                    "The candidate IDs and exact source URLs are labeled. "
                    "Returned photos are verified against the stored OSM foreign "
                    "key. These checks do not judge whether a photo is visually "
                    "relevant."
                ),
            },
        )
        print(f"JSON report written to {report_path}")

    if (
        valid_photo_count != linked_photo_count
        or photo_presence_matches != len(cases)
        or empty_results_for_candidates_without_photos
        != candidates_without_indexed_photos
    ):
        raise SystemExit(1)


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
    report_path: Path | None = None,
) -> None:
    cases = load_cases(cases_path)
    answerable_cases = [
        case
        for case in cases
        if not case["should_abstain"] and not case.get("manual_review")
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
    negative_nonempty_count = 0
    negative_case_count = 0
    associated_image_count = 0
    valid_association_count = 0
    case_reports: list[dict[str, object]] = []
    with SessionLocal() as session:
        if report_path:
            corpus_fingerprint, corpus_row_count = _image_corpus_fingerprint(
                session
            )
        else:
            corpus_fingerprint = ""
            corpus_row_count = 0

        indexed_images = set(
            session.scalars(select(ImageEmbedding.source_name).distinct()).all()
        )
        missing_indexed = sorted(expected_images - indexed_images)
        if missing_indexed:
            raise ValueError(
                "Evaluation images are not indexed in PostgreSQL: "
                + ", ".join(missing_indexed)
                + ". Run `python -m scripts.ingestion.ingest_images .\\data\\images` first."
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
            case_associations: list[dict[str, object]] = []
            if osm_places_only and matches:
                association_rows = session.execute(
                    select(
                        ImageEmbedding.id,
                        ImageEmbedding.osm_place_id,
                        OsmPlace.id,
                        OsmPlace.osm_type,
                        OsmPlace.osm_id,
                        OsmPlace.source_url,
                    )
                    .outerjoin(
                        OsmPlace,
                        ImageEmbedding.osm_place_id == OsmPlace.id,
                    )
                    .where(
                        ImageEmbedding.id.in_(
                            [match.id for match in matches]
                        )
                    )
                )
                associations = {
                    row[0]: row[1:]
                    for row in association_rows
                }
                for match in matches:
                    association = associations.get(match.id)
                    if association is None:
                        raise RuntimeError(
                            "A retrieved image is missing from image_embeddings"
                        )
                    image_place_id, place_id, osm_type, osm_id, source_url = (
                        association
                    )
                    is_associated = image_place_id is not None
                    association_valid = (
                        is_associated
                        and image_place_id == place_id
                        and match.restaurant_source_url == source_url
                    )
                    associated_image_count += int(is_associated)
                    valid_association_count += int(association_valid)
                    case_associations.append(
                        {
                            "image_id": match.id,
                            "source_name": match.source_name,
                            "osm_place": (
                                f"{osm_type}/{osm_id}" if place_id else None
                            ),
                            "restaurant_source_url": match.restaurant_source_url,
                            "association_valid": association_valid,
                        }
                    )

            if case["should_abstain"]:
                negative_case_count += 1
                if top_similarity is not None:
                    negative_top_scores.append(top_similarity)
                negative_nonempty_count += int(bool(matches))
                print(
                    "ABSTENTION REVIEW | "
                    f"top_similarity={top_similarity} | "
                    f"retrieved={retrieved_images} | query={case['query']}"
                )
                case_reports.append(
                    {
                        "query": case["query"],
                        "should_abstain": True,
                        "result_count": len(matches),
                        "retrieved": retrieved_images,
                        "top_similarity": top_similarity,
                        "osm_associations": case_associations,
                        "manual_relevance_review": True,
                    }
                )
                continue

            if case.get("manual_review"):
                print(
                    "MANUAL LABEL REVIEW | "
                    f"expected={_expected_result(case)} | "
                    f"top_similarity={top_similarity} | "
                    f"retrieved={retrieved_images} | query={case['query']}"
                )
                case_reports.append(
                    {
                        "query": case["query"],
                        "expected_result_pending_validation": _expected_result(
                            case
                        ),
                        "result_count": len(matches),
                        "retrieved": retrieved_images,
                        "top_similarity": top_similarity,
                        "manual_review_required": True,
                        "reason": (
                            "The query asks for Mestizo while the legacy label "
                            "expects Xamach; confirm the target OSM place before "
                            "using this case as a retrieval score."
                        ),
                        "osm_associations": case_associations,
                    }
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
            expected_hit_at_1 = bool(retrieved_images) and (
                retrieved_images[0] == expected_result
            )
            expected_hit_at_k = expected_result in retrieved_images
            case_reports.append(
                {
                    "query": case["query"],
                    "expected_result": expected_result,
                    "result_count": len(matches),
                    "retrieved": retrieved_images,
                    "top_similarity": top_similarity,
                    "hit_at_1": expected_hit_at_1,
                    f"hit_at_{top_k}": expected_hit_at_k,
                    "osm_associations": case_associations,
                    "visual_relevance_is_separate_from_attribution": True,
                }
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
    if negative_case_count:
        print(
            "Out-of-corpus non-empty rate: "
            f"{negative_nonempty_count}/{negative_case_count} "
            f"({negative_nonempty_count / negative_case_count:.0%}); "
            "this is not an abstention metric because search has no rejection threshold."
        )

    association_accuracy = (
        valid_association_count / associated_image_count
        if associated_image_count
        else None
    )
    if osm_places_only:
        print(
            "OSM photo attribution integrity: "
            f"{valid_association_count}/{associated_image_count} linked results "
            f"({association_accuracy:.0%})"
            if associated_image_count
            else "OSM photo attribution integrity: no linked results to score"
        )

    if _hit_count(evaluations, cutoff=top_k) != len(evaluations):
        evaluation_failed = True
    else:
        evaluation_failed = False
    if report_path:
        metadata = build_run_metadata(
            evaluator="image_search",
            cases_path=cases_path,
            configuration={
                "top_k": top_k,
                "corpus_filter": "osm_places" if osm_places_only else "sample_or_all",
                "image_text_model": CLIP_MODEL_NAME,
                "image_text_pretrained": CLIP_PRETRAINED,
                "embedding_dimensions": IMAGE_EMBEDDING_DIMENSIONS,
                "ranking": "metadata_token_counts_then_cosine_distance",
            },
            corpus_name="image_embeddings joined with osm_places",
            corpus_fingerprint=corpus_fingerprint,
            corpus_row_count=corpus_row_count,
        )
        report = {
            "metadata": metadata,
            "metrics": {
                "hit_at_1": _hit_count(evaluations, cutoff=1) / len(evaluations),
                f"hit_at_{top_k}": _hit_count(
                    evaluations,
                    cutoff=top_k,
                )
                / len(evaluations),
                "out_of_corpus_nonempty_rate": (
                    negative_nonempty_count / negative_case_count
                    if negative_case_count
                    else None
                ),
                "osm_attribution_integrity": association_accuracy,
                "osm_linked_result_count": associated_image_count,
                "valid_osm_link_count": valid_association_count,
            },
            "cases": case_reports,
            "review_note": (
                "Hit@k measures the expected image/restaurant label. "
                "OSM attribution integrity checks stored image-to-place IDs and "
                "exact source_url equality; neither metric establishes visual "
                "suitability. Cases flagged manual_review are excluded from "
                "scored relevance metrics."
            ),
        }
        write_evaluation_report(report_path, report)
        print(f"JSON report written to {report_path}")
    if evaluation_failed or (
        osm_places_only and valid_association_count != associated_image_count
    ):
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
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a JSON report with dataset, corpus, code, and configuration fingerprints.",
    )
    parser.add_argument(
        "--attribution",
        action="store_true",
        help="Evaluate candidate-ID photo attribution using the dedicated OSM cases.",
    )
    parser.add_argument(
        "--attribution-cases",
        type=Path,
        default=ATTRIBUTION_CASES_PATH,
        help="Path to OSM candidate attribution cases used with --attribution.",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    if args.attribution:
        if args.osm_places_only:
            parser.error("--attribution selects the OSM corpus automatically")
        run_attribution_evaluation(
            args.top_k,
            cases_path=args.attribution_cases,
            report_path=args.report,
        )
    else:
        run_evaluation(
            args.top_k,
            cases_path=args.cases,
            osm_places_only=args.osm_places_only,
            report_path=args.report,
        )


if __name__ == "__main__":
    main()
