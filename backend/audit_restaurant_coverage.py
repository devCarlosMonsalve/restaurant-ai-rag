import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.coverage_audit import build_coverage_report
from app.database import SessionLocal
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace

HOLDOUT_JUDGMENTS_PATH = (
    Path(__file__).parent
    / "data"
    / "restaurant_evaluation"
    / "madrid_holdout_judgments.json"
)


def load_holdout_judgments(path: Path = HOLDOUT_JUDGMENTS_PATH) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    judgments = payload["judgments"]
    if not isinstance(judgments, list):
        raise ValueError("Holdout judgments must be a JSON list")
    return judgments


def run_audit(*, audit_date: date, city: str) -> dict[str, Any]:
    with SessionLocal() as session:
        places = session.execute(
            select(OsmPlace.id, OsmPlace.features, OsmPlace.embedding).where(
                OsmPlace.city.ilike(city.strip())
            )
        ).all()
        photo_place_ids = set(
            session.scalars(
                select(ImageEmbedding.osm_place_id)
                .join(OsmPlace, ImageEmbedding.osm_place_id == OsmPlace.id)
                .where(OsmPlace.city.ilike(city.strip()))
                .distinct()
            )
        )

    return build_coverage_report(
        places,
        photo_place_ids=photo_place_ids,
        audit_date=audit_date,
        holdout_judgments=load_holdout_judgments(),
        city=city.strip(),
    )


def _print_text_report(report: dict[str, Any]) -> None:
    print(
        f"OSM restaurant coverage for {report['city']} "
        f"(audit date: {report['audit_date']})"
    )
    print(
        f"Restaurants: {report['restaurant_count']} "
        f"(embeddings: {report['embedding_count']}, "
        f"with photos: {report['photo_place_count']}, "
        f"without photos: {report['no_photo_place_count']})"
    )
    print("Feature coverage:")
    for item in report["feature_coverage"]:
        values = ", ".join(
            f"{value}={count}" for value, count in item["values"].items()
        )
        print(
            f"  {item['feature']}: {item['restaurant_count']} "
            f"({item['coverage_percent']:.2f}%)"
            + (f" [{values}]" if values else "")
        )
    kosher = report["kosher_freshness"]
    print(
        "Kosher freshness: "
        f"tagged={kosher['tagged_restaurants']}, current={kosher['current']}, "
        f"stale={kosher['stale']}, future={kosher['future_date']}, "
        f"missing_date={kosher['missing_date']}, "
        f"invalid_date={kosher['invalid_date']} "
        f"(max age {kosher['policy_max_age_days']} days)"
    )
    holdout = report["holdout_review"]
    print(
        f"Holdout: {holdout['query_count']} manually reviewed queries; "
        f"judgments={holdout['judgments']}; "
        f"evidence_statuses={holdout['evidence_statuses']}"
    )
    print("Holdout judgments are manual review, not automatic ranking metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit OpenStreetMap evidence coverage for restaurant search."
    )
    parser.add_argument("--city", default="Madrid")
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=datetime.now(timezone.utc).date(),
        help="Audit date in YYYY-MM-DD format (defaults to the current UTC date).",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format; JSON is suitable for automation.",
    )
    args = parser.parse_args()
    report = run_audit(audit_date=args.as_of, city=args.city)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
