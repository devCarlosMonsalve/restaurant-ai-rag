from collections import Counter
from datetime import date
from typing import Any, Iterable

from app.restaurant_discovery.domain.evidence import KOSHER_MAX_AGE_DAYS
from app.restaurant_discovery.domain.osm_features import OSM_FEATURE_TAGS


def build_coverage_report(
    places: Iterable[Any],
    *,
    photo_place_ids: set[Any],
    audit_date: date,
    holdout_judgments: list[dict[str, Any]],
    city: str,
) -> dict[str, Any]:
    place_list = list(places)
    feature_counts: Counter[str] = Counter()
    feature_values: dict[str, Counter[str]] = {
        label: Counter() for label in OSM_FEATURE_TAGS.values()
    }
    kosher_places = []
    embedded_count = 0

    for place in place_list:
        embedded_count += place.embedding is not None
        place_labels = set()
        place_values: set[tuple[str, str]] = set()
        for feature in place.features or []:
            label, separator, value = feature.partition(":")
            label = label.strip()
            value = value.strip()
            if not separator or label not in feature_values:
                continue
            place_labels.add(label)
            place_values.add((label, value))
        feature_counts.update(place_labels)
        for label, value in place_values:
            feature_values[label][value] += 1
        if "Comida kosher" in place_labels:
            kosher_places.append(place.features or [])

    feature_coverage = [
        {
            "feature": label,
            "restaurant_count": feature_counts[label],
            "coverage_percent": round(
                feature_counts[label] / len(place_list) * 100, 2
            )
            if place_list
            else 0.0,
            "values": dict(sorted(feature_values[label].items())),
        }
        for label in OSM_FEATURE_TAGS.values()
    ]
    kosher_status = Counter(
        _kosher_date_status(features, audit_date) for features in kosher_places
    )
    photo_place_count = sum(place.id in photo_place_ids for place in place_list)

    return {
        "audit_date": audit_date.isoformat(),
        "city": city,
        "restaurant_count": len(place_list),
        "embedding_count": embedded_count,
        "photo_place_count": photo_place_count,
        "no_photo_place_count": len(place_list) - photo_place_count,
        "feature_coverage": feature_coverage,
        "kosher_freshness": {
            "tagged_restaurants": len(kosher_places),
            "policy_max_age_days": KOSHER_MAX_AGE_DAYS,
            "current": kosher_status["current"],
            "stale": kosher_status["stale"],
            "future_date": kosher_status["future_date"],
            "missing_date": kosher_status["missing_date"],
            "invalid_date": kosher_status["invalid_date"],
        },
        "holdout_review": summarize_holdout_judgments(holdout_judgments),
    }


def summarize_holdout_judgments(
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    judgment_counts = Counter(item["judgment"] for item in judgments)
    evidence_counts = Counter(item["evidence_status"] for item in judgments)
    return {
        "query_count": len(judgments),
        "judgments": dict(sorted(judgment_counts.items())),
        "evidence_statuses": dict(sorted(evidence_counts.items())),
        "review_mode": "manual",
    }


def _kosher_date_status(features: list[str], audit_date: date) -> str:
    raw_date = next(
        (
            feature.partition(":")[2].strip()
            for feature in features
            if feature.partition(":")[0].strip() == "Última revisión kosher"
        ),
        None,
    )
    if not raw_date:
        return "missing_date"
    try:
        checked_date = date.fromisoformat(raw_date)
    except ValueError:
        return "invalid_date"
    age_days = (audit_date - checked_date).days
    if age_days < 0:
        return "future_date"
    if age_days > KOSHER_MAX_AGE_DAYS:
        return "stale"
    return "current"
