from datetime import date
from types import SimpleNamespace
from uuid import uuid4

from app.coverage_audit import build_coverage_report


def test_coverage_report_counts_restaurants_once_and_classifies_kosher_dates() -> None:
    photo_place_id = uuid4()
    places = [
        _place(
            photo_place_id,
            [
                "Mesas al aire libre: disponible",
                "Mesas al aire libre: disponible",
                "Comida kosher: disponible",
                "Última revisión kosher: 2026-10-08",
            ],
            embedding=[0.0] * 768,
        ),
        _place(
            uuid4(),
            [
                "Acceso en silla de ruedas: accesible",
                "Comida kosher: disponible",
                "Última revisión kosher: 2020-01-01",
            ],
        ),
        _place(
            uuid4(),
            [
                "Comida kosher: disponible",
                "Última revisión kosher: 2026-10-09",
            ],
            embedding=[0.0] * 768,
        ),
        _place(
            uuid4(),
            ["Comida kosher: disponible"],
            embedding=[0.0] * 768,
        ),
        _place(
            uuid4(),
            [
                "Comida kosher: disponible",
                "Última revisión kosher: not-a-date",
            ],
            embedding=[0.0] * 768,
        ),
    ]

    report = build_coverage_report(
        places,
        photo_place_ids={photo_place_id},
        audit_date=date(2026, 10, 8),
        holdout_judgments=[],
        city="Madrid",
    )

    assert report["restaurant_count"] == 5
    assert report["embedding_count"] == 4
    assert report["photo_place_count"] == 1
    assert report["no_photo_place_count"] == 4
    features = {item["feature"]: item for item in report["feature_coverage"]}
    assert features["Mesas al aire libre"]["restaurant_count"] == 1
    assert features["Mesas al aire libre"]["values"] == {"disponible": 1}
    assert features["Mesas al aire libre"]["coverage_percent"] == 20.0
    assert report["kosher_freshness"] == {
        "tagged_restaurants": 5,
        "policy_max_age_days": 365,
        "current": 1,
        "stale": 1,
        "future_date": 1,
        "missing_date": 1,
        "invalid_date": 1,
    }


def test_holdout_summary_keeps_manual_judgments_distinct() -> None:
    report = build_coverage_report(
        [],
        photo_place_ids=set(),
        audit_date=date(2026, 10, 8),
        holdout_judgments=[
            {"judgment": "relevant", "evidence_status": "verified"},
            {"judgment": "no_evidence", "evidence_status": "no_evidence"},
            {"judgment": "unverified", "evidence_status": "unverified"},
        ],
        city="Madrid",
    )

    assert report["holdout_review"] == {
        "query_count": 3,
        "judgments": {
            "no_evidence": 1,
            "relevant": 1,
            "unverified": 1,
        },
        "evidence_statuses": {
            "no_evidence": 1,
            "unverified": 1,
            "verified": 1,
        },
        "review_mode": "manual",
    }


def _place(
    place_id,
    features: list[str],
    *,
    embedding: list[float] | None = None,
):
    return SimpleNamespace(id=place_id, features=features, embedding=embedding)
