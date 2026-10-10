from sqlalchemy import and_, cast, or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.elements import ColumnElement

from app.restaurant_discovery.domain.evidence import FeatureRequirement


def feature_requirements_clause(
    features_column: ColumnElement[object],
    requirements: tuple[FeatureRequirement, ...],
) -> ColumnElement[bool]:
    if not requirements:
        raise ValueError("At least one feature requirement is required")

    features_json = cast(features_column, JSONB)
    possible_values = (
        "disponible",
        "exclusivo",
        "limitado",
        "accesible",
        "accesibilidad limitada",
        "Wi-Fi disponible",
        "acceso por cable",
    )
    clauses = []
    for requirement in requirements:
        values = (requirement.value,) if requirement.value else possible_values
        clauses.append(
            or_(
                *(
                    features_json.contains([f"{requirement.label}: {value}"])
                    for value in values
                )
            )
        )
    return and_(*clauses)
