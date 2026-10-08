from dataclasses import dataclass
import re
import unicodedata

from sqlalchemy import and_, cast, or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.elements import ColumnElement


@dataclass(frozen=True)
class FeatureRequirement:
    label: str
    value: str | None = None


@dataclass(frozen=True)
class SearchEvidenceRequest:
    feature_requirements: tuple[FeatureRequirement, ...]
    unverified_requirements: tuple[str, ...]


_FEATURE_INTENTS = (
    (
        "Mesas al aire libre",
        (
            "terraza",
            "terrazas",
            "al aire libre",
            "al fresco",
            "afuera",
            "exterior",
            "outdoor seating",
        ),
    ),
    ("Opciones veganas", ("vegano", "vegana", "veganos", "veganas", "vegan")),
    (
        "Opciones vegetarianas",
        ("vegetariano", "vegetariana", "vegetarianos", "vegetarianas", "vegetarian"),
    ),
    (
        "Opciones sin gluten",
        (
            "sin gluten",
            "celiaco",
            "celiaca",
            "celiacos",
            "celiacas",
            "celiac",
        ),
    ),
    ("Opciones sin lactosa", ("sin lactosa", "lactose free")),
    ("Comida halal", ("halal",)),
    ("Comida kosher", ("kosher",)),
    (
        "Acceso en silla de ruedas",
        (
            "silla de ruedas",
            "sillas de ruedas",
            "wheelchair",
            "accesible",
            "accesibilidad",
        ),
    ),
    ("Aire acondicionado", ("aire acondicionado", "climatizado", "climatizada")),
    ("Acceso a internet", ("wifi", "wi fi", "internet")),
    (
        "Comida para llevar",
        ("para llevar", "para recoger", "recogida", "takeaway"),
    ),
    (
        "Reparto a domicilio",
        ("a domicilio", "reparto", "delivery"),
    ),
    (
        "Reservas",
        (
            "reservar",
            "reserva mesa",
            "reservas disponibles",
            "hacer una reserva",
            "hacer reserva",
            "book a table",
        ),
    ),
)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = normalized.encode("ascii", errors="ignore").decode("ascii")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_text).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {_normalize(phrase)} " in f" {text} "


def detect_search_evidence(query: str) -> SearchEvidenceRequest:
    normalized_query = _normalize(query)
    step_free_requested = any(
        _contains_phrase(normalized_query, phrase)
        for phrase in (
            "sin escalones",
            "sin escaleras",
            "sin peldanos",
            "step free",
            "no stairs",
        )
    )
    requirements = []
    for label, phrases in _FEATURE_INTENTS:
        if any(_contains_phrase(normalized_query, phrase) for phrase in phrases):
            value = None
            if label == "Opciones veganas" and any(
                _contains_phrase(normalized_query, phrase)
                for phrase in ("totalmente vegano", "menu vegano", "todo vegano")
            ):
                value = "exclusivo"
            requirements.append(FeatureRequirement(label=label, value=value))

    if step_free_requested:
        wheelchair_label = "Acceso en silla de ruedas"
        requirements = [
            requirement
            for requirement in requirements
            if requirement.label != wheelchair_label
        ]
        requirements.append(FeatureRequirement(wheelchair_label, "accesible"))

    unverified = []
    if any(
        _contains_phrase(normalized_query, phrase)
        for phrase in (
            "romantico",
            "romantica",
            "tranquilo",
            "tranquila",
            "silencioso",
            "intimo",
            "quiet",
            "romantic",
        )
    ):
        unverified.append("el ambiente del restaurante")

    return SearchEvidenceRequest(
        feature_requirements=tuple(requirements),
        unverified_requirements=tuple(unverified),
    )


def matches_feature_requirements(
    features: list[str],
    requirements: tuple[FeatureRequirement, ...],
) -> bool:
    for requirement in requirements:
        matching_values = [
            feature.partition(":")[2].strip().casefold()
            for feature in features
            if feature.partition(":")[0].strip().casefold()
            == requirement.label.casefold()
        ]
        if not matching_values:
            return False
        if requirement.value and requirement.value.casefold() not in matching_values:
            return False
    return True


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
                    features_json.contains(
                        [f"{requirement.label}: {value}"]
                    )
                    for value in values
                )
            )
        )
    return and_(*clauses)
