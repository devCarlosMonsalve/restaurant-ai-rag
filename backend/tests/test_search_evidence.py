from datetime import date
from uuid import uuid4

from app.restaurant_search import search_osm_places_by_text
from app.search_evidence import (
    detect_search_evidence,
    matches_feature_requirements,
)


def test_search_evidence_detects_specific_features_and_unverified_details() -> None:
    evidence = detect_search_evidence(
        "entrada accesible sin escalones y un menú totalmente vegano"
    )

    assert [requirement.label for requirement in evidence.feature_requirements] == [
        "Opciones veganas",
        "Acceso en silla de ruedas",
    ]
    assert evidence.feature_requirements[0].value == "exclusivo"
    assert evidence.feature_requirements[1].value == "accesible"
    assert evidence.unverified_requirements == ()


def test_search_evidence_detects_outdoor_and_celiac_synonyms() -> None:
    outdoor = detect_search_evidence("cenar al fresco")
    gluten_free = detect_search_evidence("platos aptos para celíacos")

    assert [requirement.label for requirement in outdoor.feature_requirements] == [
        "Mesas al aire libre"
    ]
    assert [requirement.label for requirement in gluten_free.feature_requirements] == [
        "Opciones sin gluten"
    ]


def test_live_music_search_requires_osm_feature_evidence() -> None:
    evidence = detect_search_evidence("un restaurante con música en vivo")
    english_evidence = detect_search_evidence("restaurant with live music")

    assert [requirement.label for requirement in evidence.feature_requirements] == [
        "Música en vivo"
    ]
    assert english_evidence.feature_requirements == evidence.feature_requirements
    assert matches_feature_requirements(
        ["Música en vivo: disponible"],
        evidence.feature_requirements,
    )
    assert not matches_feature_requirements([], evidence.feature_requirements)


def test_feature_match_requires_all_requested_tags() -> None:
    evidence = detect_search_evidence("terraza y opciones veganas")

    assert matches_feature_requirements(
        [
            "Mesas al aire libre: disponible",
            "Opciones veganas: exclusivo",
        ],
        evidence.feature_requirements,
    )
    assert not matches_feature_requirements(
        ["Mesas al aire libre: disponible"],
        evidence.feature_requirements,
    )


def test_fully_vegan_request_requires_exclusive_osm_tag() -> None:
    evidence = detect_search_evidence("menú totalmente vegano")

    assert not matches_feature_requirements(
        ["Opciones veganas: disponible"],
        evidence.feature_requirements,
    )
    assert matches_feature_requirements(
        ["Opciones veganas: exclusivo"],
        evidence.feature_requirements,
    )


def test_kosher_search_requires_a_recent_check_date() -> None:
    requirement = detect_search_evidence("comida kosher").feature_requirements
    fresh_features = [
        "Comida kosher: disponible",
        "Certificador kosher: Certificador Ejemplo",
        "Última revisión kosher: 2025-10-08",
    ]

    assert matches_feature_requirements(
        fresh_features,
        requirement,
        today=date(2026, 10, 8),
    )
    assert not matches_feature_requirements(
        ["Comida kosher: disponible"],
        requirement,
        today=date(2026, 10, 8),
    )
    assert not matches_feature_requirements(
        [
            "Comida kosher: disponible",
            "Última revisión kosher: 2025-10-07",
        ],
        requirement,
        today=date(2026, 10, 8),
    )
    assert not matches_feature_requirements(
        [
            "Comida kosher: disponible",
            "Última revisión kosher: 2026-10-09",
        ],
        requirement,
        today=date(2026, 10, 8),
    )
    assert not matches_feature_requirements(
        [
            "Comida kosher: disponible",
            "Última revisión kosher: fecha-desconocida",
        ],
        requirement,
        today=date(2026, 10, 8),
    )


def test_step_free_access_requires_full_wheelchair_access_tag() -> None:
    evidence = detect_search_evidence("entrada accesible sin escalones")

    assert not matches_feature_requirements(
        ["Acceso en silla de ruedas: accesibilidad limitada"],
        evidence.feature_requirements,
    )
    assert matches_feature_requirements(
        ["Acceso en silla de ruedas: accesible"],
        evidence.feature_requirements,
    )


def test_feature_search_skips_unverified_semantic_matches(
    monkeypatch,
) -> None:
    untagged_place = _place("Unverified", [])
    tagged_place = _place("Tagged", ["Reservas: disponible"])

    class FakeSession:
        def execute(self, statement):
            return [
                (untagged_place, 0.01),
                (tagged_place, 0.2),
            ]

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "un restaurante donde pueda reservar mesa",
        FakeSession(),
        top_k=1,
    )

    assert [result.name for result in response.results] == ["Tagged"]
    assert response.evidence_status == "verified"


def test_feature_search_without_matching_tags_returns_no_evidence(
    monkeypatch,
) -> None:
    untagged_place = _place("Unverified", [])

    class FakeSession:
        def execute(self, statement):
            return []

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "cocina kosher",
        FakeSession(),
    )

    assert response.results == []
    assert response.evidence_status == "no_evidence"
    assert "Comida kosher" in response.evidence_message


def test_stale_kosher_data_is_not_returned_as_verified(
    monkeypatch,
) -> None:
    place = _place(
        "Old certification",
        [
            "Comida kosher: disponible",
            "Última revisión kosher: 2020-01-01",
        ],
    )

    class FakeSession:
        def execute(self, statement):
            return [(place, 0.01)]

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "cocina kosher",
        FakeSession(),
    )

    assert response.results == []
    assert response.evidence_status == "stale_evidence"
    assert "fecha de revisión vigente" in response.evidence_message


def test_unverified_ambiance_keeps_semantic_results_but_warns(
    monkeypatch,
) -> None:
    place = _place("Quiet Restaurant", [])

    class FakeSession:
        def execute(self, statement):
            return [(place, 0.2)]

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "cena romantica en un lugar tranquilo",
        FakeSession(),
    )

    assert [result.name for result in response.results] == ["Quiet Restaurant"]
    assert response.evidence_status == "unverified"
    assert "ambiente del restaurante" in response.evidence_message


def test_partial_evidence_warns_about_unverified_ambiance(
    monkeypatch,
) -> None:
    place = _place("Vegan Restaurant", ["Opciones veganas: disponible"])

    class FakeSession:
        def execute(self, statement):
            return [(place, 0.2)]

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "opciones veganas en un lugar tranquilo",
        FakeSession(),
    )

    assert [result.name for result in response.results] == ["Vegan Restaurant"]
    assert response.evidence_status == "partial"
    assert "no podemos confirmar: el ambiente del restaurante" in (
        response.evidence_message
    )


def test_step_free_access_is_confirmed_with_osm_wheelchair_yes(
    monkeypatch,
) -> None:
    place = _place("Restaurant", ["Acceso en silla de ruedas: accesible"])

    class FakeSession:
        def execute(self, statement):
            return [(place, 0.2)]

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "entrada accesible sin escalones",
        FakeSession(),
    )

    assert [result.name for result in response.results] == ["Restaurant"]
    assert response.evidence_status == "verified"
    assert "wheelchair=yes" in response.evidence_message


def test_live_music_search_only_returns_tagged_places(monkeypatch) -> None:
    untagged_place = _place("Unverified", [])
    tagged_place = _place("Live Music", ["Música en vivo: disponible"])

    class FakeSession:
        def execute(self, statement):
            return [(untagged_place, 0.01), (tagged_place, 0.2)]

    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    response = search_osm_places_by_text(
        "un restaurante con música en vivo",
        FakeSession(),
    )

    assert [result.name for result in response.results] == ["Live Music"]
    assert response.evidence_status == "verified"


def _place(name: str, features: list[str]):
    return type(
        "Place",
        (),
        {
            "id": uuid4(),
            "osm_type": "node",
            "osm_id": 1,
            "name": name,
            "city": "Madrid",
            "cuisine": None,
            "location": None,
            "latitude": None,
            "longitude": None,
            "features": features,
            "source_url": "https://www.openstreetmap.org/node/1",
        },
    )()
