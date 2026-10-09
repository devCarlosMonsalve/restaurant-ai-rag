from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings import embed_search_query
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.place_filters import cuisine_filter
from app.schemas import OsmRestaurantSearchResponse, OsmRestaurantSearchResult
from app.search_evidence import (
    SearchEvidenceRequest,
    detect_search_evidence,
    feature_requirements_clause,
    matches_feature_requirements,
)

OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_ATTRIBUTION_URL = "https://www.openstreetmap.org/copyright"


def search_osm_places_by_text(
    query: str,
    session: Session,
    *,
    top_k: int = 12,
    city: str | None = None,
    cuisine: str | None = None,
    include_places_with_photos: bool = False,
) -> OsmRestaurantSearchResponse:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = embed_search_query(query)
    evidence_request = detect_search_evidence(query)
    cosine_distance = OsmPlace.embedding.cosine_distance(query_embedding)
    has_photos = (
        select(ImageEmbedding.id)
        .where(ImageEmbedding.osm_place_id == OsmPlace.id)
        .exists()
    )
    statement = (
        select(OsmPlace, cosine_distance)
        .where(OsmPlace.embedding.is_not(None))
        .order_by(cosine_distance)
    )
    if not include_places_with_photos:
        statement = statement.where(~has_photos)
    if city:
        statement = statement.where(
            func.lower(OsmPlace.city) == city.strip().lower()
        )
    if cuisine:
        statement = statement.where(cuisine_filter(cuisine))

    if evidence_request.feature_requirements:
        statement = statement.where(
            feature_requirements_clause(
                OsmPlace.features,
                evidence_request.feature_requirements,
            )
        )
    else:
        statement = statement.limit(top_k)

    candidate_places = list(session.execute(statement))
    matching_places = [
        (place, distance)
        for place, distance in candidate_places
        if matches_feature_requirements(
            place.features or [],
            evidence_request.feature_requirements,
        )
    ]
    results = [
        OsmRestaurantSearchResult(
            id=place.id,
            name=place.name,
            city=place.city,
            cuisine=place.cuisine,
            location=place.location,
            latitude=place.latitude,
            longitude=place.longitude,
            features=place.features or [],
            source_url=place.source_url,
            attribution=OSM_ATTRIBUTION,
            attribution_url=OSM_ATTRIBUTION_URL,
            similarity=1.0 - float(distance),
        )
        for place, distance in matching_places[:top_k]
    ]
    evidence_status, evidence_message = _evidence_summary(
        evidence_request,
        len(results),
        len(candidate_places),
    )
    return OsmRestaurantSearchResponse(
        results=results,
        evidence_status=evidence_status,
        evidence_message=evidence_message,
    )


def _evidence_summary(
    evidence_request: SearchEvidenceRequest,
    result_count: int,
    candidate_count: int,
) -> tuple[str, str | None]:
    feature_labels = [
        requirement.label for requirement in evidence_request.feature_requirements
    ]
    verified_text = ", ".join(feature_labels)
    unverified_text = ", ".join(evidence_request.unverified_requirements)

    if not feature_labels and not unverified_text:
        return "not_required", None
    if feature_labels and result_count == 0:
        freshness_requirement = next(
            (
                requirement
                for requirement in evidence_request.feature_requirements
                if requirement.max_age_days is not None
            ),
            None,
        )
        if freshness_requirement and candidate_count:
            return (
                "stale_evidence",
                "Hay etiquetas kosher en OSM, pero falta una fecha de revisión "
                f"vigente (máximo {freshness_requirement.max_age_days} días). "
                "No mostramos esas fichas como certificaciones actuales.",
            )
        message = (
            f"No encontramos fichas con etiquetas OSM que confirmen: "
            f"{verified_text}. No mostramos coincidencias sin evidencia como "
            "resultados confirmados."
        )
        if unverified_text:
            message += f" Tampoco se puede verificar: {unverified_text}."
        if _requires_step_free_access(evidence_request):
            message += (
                " No encontramos lugares con wheelchair=yes, la etiqueta OSM "
                "que indica entrada sin escalones."
            )
        return "no_evidence", message
    if feature_labels and unverified_text:
        return (
            "partial",
            f"Las fichas mostradas tienen etiquetas OSM para {verified_text}, "
            f"pero no podemos confirmar: {unverified_text}. Verifica ese "
            "requisito directamente con el restaurante.",
        )
    if feature_labels:
        message = (
            f"Las fichas mostradas tienen etiquetas OSM para: {verified_text}. "
            "Esto no confirma otras condiciones no etiquetadas."
        )
        if _requires_step_free_access(evidence_request):
            message += (
                " Según la convención OSM, wheelchair=yes indica entrada y "
                "salas sin escalones; confirma que el dato siga vigente."
            )
        freshness_requirement = next(
            (
                requirement
                for requirement in evidence_request.feature_requirements
                if requirement.max_age_days is not None
            ),
            None,
        )
        if freshness_requirement:
            message += (
                f" Para kosher se exige revisión OSM dentro de los últimos "
                f"{freshness_requirement.max_age_days} días."
            )
        return "verified", message
    return (
        "unverified",
        f"OpenStreetMap no aporta datos fiables para confirmar {unverified_text}. "
        "Los resultados semánticos son sugerencias; verifica ese requisito "
        "directamente con el restaurante.",
    )


def _requires_step_free_access(
    evidence_request: SearchEvidenceRequest,
) -> bool:
    return any(
        requirement.label == "Acceso en silla de ruedas"
        and requirement.value == "accesible"
        for requirement in evidence_request.feature_requirements
    )
