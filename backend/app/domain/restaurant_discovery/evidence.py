"""Compatibility exports for Restaurant Discovery domain evidence policies."""

from app.restaurant_discovery.domain.evidence import (
    KOSHER_MAX_AGE_DAYS,
    FeatureRequirement,
    SearchEvidenceRequest,
    detect_search_evidence,
    matches_feature_requirements,
)

__all__ = [
    "KOSHER_MAX_AGE_DAYS",
    "FeatureRequirement",
    "SearchEvidenceRequest",
    "detect_search_evidence",
    "matches_feature_requirements",
]
