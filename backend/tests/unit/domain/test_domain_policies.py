from app.restaurant_discovery.domain.photo_association import (
    is_verified_photo_association,
)


def test_photo_association_requires_exact_nonempty_source_url() -> None:
    candidate_source_url = "https://www.openstreetmap.org/node/123"

    assert is_verified_photo_association(
        candidate_source_url,
        candidate_source_url,
    )
    assert not is_verified_photo_association(candidate_source_url, None)
    assert not is_verified_photo_association(candidate_source_url, "")
    assert not is_verified_photo_association(
        candidate_source_url,
        "https://www.openstreetmap.org/node/456",
    )
