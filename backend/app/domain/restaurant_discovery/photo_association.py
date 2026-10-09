def is_verified_photo_association(
    candidate_source_url: str,
    photo_source_url: str | None,
) -> bool:
    return bool(photo_source_url) and photo_source_url == candidate_source_url
