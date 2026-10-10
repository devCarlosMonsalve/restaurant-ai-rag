from collections.abc import Sequence


def osm_searchable_metadata_changed(
    *,
    current_name: str,
    current_city: str,
    current_cuisine: str | None,
    current_location: str | None,
    current_features: Sequence[str] | None,
    new_name: str,
    new_city: str,
    new_cuisine: str | None,
    new_location: str | None,
    new_features: Sequence[str],
) -> bool:
    return any(
        current != new
        for current, new in (
            (current_name, new_name),
            (current_city, new_city),
            (current_cuisine, new_cuisine),
            (current_location, new_location),
        )
    ) or (list(current_features or []) != list(new_features))
