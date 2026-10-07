from urllib.parse import parse_qs, urlparse

import pytest

from app.open_data_sources import (
    fetch_commons_photos,
    parse_commons_photos,
    parse_madrid_restaurants,
)


def test_parse_madrid_restaurants_keeps_only_places_with_commons_links() -> None:
    restaurants = parse_madrid_restaurants(
        {
            "elements": [
                {
                    "type": "node",
                    "id": 12,
                    "lat": 40.4,
                    "lon": -3.7,
                    "tags": {
                        "amenity": "restaurant",
                        "name": "Restaurante Uno",
                        "cuisine": "italian",
                        "wikimedia_commons": "Category:Restaurante Uno",
                        "addr:street": "Calle Mayor",
                        "addr:housenumber": "10",
                        "addr:postcode": "28013",
                    },
                },
                {
                    "type": "way",
                    "id": 34,
                    "center": {"lat": 40.5, "lon": -3.6},
                    "tags": {"amenity": "restaurant", "name": "Sin fotos"},
                },
            ]
        },
        limit=10,
    )

    assert len(restaurants) == 1
    restaurant = restaurants[0]
    assert restaurant.name == "Restaurante Uno"
    assert restaurant.city == "Madrid"
    assert restaurant.cuisine == "italian"
    assert restaurant.location == "Calle Mayor 10, 28013"
    assert restaurant.source_url == "https://www.openstreetmap.org/node/12"
    assert restaurant.wikimedia_commons == "Category:Restaurante Uno"


def test_parse_commons_photos_filters_unsupported_licenses() -> None:
    payload = {
        "query": {
            "pages": {
                "1": {
                    "title": "File:Reusable.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "thumburl": "https://thumb.wikimedia.org/reusable.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Reusable.jpg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY 4.0"},
                                "LicenseUrl": {
                                    "value": "https://creativecommons.org/licenses/by/4.0/"
                                },
                                "Artist": {"value": "<a>Jane Doe</a>"},
                            },
                        }
                    ],
                },
                "2": {
                    "title": "File:ShareAlike.jpg",
                    "imageinfo": [
                        {
                            "mime": "image/jpeg",
                            "thumburl": "https://upload.wikimedia.org/sharealike.jpg",
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:ShareAlike.jpg",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                "Artist": {"value": "John Doe"},
                            },
                        }
                    ],
                },
            }
        }
    }

    photos = parse_commons_photos(payload, limit=5)

    assert len(photos) == 1
    assert photos[0].title == "File:Reusable.jpg"
    assert photos[0].license_name == "CC BY 4.0"
    assert photos[0].attribution == "Jane Doe"
    assert photos[0].file_extension == ".jpg"


def test_commons_category_reference_is_sent_to_category_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_urls = []
    monkeypatch.setattr(
        "app.open_data_sources._request_json",
        lambda url, data=None: requested_urls.append(url) or {"query": {"pages": {}}},
    )

    photos = fetch_commons_photos("Category:Café Comercial", limit=2)

    assert photos == []
    parameters = parse_qs(urlparse(requested_urls[0]).query)
    assert parameters["generator"] == ["categorymembers"]
    assert parameters["gcmtitle"] == ["Category:Café Comercial"]
    assert parameters["gcmlimit"] == ["6"]


def test_commons_photo_lookup_rejects_unverified_references() -> None:
    with pytest.raises(ValueError, match="Category: or File:"):
        fetch_commons_photos("Café Comercial")
