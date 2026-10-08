from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError

import pytest

from app.open_data_sources import (
    fetch_commons_photos,
    fetch_madrid_restaurants,
    parse_commons_photos,
    parse_madrid_restaurants,
)


def test_parse_madrid_restaurants_includes_places_without_commons_links() -> None:
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
                        "outdoor_seating": "yes",
                        "diet:vegetarian": "yes",
                        "wheelchair": "limited",
                        "wheelchair:description": "Escalón de 30 cm en la entrada",
                        "diet:kosher": "only",
                        "diet:kosher:certifier": "Certificador Ejemplo",
                        "check_date:diet:kosher": "2025-01-15",
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

    assert len(restaurants) == 2
    restaurant = restaurants[0]
    assert restaurant.name == "Restaurante Uno"
    assert restaurant.city == "Madrid"
    assert restaurant.cuisine == "italian"
    assert restaurant.location == "Calle Mayor 10, 28013"
    assert restaurant.source_url == "https://www.openstreetmap.org/node/12"
    assert restaurant.wikimedia_commons == "Category:Restaurante Uno"
    assert restaurant.features == (
        "Mesas al aire libre: disponible",
        "Opciones vegetarianas: disponible",
        "Comida kosher: exclusivo",
        "Acceso en silla de ruedas: accesibilidad limitada",
        "Certificador kosher: Certificador Ejemplo",
        "Última revisión kosher: 2025-01-15",
    )
    assert restaurants[1].name == "Sin fotos"
    assert restaurants[1].wikimedia_commons is None
    assert restaurants[1].features == ()


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


def test_madrid_restaurant_fetch_includes_places_without_commons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_data = {}

    def fake_request_json(url: str, *, data: bytes | None = None) -> dict:
        request_data["query"] = parse_qs(data.decode("utf-8"))["data"][0]
        return {"elements": []}

    monkeypatch.setattr("app.open_data_sources._request_json", fake_request_json)

    assert fetch_madrid_restaurants() == []
    assert 'nwr["amenity"="restaurant"](area.city);' in request_data["query"]
    assert '"wikimedia_commons"' not in request_data["query"]
    assert "out center;" in request_data["query"]


def test_madrid_restaurant_fetch_retries_transient_overpass_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def flaky_request_json(url: str, *, data: bytes | None = None) -> dict:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HTTPError(url, 504, "Gateway Timeout", None, None)
        return {"elements": []}

    monkeypatch.setattr("app.open_data_sources._request_json", flaky_request_json)
    monkeypatch.setattr("app.open_data_sources.time.sleep", lambda _: None)

    assert fetch_madrid_restaurants() == []
    assert attempts == 2
