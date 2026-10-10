import html
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from app.restaurant_discovery.domain.osm_features import OSM_FEATURE_TAGS


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_IMAGE_HOSTS = {"upload.wikimedia.org", "thumb.wikimedia.org"}
USER_AGENT = (
    "restaurant-ai-rag/0.1 "
    "(educational project; https://github.com/devCarlosMonsalve/restaurant-ai-rag)"
)
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
SUPPORTED_LICENSES = {
    "cc0",
    "public domain",
    "cc by 1.0",
    "cc by 2.0",
    "cc by 2.5",
    "cc by 3.0",
    "cc by 4.0",
}
_POSITIVE_FEATURE_VALUES = {"yes", "only", "limited", "wlan", "wired"}
_FEATURE_VALUE_LABELS = {
    "yes": "disponible",
    "only": "exclusivo",
    "limited": "limitado",
    "wlan": "Wi-Fi disponible",
    "wired": "acceso por cable",
}


@dataclass(frozen=True)
class OSMRestaurant:
    osm_type: str
    osm_id: int
    name: str
    city: str
    cuisine: str | None
    location: str | None
    latitude: float
    longitude: float
    wikimedia_commons: str | None
    features: tuple[str, ...] = field(default_factory=tuple)

    @property
    def source_url(self) -> str:
        return f"https://www.openstreetmap.org/{self.osm_type}/{self.osm_id}"


@dataclass(frozen=True)
class CommonsPhoto:
    title: str
    image_url: str
    source_url: str
    mime_type: str
    license_name: str
    license_url: str
    attribution: str

    @property
    def file_extension(self) -> str:
        return SUPPORTED_IMAGE_MIME_TYPES[self.mime_type]


def _request_bytes(url: str, *, data: bytes | None = None) -> bytes:
    request = Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, image/*",
        },
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_IMAGE_BYTES:
            raise ValueError("Remote image exceeds the 10 MB download limit")
        body = response.read(MAX_IMAGE_BYTES + 1)
    if len(body) > MAX_IMAGE_BYTES:
        raise ValueError("Remote response exceeds the 10 MB download limit")
    return body


def _request_json(url: str, *, data: bytes | None = None) -> dict[str, Any]:
    payload = json.loads(_request_bytes(url, data=data))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object from the open-data API")
    return payload


def _extract_restaurant_features(tags: dict[str, str]) -> tuple[str, ...]:
    features = []
    for key, label in OSM_FEATURE_TAGS.items():
        raw_value = tags.get(key)
        if not raw_value:
            continue
        value = raw_value.strip()
        normalized_value = value.casefold()
        if key == "wheelchair":
            if normalized_value not in {"yes", "limited"}:
                continue
            description = (
                "accesible"
                if normalized_value == "yes"
                else "accesibilidad limitada"
            )
        else:
            if normalized_value not in _POSITIVE_FEATURE_VALUES:
                continue
            description = _FEATURE_VALUE_LABELS[normalized_value]
        features.append(f"{label}: {description}")
    if any(feature.startswith("Comida kosher:") for feature in features):
        if certifier := tags.get("diet:kosher:certifier"):
            features.append(
                f"Certificador kosher: {' '.join(certifier.split())[:255]}"
            )
        if check_date := tags.get("check_date:diet:kosher"):
            features.append(
                f"Última revisión kosher: {' '.join(check_date.split())[:32]}"
            )
    return tuple(features)


def parse_madrid_restaurants(
    payload: dict[str, Any],
    *,
    limit: int | None = None,
) -> list[OSMRestaurant]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")

    restaurants = []
    for element in payload.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        commons_reference = tags.get("wikimedia_commons")
        if not name:
            continue

        center = element.get("center", element)
        latitude = center.get("lat")
        longitude = center.get("lon")
        if latitude is None or longitude is None:
            continue

        address = tags.get("addr:full")
        if not address:
            address_parts = [
                " ".join(
                    part
                    for part in (
                        tags.get("addr:street"),
                        tags.get("addr:housenumber"),
                    )
                    if part
                ),
                tags.get("addr:postcode"),
                tags.get("addr:city"),
            ]
            address = ", ".join(part for part in address_parts if part) or None

        restaurants.append(
            OSMRestaurant(
                osm_type=element["type"],
                osm_id=int(element["id"]),
                name=name[:255],
                city="Madrid",
                cuisine=tags.get("cuisine"),
                location=address[:512] if address else None,
                latitude=float(latitude),
                longitude=float(longitude),
                wikimedia_commons=commons_reference[:512] if commons_reference else None,
                features=_extract_restaurant_features(tags),
            )
        )
        if limit is not None and len(restaurants) == limit:
            break

    return restaurants


def fetch_madrid_restaurants(*, limit: int | None = None) -> list[OSMRestaurant]:
    if limit is not None and not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000")

    output_clause = f"out center {limit};" if limit is not None else "out center;"
    query = (
        '[out:json][timeout:60];'
        'area["name"="Madrid"]["boundary"="administrative"]'
        '["admin_level"="8"]->.city;'
        'nwr["amenity"="restaurant"](area.city);'
        f"{output_clause}"
    )
    request_data = urlencode({"data": query}).encode("utf-8")
    for attempt in range(3):
        try:
            payload = _request_json(OVERPASS_URL, data=request_data)
            break
        except HTTPError as error:
            if error.code not in {429, 502, 503, 504} or attempt == 2:
                raise
            time.sleep(2**attempt)
    return parse_madrid_restaurants(payload, limit=limit)


def _metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return value if isinstance(value, str) else ""


def _plain_text(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]*>", " ", value)).strip()


def _photo_from_page(page: dict[str, Any]) -> CommonsPhoto | None:
    image_infos = page.get("imageinfo", [])
    if not image_infos:
        return None

    info = image_infos[0]
    mime_type = info.get("mime")
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        return None

    metadata = info.get("extmetadata", {})
    license_name = _plain_text(_metadata_value(metadata, "LicenseShortName"))
    if license_name.casefold() not in SUPPORTED_LICENSES:
        return None

    license_url = _metadata_value(metadata, "LicenseUrl").strip()
    source_url = info.get("descriptionurl")
    image_url = info.get("thumburl")
    if not license_url or not source_url or not image_url:
        return None

    parsed_image_url = urlparse(image_url)
    if (
        parsed_image_url.scheme != "https"
        or parsed_image_url.hostname not in COMMONS_IMAGE_HOSTS
    ):
        return None

    artist = _plain_text(
        _metadata_value(metadata, "Artist") or _metadata_value(metadata, "Credit")
    )
    if license_name.casefold().startswith("cc by") and not artist:
        return None

    return CommonsPhoto(
        title=page.get("title", ""),
        image_url=image_url,
        source_url=source_url,
        mime_type=mime_type,
        license_name=license_name,
        license_url=license_url,
        attribution=artist,
    )


def parse_commons_photos(
    payload: dict[str, Any],
    *,
    limit: int,
) -> list[CommonsPhoto]:
    pages = payload.get("query", {}).get("pages", {})
    photos = []
    for page in pages.values():
        photo = _photo_from_page(page)
        if photo is not None:
            photos.append(photo)
            if len(photos) == limit:
                break
    return photos


def fetch_commons_photos(
    reference: str,
    *,
    limit: int = 3,
) -> list[CommonsPhoto]:
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")

    if reference.startswith("Category:"):
        parameters = {
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": reference,
            "gcmtype": "file",
            "gcmlimit": limit * 3,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 768,
            "format": "json",
        }
    elif reference.startswith("File:"):
        parameters = {
            "action": "query",
            "titles": reference,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": 768,
            "format": "json",
        }
    else:
        raise ValueError(
            "Only explicit Wikimedia Commons Category: or File: references "
            "are supported"
        )

    payload = _request_json(
        f"{COMMONS_API_URL}?{urlencode(parameters)}",
    )
    return parse_commons_photos(payload, limit=limit)


def download_commons_photo(photo: CommonsPhoto) -> bytes:
    parsed_url = urlparse(photo.image_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in COMMONS_IMAGE_HOSTS
    ):
        raise ValueError("Refusing to download an image outside Wikimedia Commons")
    return _request_bytes(photo.image_url)
