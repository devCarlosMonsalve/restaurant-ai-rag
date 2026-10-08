from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.osm_place import OsmPlace
from app.open_data_sources import OSMRestaurant


def upsert_osm_place(
    restaurant: OSMRestaurant,
    session: Session,
) -> OsmPlace:
    place = session.scalar(
        select(OsmPlace).where(
            OsmPlace.osm_type == restaurant.osm_type,
            OsmPlace.osm_id == restaurant.osm_id,
        )
    )
    if place is None:
        place = OsmPlace(
            osm_type=restaurant.osm_type,
            osm_id=restaurant.osm_id,
            name=restaurant.name,
            city=restaurant.city,
            cuisine=restaurant.cuisine,
            location=restaurant.location,
            latitude=restaurant.latitude,
            longitude=restaurant.longitude,
            features=list(restaurant.features),
            wikimedia_commons=restaurant.wikimedia_commons,
            source_url=restaurant.source_url,
        )
        session.add(place)
    else:
        searchable_metadata_changed = any(
            getattr(place, field) != getattr(restaurant, field)
            for field in ("name", "city", "cuisine", "location")
        )
        searchable_metadata_changed = searchable_metadata_changed or (
            (place.features or []) != list(restaurant.features)
        )
        place.name = restaurant.name
        place.city = restaurant.city
        place.cuisine = restaurant.cuisine
        place.location = restaurant.location
        place.latitude = restaurant.latitude
        place.longitude = restaurant.longitude
        place.features = list(restaurant.features)
        place.wikimedia_commons = restaurant.wikimedia_commons
        place.source_url = restaurant.source_url
        if searchable_metadata_changed:
            place.embedding = None

    session.flush()
    return place
