from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.osm_place import OsmPlace
from app.open_data_sources import OSMRestaurant
from app.ingestion import upsert_osm_place


def test_osm_place_upsert_updates_existing_place(
    db_session: Session,
) -> None:
    first_version = OSMRestaurant(
        osm_type="node",
        osm_id=123,
        name="Old Name",
        city="Madrid",
        cuisine=None,
        location=None,
        latitude=40.4,
        longitude=-3.7,
        wikimedia_commons="Category:Example",
    )
    updated_version = OSMRestaurant(
        osm_type="node",
        osm_id=123,
        name="Updated Name",
        city="Madrid",
        cuisine="italian",
        location="Calle Mayor 10",
        latitude=40.4,
        longitude=-3.7,
        wikimedia_commons="Category:Example",
        features=("Mesas al aire libre: disponible",),
    )

    first_place = upsert_osm_place(first_version, db_session)
    first_place.embedding = [0.1] * 768
    updated_place = upsert_osm_place(updated_version, db_session)
    stored_places = db_session.scalars(select(OsmPlace)).all()

    assert updated_place.id == first_place.id
    assert updated_place.name == "Updated Name"
    assert updated_place.cuisine == "italian"
    assert updated_place.features == ["Mesas al aire libre: disponible"]
    assert updated_place.embedding is None
    assert len(stored_places) == 1
