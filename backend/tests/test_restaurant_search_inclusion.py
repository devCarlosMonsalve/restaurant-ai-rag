from pathlib import Path

import pytest
from sqlalchemy import event, literal
from sqlalchemy.orm import Session

from app.image_embeddings import IMAGE_EMBEDDING_DIMENSIONS
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.restaurant_search import search_osm_places_by_text


def test_photo_workflow_option_includes_indexed_places_without_changing_defaults(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distance_comparator = type(OsmPlace.embedding.cosine_distance.__self__)
    monkeypatch.setattr(
        distance_comparator,
        "cosine_distance",
        lambda self, query: literal(0.1),
    )
    monkeypatch.setattr(
        "app.restaurant_search.embed_search_query",
        lambda _: [0.0] * 768,
    )

    mexican_with_photo = _add_place(
        db_session,
        osm_id=101,
        name="Mexican With Photo",
        city="Madrid",
        cuisine="mexican",
        has_photo=True,
    )
    mexican_without_photo = _add_place(
        db_session,
        osm_id=102,
        name="Mexican Without Photo",
        city="Madrid",
        cuisine="mexican",
        has_photo=False,
    )
    _add_place(
        db_session,
        osm_id=103,
        name="Barcelona Mexican With Photo",
        city="Barcelona",
        cuisine="mexican",
        has_photo=True,
    )
    _add_place(
        db_session,
        osm_id=104,
        name="Madrid Italian With Photo",
        city="Madrid",
        cuisine="italian",
        has_photo=True,
    )
    db_session.flush()

    statements: list[str] = []

    def capture_statement(
        connection,
        cursor,
        statement,
        parameters,
        context,
        executemany,
    ):
        statements.append(statement.lower())

    connection = db_session.connection()
    event.listen(connection, "before_cursor_execute", capture_statement)
    try:
        default_results = search_osm_places_by_text(
            "mexican restaurants",
            db_session,
            top_k=10,
            city="Madrid",
            cuisine="mexican",
        )
        default_statement = statements[-1]
        statements.clear()
        workflow_results = search_osm_places_by_text(
            "mexican restaurants",
            db_session,
            top_k=10,
            city="Madrid",
            cuisine="mexican",
            include_places_with_photos=True,
        )
        workflow_statement = statements[-1]
    finally:
        event.remove(connection, "before_cursor_execute", capture_statement)

    assert {place.id for place in default_results.results} == {
        mexican_without_photo.id
    }
    assert {place.id for place in workflow_results.results} == {
        mexican_with_photo.id,
        mexican_without_photo.id,
    }
    assert "from osm_places" in default_statement
    assert "from osm_places" in workflow_statement
    assert "exists" in default_statement
    assert "exists" not in workflow_statement


def _add_place(
    session: Session,
    *,
    osm_id: int,
    name: str,
    city: str,
    cuisine: str,
    has_photo: bool,
) -> OsmPlace:
    place = OsmPlace(
        osm_type="node",
        osm_id=osm_id,
        name=name,
        city=city,
        cuisine=cuisine,
        location=None,
        latitude=40.4,
        longitude=-3.7,
        embedding=[0.0] * 768,
        features=[],
        wikimedia_commons=None,
        source_url=f"https://www.openstreetmap.org/node/{osm_id}",
    )
    session.add(place)
    session.flush()

    if has_photo:
        session.add(
            ImageEmbedding(
                source_name=f"{name}.jpg",
                image_path=str(Path("C:/images") / f"{osm_id}.jpg"),
                embedding=[0.0] * IMAGE_EMBEDDING_DIMENSIONS,
                osm_place_id=place.id,
            )
        )
    return place
