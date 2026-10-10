from sqlalchemy import func, or_
from sqlalchemy.sql.elements import ColumnElement

from app.models.osm_place import OsmPlace


def cuisine_filter(cuisine: str) -> ColumnElement[bool]:
    token = cuisine.strip().lower()
    escaped_token = (
        token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    normalized_cuisine = func.lower(OsmPlace.cuisine)
    return or_(
        normalized_cuisine == token,
        normalized_cuisine.like(f"{escaped_token};%", escape="\\"),
        normalized_cuisine.like(f"%;{escaped_token};%", escape="\\"),
        normalized_cuisine.like(f"%;{escaped_token}", escape="\\"),
    )
