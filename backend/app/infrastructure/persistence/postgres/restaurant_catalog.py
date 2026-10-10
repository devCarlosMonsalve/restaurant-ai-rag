from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.models.restaurant import Restaurant
from app.schemas import OsmPlaceRead, RestaurantCreate, RestaurantRead


OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_ATTRIBUTION_URL = "https://www.openstreetmap.org/copyright"


class PostgresRestaurantCatalog:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_restaurant(self, data: RestaurantCreate) -> RestaurantRead:
        restaurant = Restaurant(**data.model_dump())
        self.session.add(restaurant)
        self.session.commit()
        self.session.refresh(restaurant)
        return RestaurantRead.model_validate(restaurant)

    def list_restaurants(self) -> list[RestaurantRead]:
        restaurants = self.session.scalars(
            select(Restaurant).order_by(Restaurant.name)
        ).all()
        return [RestaurantRead.model_validate(restaurant) for restaurant in restaurants]

    def list_osm_places(self) -> list[OsmPlaceRead]:
        has_photos = (
            select(ImageEmbedding.id)
            .where(ImageEmbedding.osm_place_id == OsmPlace.id)
            .exists()
        )
        places = self.session.execute(
            select(OsmPlace, has_photos.label("has_photos")).order_by(OsmPlace.name)
        ).all()
        return [
            OsmPlaceRead(
                id=place.id,
                osm_type=place.osm_type,
                osm_id=place.osm_id,
                name=place.name,
                city=place.city,
                cuisine=place.cuisine,
                location=place.location,
                latitude=place.latitude,
                longitude=place.longitude,
                features=place.features or [],
                wikimedia_commons=place.wikimedia_commons,
                source_url=place.source_url,
                attribution=OSM_ATTRIBUTION,
                attribution_url=OSM_ATTRIBUTION_URL,
                has_photos=place_has_photos,
            )
            for place, place_has_photos in places
        ]
