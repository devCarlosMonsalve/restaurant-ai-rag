from typing import Protocol

from app.schemas import OsmPlaceRead, RestaurantCreate, RestaurantRead


class RestaurantCatalog(Protocol):
    def create_restaurant(self, data: RestaurantCreate) -> RestaurantRead: ...

    def list_restaurants(self) -> list[RestaurantRead]: ...

    def list_osm_places(self) -> list[OsmPlaceRead]: ...


def create_restaurant(
    data: RestaurantCreate,
    catalog: RestaurantCatalog,
) -> RestaurantRead:
    return catalog.create_restaurant(data)


def list_restaurants(catalog: RestaurantCatalog) -> list[RestaurantRead]:
    return catalog.list_restaurants()


def list_osm_places(catalog: RestaurantCatalog) -> list[OsmPlaceRead]:
    return catalog.list_osm_places()
