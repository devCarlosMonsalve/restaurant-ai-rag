from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.application.restaurant_catalog import (
    RestaurantCatalog,
    create_restaurant as create_restaurant_use_case,
    list_osm_places as list_osm_places_use_case,
    list_restaurants as list_restaurants_use_case,
)
from app.infrastructure.persistence.postgres.database import get_db
from app.infrastructure.persistence.postgres.restaurant_catalog import (
    PostgresRestaurantCatalog,
)
from app.interfaces.http.schemas import (
    OsmPlaceRead,
    RestaurantCreate,
    RestaurantRead,
)
from app.restaurant_discovery.application.contracts import (
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
)
from app.restaurant_discovery.application.service import (
    search_restaurants as search_restaurants_use_case,
)
from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)

router = APIRouter()


def get_restaurant_catalog(
    db: Session = Depends(get_db),
) -> RestaurantCatalog:
    return PostgresRestaurantCatalog(db)


@router.post(
    "/restaurants",
    response_model=RestaurantRead,
    status_code=status.HTTP_201_CREATED,
    tags=["restaurants"],
)
def create_restaurant(
    restaurant_data: RestaurantCreate,
    catalog: RestaurantCatalog = Depends(get_restaurant_catalog),
) -> RestaurantRead:
    return create_restaurant_use_case(restaurant_data, catalog)


@router.get("/restaurants", response_model=list[RestaurantRead], tags=["restaurants"])
def list_restaurants(
    catalog: RestaurantCatalog = Depends(get_restaurant_catalog),
) -> list[RestaurantRead]:
    return list_restaurants_use_case(catalog)


@router.get("/restaurants/osm", response_model=list[OsmPlaceRead], tags=["restaurants"])
def list_osm_places(
    catalog: RestaurantCatalog = Depends(get_restaurant_catalog),
) -> list[OsmPlaceRead]:
    return list_osm_places_use_case(catalog)


@router.post(
    "/restaurants/osm/search",
    response_model=OsmRestaurantSearchResponse,
    tags=["restaurants"],
)
def search_osm_restaurants(
    request: OsmRestaurantSearchRequest,
    db: Session = Depends(get_db),
) -> OsmRestaurantSearchResponse:
    return search_restaurants_use_case(
        request,
        PostgresRestaurantDiscoveryAdapter(db),
    )
