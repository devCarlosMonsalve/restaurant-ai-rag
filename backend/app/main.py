from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.restaurant_search_agent import (
    RestaurantSearchAgentError,
    run_restaurant_search_agent,
)
from app.agents.schemas import (
    RestaurantSearchAgentRequest,
    RestaurantSearchAgentResponse,
)
from app.application.knowledge import (
    answer_from_documents as answer_documents_use_case,
    search_documents as search_documents_use_case,
)
from app.application.restaurant_discovery import (
    search_restaurant_photos as search_restaurant_photos_use_case,
    search_restaurants as search_restaurants_use_case,
)
from app.database import get_db
from app.models.image_embedding import ImageEmbedding
from app.models.osm_place import OsmPlace
from app.models.restaurant import Restaurant
from app.schemas import (
    DocumentSearchRequest,
    DocumentSearchResult,
    ImageSearchRequest,
    ImageSearchResult,
    OsmPlaceRead,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
    RagAnswerResponse,
    RagQuestionRequest,
    RestaurantCreate,
    RestaurantRead,
)
app = FastAPI(title="Restaurant AI API")

BACKEND_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BACKEND_DIR / "static"
COMMONS_IMAGES_DIR = BACKEND_DIR / "data" / "images" / "commons"

OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_ATTRIBUTION_URL = "https://www.openstreetmap.org/copyright"


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/restaurants",
    response_model=RestaurantRead,
    status_code=status.HTTP_201_CREATED,
    tags=["restaurants"],
)
def create_restaurant(
    restaurant_data: RestaurantCreate,
    db: Session = Depends(get_db),
) -> Restaurant:
    restaurant = Restaurant(**restaurant_data.model_dump())
    db.add(restaurant)
    db.commit()
    db.refresh(restaurant)
    return restaurant


@app.get("/restaurants", response_model=list[RestaurantRead], tags=["restaurants"])
def list_restaurants(db: Session = Depends(get_db)) -> list[Restaurant]:
    return list(db.scalars(select(Restaurant).order_by(Restaurant.name)).all())


@app.get("/restaurants/osm", response_model=list[OsmPlaceRead], tags=["restaurants"])
def list_osm_places(db: Session = Depends(get_db)) -> list[OsmPlaceRead]:
    has_photos = (
        select(ImageEmbedding.id)
        .where(ImageEmbedding.osm_place_id == OsmPlace.id)
        .exists()
    )
    places = db.execute(
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


@app.post(
    "/restaurants/osm/search",
    response_model=OsmRestaurantSearchResponse,
    tags=["restaurants"],
)
def search_osm_restaurants(
    request: OsmRestaurantSearchRequest,
    db: Session = Depends(get_db),
) -> OsmRestaurantSearchResponse:
    return search_restaurants_use_case(request, db)


@app.post(
    "/documents/search",
    response_model=list[DocumentSearchResult],
    tags=["documents"],
)
def search_documents(
    request: DocumentSearchRequest,
    db: Session = Depends(get_db),
) -> list[DocumentSearchResult]:
    return search_documents_use_case(request, db)


@app.post(
    "/documents/ask",
    response_model=RagAnswerResponse,
    tags=["documents"],
)
def ask_documents(
    request: RagQuestionRequest,
    db: Session = Depends(get_db),
) -> RagAnswerResponse:
    return answer_documents_use_case(request, db)


@app.post(
    "/agents/restaurant-search",
    response_model=RestaurantSearchAgentResponse,
    tags=["agents"],
)
def restaurant_search_agent(
    request: RestaurantSearchAgentRequest,
    db: Session = Depends(get_db),
) -> RestaurantSearchAgentResponse:
    try:
        return run_restaurant_search_agent(request.query, db)
    except RestaurantSearchAgentError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The restaurant search Agent could not complete the request.",
        ) from error


@app.post(
    "/images/search",
    response_model=list[ImageSearchResult],
    tags=["images"],
)
def search_images(
    request: ImageSearchRequest,
    db: Session = Depends(get_db),
) -> list[ImageSearchResult]:
    return search_restaurant_photos_use_case(request, db)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/images/files/{filename}", include_in_schema=False)
def get_commons_image(filename: str) -> FileResponse:
    image_path = (COMMONS_IMAGES_DIR / filename).resolve()
    if image_path.parent != COMMONS_IMAGES_DIR.resolve() or not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(image_path)
