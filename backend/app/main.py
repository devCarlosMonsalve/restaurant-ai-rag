from fastapi import Depends, FastAPI, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.document_search import search_document_chunks
from app.image_search import search_images_by_text
from app.models.osm_place import OsmPlace
from app.models.restaurant import Restaurant
from app.schemas import (
    DocumentSearchRequest,
    DocumentSearchResult,
    ImageSearchRequest,
    ImageSearchResult,
    OsmPlaceRead,
    RagAnswerResponse,
    RagQuestionRequest,
    RestaurantCreate,
    RestaurantRead,
)
from app.rag import answer_with_rag

app = FastAPI(title="Restaurant AI API")

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
    places = db.scalars(select(OsmPlace).order_by(OsmPlace.name)).all()
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
            wikimedia_commons=place.wikimedia_commons,
            source_url=place.source_url,
            attribution=OSM_ATTRIBUTION,
            attribution_url=OSM_ATTRIBUTION_URL,
        )
        for place in places
    ]


@app.post(
    "/documents/search",
    response_model=list[DocumentSearchResult],
    tags=["documents"],
)
def search_documents(
    request: DocumentSearchRequest,
    db: Session = Depends(get_db),
) -> list[DocumentSearchResult]:
    return search_document_chunks(request.query, db, top_k=request.top_k)


@app.post(
    "/documents/ask",
    response_model=RagAnswerResponse,
    tags=["documents"],
)
def ask_documents(
    request: RagQuestionRequest,
    db: Session = Depends(get_db),
) -> RagAnswerResponse:
    return answer_with_rag(request, db)


@app.post(
    "/images/search",
    response_model=list[ImageSearchResult],
    tags=["images"],
)
def search_images(
    request: ImageSearchRequest,
    db: Session = Depends(get_db),
) -> list[ImageSearchResult]:
    return search_images_by_text(request.query, db, top_k=request.top_k)
