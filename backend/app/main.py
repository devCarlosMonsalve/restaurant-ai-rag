import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.agents.restaurant_search_agent import (
    RestaurantSearchAgentError,
    run_restaurant_search_agent,
)
from app.agents.schemas import (
    RestaurantSearchAgentRequest,
    RestaurantSearchAgentResponse,
)
from app.application.document_answer import (
    answer_from_documents_with_photos as answer_documents_use_case,
)
from app.restaurant_discovery.application.service import (
    search_restaurant_photos as search_restaurant_photos_use_case,
    search_restaurants as search_restaurants_use_case,
)
from app.infrastructure.persistence.postgres.database import get_db
from app.application.restaurant_catalog import (
    RestaurantCatalog,
    create_restaurant as create_restaurant_use_case,
    list_osm_places as list_osm_places_use_case,
    list_restaurants as list_restaurants_use_case,
)
from app.infrastructure.persistence.postgres.restaurant_catalog import (
    PostgresRestaurantCatalog,
)
from app.knowledge.application.search_documents import (
    search_documents as search_documents_use_case,
)
from app.knowledge.infrastructure.generation.answer_chain import (
    generate_grounded_answer,
)
from app.knowledge.infrastructure.postgres.retriever import (
    PostgresDocumentRetriever,
)
from app.knowledge.application.contracts import (
    DocumentSearchRequest,
    DocumentSearchResult,
)
from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)
from app.restaurant_discovery.application.contracts import (
    ImageSearchRequest,
    ImageSearchResult,
    OsmRestaurantSearchRequest,
    OsmRestaurantSearchResponse,
)
from app.infrastructure.observability import (
    configure_phoenix_tracing,
    shutdown_phoenix_tracing,
)
from app.interfaces.http.schemas import (
    OsmPlaceRead,
    RagAnswerWithPhotosResponse,
    RagQuestionWithPhotosRequest,
    RestaurantCreate,
    RestaurantRead,
)
from app.workflows.restaurant_photo_search import (
    RestaurantPhotoWorkflowError,
    run_restaurant_photo_workflow,
)
from app.workflows.schemas import (
    RestaurantPhotoWorkflowRequest,
    RestaurantPhotoWorkflowResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    del _app
    tracer_provider = configure_phoenix_tracing()
    try:
        yield
    finally:
        if tracer_provider is not None and not shutdown_phoenix_tracing():
            logger.warning("Phoenix spans were not fully exported during shutdown")


app = FastAPI(title="Restaurant AI API", lifespan=lifespan)

BACKEND_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BACKEND_DIR / "static"
COMMONS_IMAGES_DIR = BACKEND_DIR / "data" / "images" / "commons"



@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


def get_restaurant_catalog(
    db: Session = Depends(get_db),
) -> RestaurantCatalog:
    return PostgresRestaurantCatalog(db)


@app.post(
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


@app.get("/restaurants", response_model=list[RestaurantRead], tags=["restaurants"])
def list_restaurants(
    catalog: RestaurantCatalog = Depends(get_restaurant_catalog),
) -> list[RestaurantRead]:
    return list_restaurants_use_case(catalog)


@app.get("/restaurants/osm", response_model=list[OsmPlaceRead], tags=["restaurants"])
def list_osm_places(
    catalog: RestaurantCatalog = Depends(get_restaurant_catalog),
) -> list[OsmPlaceRead]:
    return list_osm_places_use_case(catalog)


@app.post(
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


@app.post(
    "/documents/search",
    response_model=list[DocumentSearchResult],
    tags=["documents"],
)
def search_documents(
    request: DocumentSearchRequest,
    db: Session = Depends(get_db),
) -> list[DocumentSearchResult]:
    return search_documents_use_case(request, PostgresDocumentRetriever(db))


@app.post(
    "/documents/ask",
    response_model=RagAnswerWithPhotosResponse,
    tags=["documents"],
)
def ask_documents(
    request: RagQuestionWithPhotosRequest,
    db: Session = Depends(get_db),
) -> RagAnswerWithPhotosResponse:
    return answer_documents_use_case(
        request,
        PostgresDocumentRetriever(db),
        generate_grounded_answer,
        PostgresRestaurantDiscoveryAdapter(db),
    )


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
    "/workflows/restaurant-photo-search",
    response_model=RestaurantPhotoWorkflowResponse,
    tags=["workflows"],
)
def restaurant_photo_search_workflow(
    request: RestaurantPhotoWorkflowRequest,
    db: Session = Depends(get_db),
) -> RestaurantPhotoWorkflowResponse:
    try:
        return run_restaurant_photo_workflow(request, db)
    except RestaurantPhotoWorkflowError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The restaurant photo workflow could not complete the request.",
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
    return search_restaurant_photos_use_case(
        request,
        PostgresRestaurantDiscoveryAdapter(db),
    )


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/images/files/{filename}", include_in_schema=False)
def get_commons_image(filename: str) -> FileResponse:
    image_path = (COMMONS_IMAGES_DIR / filename).resolve()
    if image_path.parent != COMMONS_IMAGES_DIR.resolve() or not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(image_path)
