from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.infrastructure.persistence.postgres.database import get_db
from app.interfaces.http.schemas import ImageSearchResult
from app.restaurant_discovery.application.contracts import ImageSearchRequest
from app.restaurant_discovery.application.service import (
    search_restaurant_photos as search_restaurant_photos_use_case,
)
from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)

router = APIRouter()

BACKEND_DIR = Path(__file__).resolve().parents[3]
STATIC_DIR = BACKEND_DIR / "static"
COMMONS_IMAGES_DIR = BACKEND_DIR / "data" / "images" / "commons"


@router.post(
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


@router.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/images/files/{filename}", include_in_schema=False)
def get_commons_image(filename: str) -> FileResponse:
    image_path = (COMMONS_IMAGES_DIR / filename).resolve()
    if image_path.parent != COMMONS_IMAGES_DIR.resolve() or not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(image_path)
