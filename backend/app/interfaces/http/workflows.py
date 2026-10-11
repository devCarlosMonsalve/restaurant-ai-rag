from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.infrastructure.persistence.postgres.database import get_db
from app.workflows.restaurant_photo_search import (
    RestaurantPhotoWorkflowError,
    run_restaurant_photo_workflow,
)
from app.workflows.schemas import (
    RestaurantPhotoWorkflowRequest,
    RestaurantPhotoWorkflowResponse,
)

router = APIRouter()


@router.post(
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
