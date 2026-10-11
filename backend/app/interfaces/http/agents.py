from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agents.restaurant_search_agent import (
    RestaurantSearchAgentError,
    run_restaurant_search_agent,
)
from app.agents.schemas import (
    RestaurantSearchAgentRequest,
    RestaurantSearchAgentResponse,
)
from app.infrastructure.persistence.postgres.database import get_db

router = APIRouter()


@router.post(
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
