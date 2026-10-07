from fastapi import Depends, FastAPI, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.restaurant import Restaurant
from app.schemas import RestaurantCreate, RestaurantRead

app = FastAPI(title="Restaurant AI API")


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
