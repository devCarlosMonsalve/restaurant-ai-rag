from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RestaurantCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1)
    cuisine: str = Field(min_length=1, max_length=100)
    price_range: str = Field(min_length=1, max_length=10)
    location: str = Field(min_length=1, max_length=255)


class RestaurantRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str
    cuisine: str
    price_range: str
    location: str
