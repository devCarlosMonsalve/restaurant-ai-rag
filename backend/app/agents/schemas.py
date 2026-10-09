from pydantic import BaseModel, ConfigDict, Field


class RestaurantSearchAgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)


class RestaurantSearchAgentPhoto(BaseModel):
    image_url: str
    source_url: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    restaurant_name: str | None = None
    restaurant_location: str | None = None
    restaurant_cuisine: str | None = None
    restaurant_source_url: str | None = None
    restaurant_attribution: str | None = None
    restaurant_attribution_url: str | None = None


class RestaurantSearchAgentCandidate(BaseModel):
    name: str
    city: str
    cuisine: str | None
    location: str | None
    latitude: float | None
    longitude: float | None
    features: list[str] = Field(default_factory=list)
    source_url: str
    attribution: str
    attribution_url: str
    similarity: float


class RestaurantSearchAgentResponse(BaseModel):
    answer: str
    photos: list[RestaurantSearchAgentPhoto] = Field(default_factory=list)
    restaurants: list[RestaurantSearchAgentCandidate] = Field(default_factory=list)
