from pydantic import BaseModel, ConfigDict, Field


class RestaurantSearchAgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)


class RestaurantSearchAgentResponse(BaseModel):
    answer: str
