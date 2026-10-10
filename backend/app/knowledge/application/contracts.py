from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentSearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class DocumentSearchResult(BaseModel):
    document_id: UUID
    source_name: str
    chunk_index: int
    content: str
    similarity: float


class RagQuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RagSource(BaseModel):
    document_id: UUID
    source_name: str
    chunk_index: int
    similarity: float


class RagAnswerResponse(BaseModel):
    answer: str
    sources: list[RagSource]
