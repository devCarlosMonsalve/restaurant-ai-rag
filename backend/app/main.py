import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.observability import (
    configure_phoenix_tracing,
    shutdown_phoenix_tracing,
)
from app.interfaces.http.agents import router as agents_router
from app.interfaces.http.images import router as images_router
from app.interfaces.http.knowledge import router as knowledge_router
from app.interfaces.http.restaurants import router as restaurants_router
from app.interfaces.http.workflows import router as workflows_router

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


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(restaurants_router)
app.include_router(knowledge_router)
app.include_router(agents_router)
app.include_router(workflows_router)
app.include_router(images_router)
