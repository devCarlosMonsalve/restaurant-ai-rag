import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from a2a.helpers import (
    get_message_text,
    new_data_part,
    new_task_from_user_message,
    new_text_message,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
    TaskState,
)
from pydantic import ValidationError
from starlette.applications import Starlette

from app.agents.restaurant_search_agent import (
    RestaurantSearchAgentError,
    run_restaurant_search_agent,
)
from app.agents.schemas import (
    RestaurantSearchAgentRequest,
    RestaurantSearchAgentResponse,
)
from app.infrastructure.persistence.postgres.database import SessionLocal, engine
from app.infrastructure.observability import (
    configure_phoenix_tracing,
    shutdown_phoenix_tracing,
    traced_span,
)

logger = logging.getLogger(__name__)

A2A_HOST = "127.0.0.1"
A2A_PORT = 8001
A2A_URL = f"http://{A2A_HOST}:{A2A_PORT}"

_RESTAURANT_SEARCH_SKILL = AgentSkill(
    id="restaurant_discovery",
    name="Restaurant discovery",
    description=(
        "Find indexed restaurant candidates and, when requested, their "
        "associated photos with source attribution."
    ),
    tags=["restaurants", "local-discovery"],
    examples=["Find vegetarian restaurants in Madrid"],
    input_modes=["text/plain"],
    output_modes=["application/json"],
)

AGENT_CARD = AgentCard(
    name="Restaurant Discovery Agent",
    description=(
        "Discovers indexed restaurant candidates and associated photo evidence. "
        "It does not plan itineraries, make reservations, or verify live details."
    ),
    version="1.0.0",
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url=A2A_URL,
            protocol_version="1.0",
        )
    ],
    capabilities=AgentCapabilities(streaming=False),
    default_input_modes=["text/plain"],
    default_output_modes=["application/json"],
    skills=[_RESTAURANT_SEARCH_SKILL],
)


def _run_restaurant_search(query: str) -> RestaurantSearchAgentResponse:
    with SessionLocal() as session:
        return run_restaurant_search_agent(query, session)


class RestaurantDiscoveryExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        incoming_message = context.message
        if incoming_message is None or incoming_message.role != Role.ROLE_USER:
            raise ValueError("A user text message is required")

        task = context.current_task
        if task is None:
            task = new_task_from_user_message(incoming_message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        with traced_span("a2a.restaurant_discovery_task") as span:
            await updater.update_status(state=TaskState.TASK_STATE_WORKING)
            try:
                request = RestaurantSearchAgentRequest(
                    query=get_message_text(incoming_message),
                )
            except ValidationError:
                span.set_attribute("a2a.task.status", "invalid_input")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "Provide a non-empty text request of at most 2,000 characters."
                    ),
                )
                return

            try:
                response = await asyncio.to_thread(
                    _run_restaurant_search,
                    request.query,
                )
            except RestaurantSearchAgentError as error:
                logger.error(
                    "A2A restaurant discovery failed with %s",
                    type(error).__name__,
                )
                span.set_attribute("a2a.task.status", "failed")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "The restaurant discovery Agent could not complete the request."
                    ),
                )
                return
            except Exception as error:
                logger.error(
                    "A2A restaurant discovery failed with %s",
                    type(error).__name__,
                )
                span.set_attribute("a2a.task.status", "failed")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "The restaurant discovery Agent could not complete the request."
                    ),
                )
                return

            await updater.add_artifact(
                parts=[
                    new_data_part(
                        response.model_dump(mode="json"),
                        media_type="application/json",
                    )
                ],
                name="restaurant_discovery_result",
            )
            span.set_attribute("a2a.task.status", "completed")
            span.set_attribute("agent.photo_count", len(response.photos))
            await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        del context, event_queue
        raise NotImplementedError("Restaurant discovery tasks cannot be cancelled.")


@asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
    del _app
    tracer_provider = None
    try:
        tracer_provider = configure_phoenix_tracing()
        yield
    finally:
        try:
            if tracer_provider is not None and not shutdown_phoenix_tracing():
                logger.warning("Phoenix spans were not fully exported at A2A shutdown")
        finally:
            engine.dispose()


_REQUEST_HANDLER = DefaultRequestHandler(
    agent_executor=RestaurantDiscoveryExecutor(),
    task_store=InMemoryTaskStore(),
    agent_card=AGENT_CARD,
)

app = Starlette(
    routes=[
        *create_agent_card_routes(AGENT_CARD),
        *create_jsonrpc_routes(_REQUEST_HANDLER, "/"),
    ],
    lifespan=lifespan,
)


def main() -> None:
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)


if __name__ == "__main__":
    main()
