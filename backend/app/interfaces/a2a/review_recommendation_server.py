import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from a2a.helpers import new_data_part, new_task_from_user_message, new_text_message
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
    Message,
    Role,
    TaskState,
)
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError
from starlette.applications import Starlette

from app.agents.review_recommendation_agent import (
    ReviewRecommendationAgentError,
    run_review_recommendation_agent,
)
from app.agents.schemas import (
    ReviewRecommendationRequest,
    ReviewRecommendationResponse,
)
from app.infrastructure.observability import (
    configure_phoenix_tracing,
    shutdown_phoenix_tracing,
    traced_span,
)
from app.infrastructure.persistence.postgres.database import SessionLocal, engine

logger = logging.getLogger(__name__)

A2A_HOST = "127.0.0.1"
A2A_PORT = 8003
A2A_URL = f"http://{A2A_HOST}:{A2A_PORT}"
REVIEW_RECOMMENDATION_ARTIFACT = "review_recommendation_result"

_REVIEW_RECOMMENDATION_SKILL = AgentSkill(
    id="review_recommendation",
    name="Review and recommendation",
    description=(
        "Compares supplied restaurant candidates against explicit preferences "
        "using available candidate evidence, and reports document attribution "
        "limitations."
    ),
    tags=["restaurants", "comparison", "recommendation"],
    examples=[
        '{"query":"compare vegetarian restaurants","preferences":["vegetarian options"],"candidates":[{"name":"Casa Verde","city":"Madrid","cuisine":"vegetarian","location":"Calle Mayor 1","features":["Opciones vegetarianas: disponible"],"source_url":"https://www.openstreetmap.org/node/123","attribution":"© OpenStreetMap contributors","attribution_url":"https://www.openstreetmap.org/copyright"}]}'
    ],
    input_modes=["application/json"],
    output_modes=["application/json"],
)

AGENT_CARD = AgentCard(
    name="Review & Recommendation Agent",
    description=(
        "Compares restaurant candidates using supplied evidence. The indexed "
        "document corpus is not reliably linked to restaurants or review sources."
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
    default_input_modes=["application/json"],
    default_output_modes=["application/json"],
    skills=[_REVIEW_RECOMMENDATION_SKILL],
)


def _read_recommendation_request(message: Message) -> ReviewRecommendationRequest:
    if len(message.parts) != 1:
        raise ValueError("Exactly one JSON data part is required.")

    part = message.parts[0]
    if (
        part.WhichOneof("content") != "data"
        or part.media_type != "application/json"
    ):
        raise ValueError("The request must be an application/json data part.")

    payload = MessageToDict(part.data)
    if set(payload) != {"query", "preferences", "candidates"}:
        raise ValueError(
            "The request must contain query, preferences, and candidates only."
        )
    return ReviewRecommendationRequest.model_validate(payload, strict=True)


def _run_recommendation(
    request: ReviewRecommendationRequest,
) -> ReviewRecommendationResponse:
    with SessionLocal() as session:
        return run_review_recommendation_agent(request, session)


class ReviewRecommendationExecutor(AgentExecutor):
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        incoming_message = context.message
        if incoming_message is None or incoming_message.role != Role.ROLE_USER:
            raise ValueError("A user JSON message is required.")

        task = context.current_task
        if task is None:
            task = new_task_from_user_message(incoming_message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )

        with traced_span("a2a.review_recommendation_task") as span:
            await updater.update_status(state=TaskState.TASK_STATE_WORKING)
            try:
                request = _read_recommendation_request(incoming_message)
            except (TypeError, ValueError, ValidationError):
                span.set_attribute("a2a.task.status", "invalid_input")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "Provide one application/json request with a query, "
                        "preferences, and restaurant candidates."
                    ),
                )
                return

            try:
                response = await asyncio.to_thread(_run_recommendation, request)
            except Exception as error:
                logger.error(
                    "A2A review recommendation failed with %s",
                    type(error).__name__,
                )
                span.set_attribute("a2a.task.status", "failed")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "The Review & Recommendation Agent could not complete "
                        "the request."
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
                name=REVIEW_RECOMMENDATION_ARTIFACT,
            )
            span.set_attribute("a2a.task.status", "completed")
            span.set_attribute(
                "recommendation.candidate_count",
                len(response.comparisons),
            )
            await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        del context, event_queue
        raise NotImplementedError(
            "Review recommendation tasks cannot be cancelled."
        )


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
                logger.warning(
                    "Phoenix spans were not fully exported at A2A shutdown"
                )
        finally:
            engine.dispose()


_REQUEST_HANDLER = DefaultRequestHandler(
    agent_executor=ReviewRecommendationExecutor(),
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
