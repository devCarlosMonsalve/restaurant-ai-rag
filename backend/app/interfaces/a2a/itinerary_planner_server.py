import logging
import math
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

from app.itinerary_planning.application.planner import (
    ItineraryPlanningRequest,
    plan_itinerary_dining,
)
from app.itinerary_planning.application.ports import RestaurantEvidenceProvider
from app.itinerary_planning.infrastructure.a2a.restaurant_discovery_client import (
    A2ARestaurantEvidenceProvider,
)
from app.infrastructure.observability import (
    configure_phoenix_tracing,
    shutdown_phoenix_tracing,
    traced_span,
)

logger = logging.getLogger(__name__)

A2A_HOST = "127.0.0.1"
A2A_PORT = 8002
A2A_URL = f"http://{A2A_HOST}:{A2A_PORT}"
ITINERARY_DRAFT_ARTIFACT = "itinerary_dining_draft"

_ITINERARY_PLANNING_SKILL = AgentSkill(
    id="itinerary_dining_planning",
    name="Itinerary dining planning",
    description=(
        "Creates a deterministic dining draft from restaurant evidence returned "
        "by the Restaurant Discovery Agent."
    ),
    tags=["itinerary", "restaurants", "local-discovery"],
    examples=['{"query":"vegetarian restaurants in Madrid","day_count":3}'],
    input_modes=["application/json"],
    output_modes=["application/json"],
)

AGENT_CARD = AgentCard(
    name="Itinerary Planner Agent",
    description=(
        "Delegates restaurant discovery to a separate A2A Agent and returns a "
        "dining draft. It does not infer routes, opening hours, availability, "
        "or reservations."
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
    skills=[_ITINERARY_PLANNING_SKILL],
)


def _read_planning_request(message: Message) -> ItineraryPlanningRequest:
    if len(message.parts) != 1:
        raise ValueError("Exactly one JSON data part is required.")

    part = message.parts[0]
    if (
        part.WhichOneof("content") != "data"
        or part.media_type != "application/json"
    ):
        raise ValueError("The request must be an application/json data part.")

    payload = MessageToDict(part.data)
    if set(payload) != {"query", "day_count"}:
        raise ValueError("The request must contain query and day_count only.")

    day_count = payload["day_count"]
    if isinstance(day_count, bool) or not isinstance(day_count, (int, float)):
        raise ValueError("day_count must be a positive whole number.")
    if isinstance(day_count, float) and (
        not math.isfinite(day_count) or not day_count.is_integer()
    ):
        raise ValueError("day_count must be a positive whole number.")

    return ItineraryPlanningRequest.model_validate(
        {
            "query": payload["query"],
            "day_count": int(day_count),
        },
        strict=True,
    )


class ItineraryPlannerExecutor(AgentExecutor):
    def __init__(self, evidence_provider: RestaurantEvidenceProvider) -> None:
        self._evidence_provider = evidence_provider

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

        with traced_span("a2a.itinerary_planning_task") as span:
            await updater.update_status(state=TaskState.TASK_STATE_WORKING)
            try:
                request = _read_planning_request(incoming_message)
            except (TypeError, ValueError, ValidationError):
                span.set_attribute("a2a.task.status", "invalid_input")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "Provide one application/json request with a non-empty "
                        "query and a positive day_count."
                    ),
                )
                return

            try:
                draft = await plan_itinerary_dining(
                    request.query,
                    request.day_count,
                    evidence_provider=self._evidence_provider,
                )
            except Exception as error:
                logger.error(
                    "A2A itinerary planning failed with %s",
                    type(error).__name__,
                )
                span.set_attribute("a2a.task.status", "failed")
                await updater.update_status(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_text_message(
                        "The Itinerary Planner Agent could not complete the request."
                    ),
                )
                return

            await updater.add_artifact(
                parts=[
                    new_data_part(
                        draft.model_dump(mode="json"),
                        media_type="application/json",
                    )
                ],
                name=ITINERARY_DRAFT_ARTIFACT,
            )
            span.set_attribute("a2a.task.status", "completed")
            span.set_attribute("itinerary.requested_days", request.day_count)
            span.set_attribute("itinerary.assigned_days", len(draft.days))
            span.set_attribute("itinerary.unfilled_days", draft.unfilled_days)
            await updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        del context, event_queue
        raise NotImplementedError("Itinerary planning tasks cannot be cancelled.")


@asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
    del _app
    tracer_provider = None
    try:
        tracer_provider = configure_phoenix_tracing()
        yield
    finally:
        if tracer_provider is not None and not shutdown_phoenix_tracing():
            logger.warning("Phoenix spans were not fully exported at A2A shutdown")


_REQUEST_HANDLER = DefaultRequestHandler(
    agent_executor=ItineraryPlannerExecutor(A2ARestaurantEvidenceProvider()),
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
