import argparse
import asyncio

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.client.errors import A2AClientError
from a2a.helpers import get_message_text, new_data_message
from a2a.types import Role, SendMessageRequest, Task, TaskState
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from app.itinerary_planner import ItineraryDiningDraft, ItineraryPlanningRequest
from app.itinerary_planner_server import ITINERARY_DRAFT_ARTIFACT

DEFAULT_ITINERARY_PLANNER_AGENT_URL = "http://127.0.0.1:8002"


class ItineraryPlannerDelegationError(RuntimeError):
    """The caller could not consume an itinerary-planning task."""


async def delegate_itinerary_plan(
    query: str,
    day_count: int,
    *,
    agent_url: str = DEFAULT_ITINERARY_PLANNER_AGENT_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ItineraryDiningDraft:
    try:
        request_data = ItineraryPlanningRequest(
            query=query,
            day_count=day_count,
        )
    except ValidationError:
        raise ValueError(
            "Provide a non-empty restaurant query and a positive day count."
        ) from None

    async with httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(120.0, connect=5.0),
        trust_env=False,
    ) as http_client:
        try:
            agent_card = await A2ACardResolver(
                http_client,
                agent_url,
            ).get_agent_card()
            client = await create_client(
                agent=agent_card,
                client_config=ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                    accepted_output_modes=["application/json"],
                ),
            )
            try:
                request = SendMessageRequest(
                    message=new_data_message(
                        request_data.model_dump(mode="json"),
                        media_type="application/json",
                        role=Role.ROLE_USER,
                    )
                )
                responses = [
                    response async for response in client.send_message(request)
                ]
            finally:
                await client.close()
        except (A2AClientError, httpx.HTTPError) as error:
            raise ItineraryPlannerDelegationError(
                "Could not communicate with the Itinerary Planner Agent."
            ) from error

    task_responses = [
        response.task
        for response in responses
        if response.WhichOneof("payload") == "task"
    ]
    if len(task_responses) != 1:
        raise ItineraryPlannerDelegationError(
            "The Itinerary Planner Agent did not return one completed task."
        )

    return _read_itinerary_draft(task_responses[0])


def _read_itinerary_draft(task: Task) -> ItineraryDiningDraft:
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        message = (
            get_message_text(task.status.message).strip()
            if task.status.HasField("message")
            else ""
        )
        raise ItineraryPlannerDelegationError(
            message or "The Itinerary Planner Agent did not complete the task."
        )

    artifacts = [
        artifact
        for artifact in task.artifacts
        if artifact.name == ITINERARY_DRAFT_ARTIFACT
    ]
    if len(artifacts) != 1:
        raise ItineraryPlannerDelegationError(
            "The completed task did not contain one itinerary draft artifact."
        )

    json_parts = [
        part
        for part in artifacts[0].parts
        if part.WhichOneof("content") == "data"
        and part.media_type == "application/json"
    ]
    if len(json_parts) != 1:
        raise ItineraryPlannerDelegationError(
            "The itinerary draft artifact was not a single JSON data part."
        )

    try:
        payload = MessageToDict(json_parts[0].data)
        return ItineraryDiningDraft.model_validate(payload)
    except (TypeError, ValueError, ValidationError):
        raise ItineraryPlannerDelegationError(
            "The Itinerary Planner Agent returned an invalid dining draft."
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Request a dining draft from the Itinerary Planner Agent."
    )
    parser.add_argument("query", help="Restaurant request for the itinerary.")
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Number of itinerary days to fill with restaurant suggestions.",
    )
    parser.add_argument(
        "--agent-url",
        default=DEFAULT_ITINERARY_PLANNER_AGENT_URL,
        help="Base URL of the Itinerary Planner Agent.",
    )
    args = parser.parse_args()
    draft = asyncio.run(
        delegate_itinerary_plan(
            args.query,
            args.days,
            agent_url=args.agent_url,
        )
    )
    print(draft.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
