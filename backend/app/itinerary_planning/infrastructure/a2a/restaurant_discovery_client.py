import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.client.errors import A2AClientError
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, Task, TaskState
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from app.agents.schemas import (
    RestaurantSearchAgentRequest,
    RestaurantSearchAgentResponse,
)

DEFAULT_RESTAURANT_AGENT_URL = "http://127.0.0.1:8001"
RESTAURANT_DISCOVERY_ARTIFACT = "restaurant_discovery_result"


class RestaurantDiscoveryDelegationError(RuntimeError):
    """The itinerary planner could not consume a restaurant discovery task."""


async def delegate_restaurant_discovery(
    query: str,
    *,
    agent_url: str = DEFAULT_RESTAURANT_AGENT_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RestaurantSearchAgentResponse:
    try:
        request_data = RestaurantSearchAgentRequest(query=query)
    except ValidationError:
        raise ValueError(
            "Restaurant request must contain between 1 and 2,000 characters."
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
                    message=new_text_message(
                        request_data.query,
                        role=Role.ROLE_USER,
                    )
                )
                responses = [
                    response async for response in client.send_message(request)
                ]
            finally:
                await client.close()
        except (A2AClientError, httpx.HTTPError) as error:
            raise RestaurantDiscoveryDelegationError(
                "Could not communicate with the restaurant discovery Agent."
            ) from error

    task_responses = [
        response.task
        for response in responses
        if response.WhichOneof("payload") == "task"
    ]
    if len(task_responses) != 1:
        raise RestaurantDiscoveryDelegationError(
            "The restaurant discovery Agent did not return one completed task."
        )

    return _read_restaurant_result(task_responses[0])


def _read_restaurant_result(task: Task) -> RestaurantSearchAgentResponse:
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        message = (
            get_message_text(task.status.message).strip()
            if task.status.HasField("message")
            else ""
        )
        raise RestaurantDiscoveryDelegationError(
            message or "The restaurant discovery Agent did not complete the task."
        )

    artifacts = [
        artifact
        for artifact in task.artifacts
        if artifact.name == RESTAURANT_DISCOVERY_ARTIFACT
    ]
    if len(artifacts) != 1:
        raise RestaurantDiscoveryDelegationError(
            "The completed task did not contain one restaurant discovery artifact."
        )

    json_parts = [
        part
        for part in artifacts[0].parts
        if part.WhichOneof("content") == "data"
        and part.media_type == "application/json"
    ]
    if len(json_parts) != 1:
        raise RestaurantDiscoveryDelegationError(
            "The restaurant discovery artifact was not a single JSON data part."
        )

    try:
        payload = MessageToDict(json_parts[0].data)
        return RestaurantSearchAgentResponse.model_validate(payload)
    except (TypeError, ValueError, ValidationError):
        raise RestaurantDiscoveryDelegationError(
            "The restaurant discovery Agent returned an invalid JSON result."
        ) from None
