import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.client.errors import A2AClientError
from a2a.helpers import get_message_text, new_data_message
from a2a.types import Role, SendMessageRequest, Task, TaskState
from google.protobuf.json_format import MessageToDict
from pydantic import ValidationError

from app.agents.schemas import (
    ReviewRecommendationRequest,
    ReviewRecommendationResponse,
)
from app.core.config import settings

REVIEW_RECOMMENDATION_ARTIFACT = "review_recommendation_result"


class ReviewRecommendationDelegationError(RuntimeError):
    """The Restaurant Search Agent could not consume the recommendation task."""


async def delegate_review_recommendation(
    request: ReviewRecommendationRequest,
    *,
    agent_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ReviewRecommendationResponse:
    try:
        validated_request = ReviewRecommendationRequest.model_validate(request)
    except ValidationError as error:
        raise ValueError("The review recommendation request is invalid.") from error

    async with httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(120.0, connect=5.0),
        trust_env=False,
    ) as http_client:
        try:
            agent_card = await A2ACardResolver(
                http_client,
                agent_url or settings.review_recommendation_agent_url,
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
                a2a_request = SendMessageRequest(
                    message=new_data_message(
                        validated_request.model_dump(mode="json"),
                        media_type="application/json",
                        role=Role.ROLE_USER,
                    )
                )
                responses = [
                    response async for response in client.send_message(a2a_request)
                ]
            finally:
                await client.close()
        except (A2AClientError, httpx.HTTPError) as error:
            raise ReviewRecommendationDelegationError(
                "Could not communicate with the Review & Recommendation Agent."
            ) from error

    task_responses = [
        response.task
        for response in responses
        if response.WhichOneof("payload") == "task"
    ]
    if len(task_responses) != 1:
        raise ReviewRecommendationDelegationError(
            "The Review & Recommendation Agent did not return one task."
        )
    return _read_recommendation_result(task_responses[0])


def _read_recommendation_result(task: Task) -> ReviewRecommendationResponse:
    if task.status.state != TaskState.TASK_STATE_COMPLETED:
        message = (
            get_message_text(task.status.message).strip()
            if task.status.HasField("message")
            else ""
        )
        raise ReviewRecommendationDelegationError(
            message or "The Review & Recommendation Agent did not complete the task."
        )

    artifacts = [
        artifact
        for artifact in task.artifacts
        if artifact.name == REVIEW_RECOMMENDATION_ARTIFACT
    ]
    if len(artifacts) != 1:
        raise ReviewRecommendationDelegationError(
            "The completed task did not contain one recommendation artifact."
        )

    json_parts = [
        part
        for part in artifacts[0].parts
        if part.WhichOneof("content") == "data"
        and part.media_type == "application/json"
    ]
    if len(json_parts) != 1:
        raise ReviewRecommendationDelegationError(
            "The recommendation artifact was not one JSON data part."
        )

    try:
        payload = MessageToDict(json_parts[0].data)
        return ReviewRecommendationResponse.model_validate(payload)
    except (TypeError, ValueError, ValidationError):
        raise ReviewRecommendationDelegationError(
            "The Review & Recommendation Agent returned an invalid result."
        ) from None
