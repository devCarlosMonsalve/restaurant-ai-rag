import json
import logging
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from litellm import Router
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agents.schemas import (
    CandidateComparison,
    ReviewCandidate,
    ReviewRecommendationRequest,
    ReviewRecommendationResponse,
    UnattributedDocumentSource,
)
from app.core.config import settings
from app.infrastructure.llm.model_routing import (
    ModelRouteError,
    call_restaurant_search_model,
    create_restaurant_search_router,
)
from app.knowledge.application.contracts import DocumentSearchResult
from app.tools.document_search import search_documents

logger = logging.getLogger(__name__)

MAX_DOCUMENT_CHUNKS = 5
REVIEW_CORPUS_LIMITATION = (
    "Indexed documents have no reliable restaurant or review metadata. "
    "Retrieved chunks are not attributed to candidates and were not used as "
    "candidate-specific evidence."
)
OSM_EVIDENCE_LIMITATION = (
    "The comparison uses only the supplied indexed OSM fields. Missing fields "
    "do not prove that a restaurant lacks a feature."
)
SYSTEM_INSTRUCTION = """\
You compare the supplied restaurant candidates against the user's explicit
preferences. Use only the candidate fields supplied in the JSON input. These
fields are indexed OpenStreetMap evidence, not reviews and not proof of current
conditions. Do not use semantic similarity as quality or confidence.

The indexed document corpus has no reliable restaurant association or review
type. Any document source references in the input are unlinked and MUST NOT be
used to make claims about a candidate. Do not invent review content, ratings,
prices, opening hours, availability, or attributes absent from candidate data.

Return only JSON matching the supplied response schema. Every evidence item
must copy an exact value from its candidate and name one of the user's
preferences. Mark unsupported preferences unresolved. Recommend a candidate
only when the supplied fields directly support every preference; otherwise
set recommended_candidate_source_url to null. Do not return numeric scores.
"""


class _ReviewState(TypedDict):
    request: ReviewRecommendationRequest
    documents: list[DocumentSearchResult]
    retrieval_error: str | None
    response: ReviewRecommendationResponse | None


class _ReviewContext(TypedDict):
    router: Router
    session: Session


class ReviewRecommendationAgentError(RuntimeError):
    """The Agent could not produce a validated comparison."""


def run_review_recommendation_agent(
    request: ReviewRecommendationRequest,
    session: Session,
) -> ReviewRecommendationResponse:
    try:
        validated_request = ReviewRecommendationRequest.model_validate(request)
    except ValidationError as error:
        raise ValueError("The review request is invalid.") from error

    api_key = settings.gemini_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise ReviewRecommendationAgentError(
            "GEMINI_API_KEY must be set to run the Review & Recommendation Agent"
        )

    openai_api_key = (
        settings.openai_api_key.get_secret_value().strip()
        if settings.openai_api_key is not None
        else ""
    )
    router = create_restaurant_search_router(
        api_key.get_secret_value(),
        openai_api_key=openai_api_key or None,
    )
    result = _REVIEW_GRAPH.invoke(
        {
            "request": validated_request,
            "documents": [],
            "retrieval_error": None,
            "response": None,
        },
        context={"router": router, "session": session},
        config={"recursion_limit": 5},
    )
    response = result["response"]
    if response is None:
        raise ReviewRecommendationAgentError(
            "The Review & Recommendation graph did not produce a comparison"
        )
    return response


def _retrieve_documents_node(
    state: _ReviewState,
    runtime: Runtime[_ReviewContext],
) -> dict[str, object]:
    try:
        documents = search_documents(
            state["request"].query,
            runtime.context["session"],
            top_k=MAX_DOCUMENT_CHUNKS,
        )
    except Exception as error:
        logger.exception("Review Agent document retrieval failed")
        return {
            "documents": [],
            "retrieval_error": type(error).__name__,
        }
    return {"documents": documents, "retrieval_error": None}


def _compare_candidates_node(
    state: _ReviewState,
    runtime: Runtime[_ReviewContext],
) -> dict[str, ReviewRecommendationResponse]:
    request = state["request"]
    unattributed_documents = [
        UnattributedDocumentSource(
            source_name=document.source_name,
            chunk_index=document.chunk_index,
        )
        for document in state["documents"]
    ]
    prompt_payload = {
        "request": request.model_dump(mode="json"),
        "unattributed_document_sources": [
            document.model_dump(mode="json")
            for document in unattributed_documents
        ],
        "response_schema": ReviewRecommendationResponse.model_json_schema(),
    }
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=json.dumps(prompt_payload, ensure_ascii=False))],
        )
    ]
    try:
        model_response = call_restaurant_search_model(
            runtime.context["router"],
            contents,
            SYSTEM_INSTRUCTION,
            tools_enabled=False,
        )
    except ModelRouteError as error:
        raise ReviewRecommendationAgentError(
            "The Review & Recommendation Agent could not obtain a model response."
        ) from error

    output = model_response.text
    if output is None or not output.strip():
        raise ReviewRecommendationAgentError(
            "The Review & Recommendation model returned an empty response"
        )
    try:
        response = ReviewRecommendationResponse.model_validate_json(output)
    except ValidationError as error:
        raise ReviewRecommendationAgentError(
            "The Review & Recommendation model returned an invalid comparison"
        ) from error

    _validate_comparison(response, request)
    comparisons = [
        comparison.model_copy(
            update={
                "reasons": _grounded_reasons(comparison),
            }
        )
        for comparison in response.comparisons
    ]
    limitations = [REVIEW_CORPUS_LIMITATION, OSM_EVIDENCE_LIMITATION]
    if state["retrieval_error"]:
        limitations.append(
            "Document retrieval was unavailable; no document excerpts were considered."
        )
    response = response.model_copy(
        update={
            "comparisons": comparisons,
            "unattributed_documents": unattributed_documents,
            "limitations": limitations,
        }
    )
    _validate_recommendation(response, request)
    return {"response": response}


def _validate_comparison(
    response: ReviewRecommendationResponse,
    request: ReviewRecommendationRequest,
) -> None:
    candidates = {candidate.source_url: candidate for candidate in request.candidates}
    comparisons = {
        comparison.candidate_source_url: comparison
        for comparison in response.comparisons
    }
    if len(comparisons) != len(response.comparisons) or set(comparisons) != set(
        candidates
    ):
        raise ReviewRecommendationAgentError(
            "The model comparison did not cover each candidate exactly once"
        )

    allowed_preferences = set(request.preferences)
    for source_url, comparison in comparisons.items():
        candidate = candidates[source_url]
        evidenced_preferences: set[str] = set()
        for preference in comparison.unresolved_preferences:
            if preference not in allowed_preferences:
                raise ReviewRecommendationAgentError(
                    "The model returned an unknown preference"
                )
        for evidence in comparison.evidence:
            if evidence.preference not in allowed_preferences:
                raise ReviewRecommendationAgentError(
                    "The model evidence refers to an unknown preference"
                )
            candidate_value = getattr(candidate, evidence.field)
            allowed_values = (
                candidate_value if isinstance(candidate_value, list) else [candidate_value]
            )
            if evidence.value not in allowed_values:
                raise ReviewRecommendationAgentError(
                    "The model cited evidence not present in the candidate"
                )
            evidenced_preferences.add(evidence.preference)
        unresolved_preferences = set(comparison.unresolved_preferences)
        if evidenced_preferences & unresolved_preferences or (
            evidenced_preferences | unresolved_preferences
        ) != allowed_preferences:
            raise ReviewRecommendationAgentError(
                "The model must account for each preference with evidence or mark it unresolved"
            )
        if comparison.assessment == "matches" and (
            unresolved_preferences or evidenced_preferences != allowed_preferences
        ):
            raise ReviewRecommendationAgentError(
                "A full match must cite evidence for every preference"
            )
        if comparison.assessment == "partially_matches" and not (
            evidenced_preferences and unresolved_preferences
        ):
            raise ReviewRecommendationAgentError(
                "A partial match must contain supported and unresolved preferences"
            )
        if comparison.assessment == "insufficient_evidence" and (
            evidenced_preferences or unresolved_preferences != allowed_preferences
        ):
            raise ReviewRecommendationAgentError(
                "Insufficient evidence must leave every preference unresolved"
            )


def _grounded_reasons(comparison: CandidateComparison) -> list[str]:
    reasons = [
        f"{evidence.field}={evidence.value}; preference={evidence.preference}"
        for evidence in comparison.evidence
    ]
    reasons.extend(
        f"No supplied field verifies: {preference}"
        for preference in comparison.unresolved_preferences
    )
    return reasons or ["No supplied candidate field verifies a preference."]


def _validate_recommendation(
    response: ReviewRecommendationResponse,
    request: ReviewRecommendationRequest,
) -> None:
    if response.recommended_candidate_source_url is None:
        return
    comparison = next(
        (
            item
            for item in response.comparisons
            if item.candidate_source_url == response.recommended_candidate_source_url
        ),
        None,
    )
    if comparison is None:
        raise ReviewRecommendationAgentError(
            "The recommended candidate was not in the request"
        )
    if (
        comparison.assessment != "matches"
        or comparison.unresolved_preferences
        or {
            evidence.preference for evidence in comparison.evidence
        }
        != set(request.preferences)
    ):
        raise ReviewRecommendationAgentError(
            "The recommended candidate lacks evidence for every preference"
        )


def _build_review_graph():
    graph = StateGraph(_ReviewState, context_schema=_ReviewContext)
    graph.add_node("retrieve_documents", _retrieve_documents_node)
    graph.add_node("compare_candidates", _compare_candidates_node)
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "compare_candidates")
    graph.add_edge("compare_candidates", END)
    return graph.compile()


_REVIEW_GRAPH = _build_review_graph()
