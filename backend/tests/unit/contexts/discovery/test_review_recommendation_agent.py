from collections.abc import Generator
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from langsmith.utils import get_env_var
from pydantic import SecretStr

from app.agents import review_recommendation_agent as review_agent
from app.agents.schemas import ReviewCandidate, ReviewRecommendationRequest
from app.core.config import settings
from app.knowledge.application.contracts import DocumentSearchResult


@pytest.fixture(autouse=True)
def clear_langsmith_environment_cache() -> Generator[None, None, None]:
    get_env_var.cache_clear()
    yield
    get_env_var.cache_clear()


def make_request(
    preferences: list[str] | None = None,
) -> ReviewRecommendationRequest:
    return ReviewRecommendationRequest(
        query="Compare vegetarian restaurants in Madrid",
        preferences=preferences or ["vegetarian options"],
        candidates=[
            ReviewCandidate(
                name="Casa Verde",
                city="Madrid",
                cuisine="vegetarian",
                location="Calle Mayor 1",
                features=["Opciones vegetarianas: disponible"],
                source_url="https://www.openstreetmap.org/node/123",
                attribution="© OpenStreetMap contributors",
                attribution_url="https://www.openstreetmap.org/copyright",
            )
        ],
    )


def model_payload(
    request: ReviewRecommendationRequest,
    *,
    assessment: str = "matches",
    unresolved: list[str] | None = None,
    evidence: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    candidate = request.candidates[0]
    unresolved_preferences = unresolved or []
    matching_evidence = evidence or [
        {
            "field": "features",
            "value": "Opciones vegetarianas: disponible",
            "preference": request.preferences[0],
        }
    ]
    return {
        "recommended_candidate_source_url": (
            candidate.source_url if assessment == "matches" else None
        ),
        "comparisons": [
            {
                "candidate_source_url": candidate.source_url,
                "assessment": assessment,
                "reasons": ["The indexed candidate features match the preference."],
                "evidence": matching_evidence,
                "unresolved_preferences": unresolved_preferences,
            }
        ],
        "unattributed_documents": [],
        "limitations": ["No review content was used."],
    }


def configure_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_output: str,
    documents: list[DocumentSearchResult] | None = None,
    retrieval_error: Exception | None = None,
    captured_prompt: list[dict[str, object]] | None = None,
) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", SecretStr("test-api-key"))
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.setattr(
        review_agent,
        "create_restaurant_search_router",
        lambda *_args, **_kwargs: object(),
    )

    def retrieve(*_args: object, **_kwargs: object) -> list[DocumentSearchResult]:
        if retrieval_error is not None:
            raise retrieval_error
        return documents or []

    def call_model(
        _router: object,
        contents: list[object],
        _system_instruction: str,
        *,
        tools_enabled: bool,
    ) -> SimpleNamespace:
        assert tools_enabled is False
        if captured_prompt is not None:
            captured_prompt.append(json.loads(contents[0].parts[0].text))
        return SimpleNamespace(text=model_output)

    monkeypatch.setattr(review_agent, "search_documents", retrieve)
    monkeypatch.setattr(review_agent, "call_restaurant_search_model", call_model)


def test_review_agent_returns_grounded_comparison_and_unattributed_document_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = make_request()
    source = DocumentSearchResult(
        document_id=UUID(int=1),
        source_name="eval_menu.txt",
        chunk_index=2,
        content="This excerpt is not linked to the candidate.",
        similarity=0.91,
    )
    captured_prompt: list[dict[str, object]] = []
    configure_agent(
        monkeypatch,
        model_output=json.dumps(model_payload(request)),
        documents=[source],
        captured_prompt=captured_prompt,
    )

    response = review_agent.run_review_recommendation_agent(request, object())

    assert response.recommended_candidate_source_url == request.candidates[0].source_url
    assert response.comparisons[0].evidence[0].value in request.candidates[0].features
    assert response.unattributed_documents[0].source_name == "eval_menu.txt"
    assert review_agent.REVIEW_CORPUS_LIMITATION in response.limitations
    assert len(captured_prompt) == 1
    prompt_documents = captured_prompt[0]["unattributed_document_sources"]
    assert prompt_documents == [{"source_name": "eval_menu.txt", "chunk_index": 2}]
    assert "This excerpt is not linked to the candidate." not in json.dumps(
        captured_prompt[0]
    )


def test_review_agent_does_not_recommend_when_preferences_are_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = make_request(["vegetarian options", "quiet atmosphere"])
    payload = model_payload(
        request,
        assessment="partially_matches",
        unresolved=["quiet atmosphere"],
        evidence=[
            {
                "field": "features",
                "value": "Opciones vegetarianas: disponible",
                "preference": "vegetarian options",
            }
        ],
    )
    payload["recommended_candidate_source_url"] = None
    configure_agent(
        monkeypatch,
        model_output=json.dumps(payload),
    )

    response = review_agent.run_review_recommendation_agent(request, object())

    assert response.recommended_candidate_source_url is None
    assert response.comparisons[0].assessment == "partially_matches"
    assert response.comparisons[0].unresolved_preferences == ["quiet atmosphere"]


@pytest.mark.parametrize(
    "model_output",
    [
        "not JSON",
        json.dumps(
            {
                "recommended_candidate_source_url": "https://attacker.invalid/",
                "comparisons": [],
                "unattributed_documents": [],
                "limitations": ["No review data."],
            }
        ),
        json.dumps(
            {
                "recommended_candidate_source_url": "https://www.openstreetmap.org/node/123",
                "comparisons": [
                    {
                        "candidate_source_url": "https://www.openstreetmap.org/node/123",
                        "assessment": "matches",
                        "reasons": ["Unsupported fact."],
                        "evidence": [
                            {
                                "field": "cuisine",
                                "value": "michelin-starred",
                                "preference": "vegetarian options",
                            }
                        ],
                        "unresolved_preferences": [],
                    }
                ],
                "unattributed_documents": [],
                "limitations": ["No review data."],
            }
        ),
    ],
)
def test_review_agent_rejects_invalid_or_unsupported_model_output(
    monkeypatch: pytest.MonkeyPatch,
    model_output: str,
) -> None:
    request = make_request()
    configure_agent(monkeypatch, model_output=model_output)

    with pytest.raises(review_agent.ReviewRecommendationAgentError):
        review_agent.run_review_recommendation_agent(request, object())


def test_review_agent_reports_document_retrieval_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = make_request()
    configure_agent(
        monkeypatch,
        model_output=json.dumps(model_payload(request)),
        retrieval_error=RuntimeError("private embedding provider detail"),
    )

    response = review_agent.run_review_recommendation_agent(request, object())

    assert response.unattributed_documents == []
    assert any(
        "Document retrieval was unavailable" in limitation
        for limitation in response.limitations
    )
