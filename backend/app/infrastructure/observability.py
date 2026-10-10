from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode
from langsmith.run_helpers import tracing_context

from app.core.config import settings

_provider: TracerProvider | None = None
_tracer = trace.get_tracer("restaurant-ai-rag")


def configure_phoenix_tracing() -> TracerProvider | None:
    global _provider

    if not settings.phoenix_tracing_enabled:
        return None
    if _provider is not None:
        return _provider

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "restaurant-ai-rag",
                "openinference.project.name": settings.phoenix_project_name,
            }
        )
    )
    exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _provider = provider
    return provider


def shutdown_phoenix_tracing(timeout_millis: int = 5000) -> bool:
    global _provider

    provider = _provider
    _provider = None
    if provider is None:
        return True
    try:
        return provider.force_flush(timeout_millis=timeout_millis)
    finally:
        provider.shutdown()


@contextmanager
def disable_automatic_langchain_tracing() -> Generator[None, None, None]:
    with tracing_context(enabled=False):
        yield


@contextmanager
def traced_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Generator[Span, None, None]:
    tracer = (
        _provider.get_tracer("restaurant-ai-rag")
        if _provider is not None
        else _tracer
    )
    with tracer.start_as_current_span(
        name,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as error:
            span.set_attribute("error.type", type(error).__name__)
            span.set_status(Status(StatusCode.ERROR))
            raise
