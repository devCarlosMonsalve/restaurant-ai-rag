# Backend architecture and file decisions

## Strategic model

The backend is a modular monolith. Contexts own their language and policies;
HTTP, Tools, MCP, A2A, and command-line programs are delivery adapters, not
bounded contexts.

| Context or capability | Responsibility | Relationship |
| --- | --- | --- |
| Restaurant Discovery | Searches OSM places and indexed restaurant imagery; applies supported feature evidence, freshness, corpus, and photo-association rules. | Supplies ranked candidates and evidence to Tools, Agents, workflows, and HTTP. |
| Knowledge | Prepares and retrieves indexed documents and generates answers grounded in retrieved excerpts. | Receives document chunks from ingestion; uses PostgreSQL retrieval and a Gemini generation adapter. |
| Ingestion | Coordinates text, image, and OSM indexing flows while retaining their existing transaction and provenance behavior. | Writes to the Knowledge or Restaurant Discovery stores; filesystem, embeddings, OSM, and Commons remain technical adapters. |
| Agent orchestration | Runs model/tool cycles and the restaurant-photo workflow. | Calls application use cases through the Tool registry; does not own restaurant evidence or retrieval rules. |
| Itinerary dining planning | Produces the current ranked-candidate draft for requested days. | Consumes restaurant discovery over the A2A boundary; it does not promise route feasibility, opening hours, availability, or reservations. |
| Shared infrastructure | SQLAlchemy sessions/ORM, PostgreSQL/pgvector, Gemini and CLIP embeddings, LiteLLM routing, tracing, and external-data clients. | Technical capabilities shared by contexts; not a domain or shared domain model. |
| Delivery interfaces | FastAPI, Tools, MCP, A2A, and CLI entry points. | Validate/translate public contracts and compose application dependencies. |

The main context relationships are:

- Ingestion writes data consumed by Knowledge and Restaurant Discovery.
- Agents and workflows consume application capabilities; they do not duplicate
  search, evidence, or attribution policies.
- Itinerary planning consumes Restaurant Discovery through the existing A2A
  protocol adapter.
- FastAPI, Tools, MCP, and A2A expose capabilities while preserving their
  current public schemas and artifact names.

## Tactical DDD decisions

Use DDD only where code demonstrates a domain rule:

- Restaurant feature requirements, evidence status, kosher freshness, query
  intent, and photo association are policies in
  `app.restaurant_discovery.domain`.
- Candidate/photo association requires exact equality of non-empty OSM source
  URLs. Names, filenames, and similarity are not substitutes for identity.
- Knowledge text normalization and chunking are pure content-preparation
  policies; the retrieval ranking remains an infrastructure concern.
- Existing request/response schemas are DTOs. `OsmPlace`, `ImageEmbedding`,
  and `DocumentChunk` are ORM persistence records, not aggregates.
- No current behavior justifies a generic Restaurant aggregate, a
  recommendation aggregate, or a confidence value around similarity. The
  itinerary flow does not currently implement enough feasibility rules to
  justify an itinerary aggregate.

The tests under `tests/architecture/` protect the dependency-free domains in
Restaurant Discovery, Knowledge, and the remaining shared domain package,
statically check the `app.*` import graph for cycles, and prevent `main.py`
from importing ORM models or SQL query builders. Static checks do not detect
dynamic imports.

## Current backend tree

```text
backend/
├── app/
│   ├── agents/                         # model/tool orchestration
│   ├── application/                    # shared cross-context application flows and legacy exports
│   ├── core/                           # settings
│   ├── domain/                         # remaining shared OSM indexing policy and legacy exports
│   ├── restaurant_discovery/           # restaurant discovery bounded context
│   │   ├── application/                # contracts, use cases, and repository port
│   │   ├── domain/                     # evidence, query intent, and photo policies
│   │   └── infrastructure/postgres.py  # context PostgreSQL adapter
│   ├── infrastructure/
│   │   ├── filesystem/                 # document file adapter
│   │   └── persistence/postgres/       # catalog, ingestion, pgvector, and search adapters
│   ├── ingestion/
│   │   ├── application/                # ports and provider-independent use cases
│   │   └── composition.py              # concrete provider and persistence wiring
│   ├── knowledge/
│   │   ├── application/                # contracts, retrieval and answer use cases
│   │   ├── domain/                     # text preparation
│   │   └── infrastructure/             # PostgreSQL and generation adapters
│   ├── models/                         # SQLAlchemy ORM records
│   ├── tools/                          # model-facing Tool adapters/registry
│   ├── workflows/                     # LangGraph workflow adapters
│   └── *.py                            # compatibility, provider, and entry modules
├── alembic/                            # migration history; unchanged
├── data/                               # corpus and evaluation fixtures
├── evaluation/                          # shared offline evaluation utilities
├── scripts/                             # operational diagnostics
├── static/                             # backend-served static page
├── tests/
│   ├── architecture/                   # architectural boundary checks
│   ├── contracts/                      # HTTP, Tools, MCP, and A2A contracts
│   ├── evaluation/                     # evaluator and dataset tests
│   ├── integration/                    # opt-in PostgreSQL/pgvector tests
│   └── unit/                           # domain, application, context, and adapter tests
├── *.py                                # documented CLI entrypoints retained for compatibility
├── alembic.ini, pytest.ini             # tool-discovered configuration
└── requirements*.txt, README.md        # environment and operations documentation
```

This is an incremental structure, not a claim that every root-level module has
already moved into its final context. In particular, the original import paths
remain where they are compatibility entry points or are shared across
contexts.

The three basic catalog endpoints use the `RestaurantCatalog` application
port. `main.py` constructs `PostgresRestaurantCatalog` through a FastAPI
dependency; ORM queries, commit/refresh, and OSM response mapping are in the
PostgreSQL adapter. This keeps the HTTP handlers focused on schemas, status
codes, and dependency wiring without inventing a catalog aggregate where no
business invariant currently exists.

| Package files | Responsibility / decision |
| --- | --- |
| `app/domain/__init__.py`, `app/domain/osm_place.py` | Keep `osm_place.py` as the pure OSM embedding invalidation policy. `domain/restaurant_discovery/*` and `app/application/{ports.py,restaurant_discovery.py}` are compatibility re-exports to the canonical context modules. |
| `app/application/{__init__.py,document_answer.py,restaurant_catalog.py}` | Keep shared application flows that coordinate more than one bounded context and the restaurant catalog operations; they contain no ORM queries. |
| `app/restaurant_discovery/{__init__.py,domain/{__init__.py,evidence.py,photo_association.py,query_intent.py},application/{__init__.py,contracts.py,ports.py,service.py},infrastructure/{__init__.py,postgres.py}}` | Canonical Restaurant Discovery context: pure feature/evidence and identity rules, context-owned request/response contracts, application orchestration behind a repository port, and a PostgreSQL adapter. Validate use cases, Tools, Agent, workflow, query, and API regressions. |
| `app/infrastructure/{__init__.py,filesystem/__init__.py,filesystem/text_documents.py}` | Keep; filesystem reading and text-file normalization are adapters, not HTTP/application code. |
| `app/infrastructure/persistence/{__init__.py,postgres/__init__.py,postgres/evidence_queries.py,postgres/restaurant_queries.py,postgres/image_queries.py,postgres/restaurant_catalog.py,postgres/ingestion.py}` | Keep; shared PostgreSQL/pgvector query builders, catalog mapping, and ingestion persistence/transaction ownership belong here. The former discovery adapter path is a compatibility re-export. Validate via focused query/API and ingestion tests plus the optional PostgreSQL integration suite. |
| `app/knowledge/{__init__.py,domain/__init__.py,domain/text.py,application/__init__.py,application/contracts.py,application/ports.py,application/answer_question.py,application/search_documents.py,infrastructure/__init__.py,infrastructure/postgres/__init__.py,infrastructure/postgres/retriever.py,infrastructure/generation/__init__.py,infrastructure/generation/answer_chain.py}` | Keep the context-first structure; it owns document-search/RAG contracts and separates text policy, retrieval/answer use cases, database retrieval, and Gemini/LangChain generation. Validate with RAG/retrieval tests using mocks. |
| `app/ingestion/{__init__.py,composition.py,application/{__init__.py,ports.py,documents_usecase.py,images_usecase.py}}` | Keep; application use cases depend on ports, while composition wires filesystem/embedding providers and PostgreSQL adapters. `application/{documents.py,images.py,osm.py}` remain thin temporary compatibility facades. Validate with use-case, transaction, ingestion, and compatibility tests. |
| `app/models/{__init__.py,base.py,restaurant.py,osm_place.py,image_embedding.py,document_chunk.py}` | Keep as ORM persistence records. `alembic/env.py` imports `app.models` so all tables remain registered. Validate metadata imports and Alembic history. |
| `app/agents/{__init__.py,schemas.py,restaurant_search_agent.py}` | Keep public Agent schemas unchanged. `restaurant_search_agent.py` remains the LangGraph/model/tool orchestration boundary and calls the Discovery context for policies and use cases; turn/tool limits and photo dispatch remain contract-sensitive graph concerns. |
| `app/workflows/{__init__.py,schemas.py,restaurant_photo_search.py}` | Keep workflow DTOs and graph boundary. The photo workflow coordinates discovery and scoped image lookup; exact source-URL association remains a domain policy. Do not merge it into an aggregate. |
| `app/tools/{__init__.py,registry.py,restaurant_search.py,restaurant_photos.py,document_search.py,document_answer.py}` | Keep the registry and thin adapters. MCP and the model Agent share names, input schemas, hidden arguments, result serialization, and stable error categories. |

Ingestion document/image use cases depend on typed application ports, while
the composition module wires filesystem/embedding providers and PostgreSQL
adapters. The OSM import path remains a compatibility boundary around an
infrastructure upsert; its searchable-metadata invalidation policy is pure
domain logic. Provider DTOs and shared schema contracts remain candidates for
later context ownership, but must not be moved without preserving CLI and
external integrations.

## `backend/app/` root module decisions

| Current file or group | Responsibility / category | Decision | Reason and validation |
| --- | --- | --- | --- |
| `__init__.py` | Python package marker | Keep | Required package path; no behavior. |
| `main.py` | FastAPI ASGI entry point, routes, and explicit dependency composition | Keep as the stable ASGI entry point; catalog endpoints inject the application port and delegate. | `uvicorn app.main:app`, API tests, and `get_db` overrides depend on it. Validate `test_api.py`; route paths, response models, ordering, and status codes remain unchanged. |
| `a2a_server.py`, `itinerary_planner_server.py`, `itinerary_planner_client.py`, `itinerary_planner_agent_client.py` | A2A server/client adapters | Keep as protocol/import entry points. | Agent cards, artifact names, JSON parts, and client imports are contracts; validate with the four A2A test modules. |
| `mcp_server.py`, `mcp_schemas.py` | MCP delivery adapter and public MCP DTOs | Keep as protocol/import entry points. | Stdio launch and public schemas are compatibility boundaries; validate with `test_mcp_server.py`. |
| `schemas.py` | Remaining shared FastAPI/Tool request and response DTOs | Keep as a compatibility aggregator; Discovery contracts are owned by `restaurant_discovery.application.contracts`, Knowledge search/answer contracts by `knowledge.application.contracts`, and both are re-exported here. | Preserves existing schema import paths and class identity while establishing context ownership. Validate API, Tool, MCP, and Agent contracts. |
| `restaurant_search.py`, `image_search.py` | Legacy search import paths | Keep as compatibility facades; SQL/pgvector implementations are in `infrastructure/persistence/postgres/restaurant_queries.py` and `image_queries.py`. | Retains existing callers while making PostgreSQL infrastructure own the queries. Validate discovery, image pipeline, and PostgreSQL query tests. |
| `document_ingestion.py`, `image_ingestion.py`, `osm_ingestion.py` | Legacy ingestion import paths | Keep as stable facades into `app.ingestion`; composition wires provider-independent use cases to PostgreSQL adapters. | Existing CLI and test imports continue to work. Validate document/image/OSM ingestion and compatibility tests. |
| `coverage_audit.py` | Aggregation for an operational coverage report | Keep for now as a shared audit helper. | The report consumes ORM-shaped records and OSM feature labels; splitting it from the CLI is not needed to establish a business aggregate. Validate with `test_coverage_audit.py`. |
| `database.py` | SQLAlchemy engine, session factory, FastAPI session dependency | Keep as explicit shared composition infrastructure. | Alembic, FastAPI, MCP, A2A, and tests rely on the same session factory; moving it would add compatibility wrappers without changing the boundary. Validate imports, API/MCP/A2A tests, and Alembic. |
| `embeddings.py`, `image_embeddings.py` | Gemini text and OpenCLIP provider adapters | Keep as shared provider modules for now. | Multiple ingestion/search contexts use them. Preserve model IDs, prefixes, 768/512 dimensions, normalization, and retry behavior; validate embedding and image-pipeline tests. |
| `open_data_sources.py` | Overpass and Wikimedia Commons clients, parsing, and provenance/license checks | Keep as the shared external-data adapter import path. | Ingestion scripts and tests depend on it; moving it is unnecessary for the new ingestion use-case boundary. Validate `test_open_data_sources.py` and offline ingestion tests. |
| `model_routing.py` | LiteLLM model/provider routing | Keep as the current provider adapter used by the discovery Agent. | It is an infrastructure capability, not discovery domain logic; validate `test_model_routing.py` and Agent tests with mocks. |
| `observability.py` | OpenTelemetry/Phoenix and LangSmith tracing helpers | Keep as shared cross-cutting infrastructure. | Several contexts use the same trace policy; validate observability and RAG tests without starting an exporter. |
| `place_filters.py` | Search filter helpers | Keep at the compatibility boundary pending a dedicated import migration. | Query semantics must remain identical; validate restaurant inclusion and PostgreSQL query tests. |
| `image_presentation.py` | Converts local image paths to safe API-relative URLs | Keep as a small presentation helper. | It is used at delivery boundaries and does not belong in the domain; validate API, Tools, and MCP output contracts. |
| `itinerary_planner.py` | Itinerary dining draft application flow | Keep at its stable import path; do not model an aggregate without additional rules. | A2A server/client and tests depend on this module; validate itinerary planner tests. |
| `core/{__init__.py,config.py}` | Environment settings and secrets | Keep under `core/`. | Shared configuration and test setup depend on it; no domain imports should reach it. |

`app/application/restaurant_catalog.py` defines the small catalog port and
application operations for create/list; it contains no SQL or ORM types.
`app/infrastructure/persistence/postgres/restaurant_catalog.py` owns the
corresponding SQLAlchemy queries, transaction behavior, and ORM-to-response
mapping. No schema, table, or migration changed.

## Backend-root files, configuration, and data

| Current file or group | Decision | Reason / validation |
| --- | --- | --- |
| `.env.example` | Keep in `backend/` | Conventionally discovered environment template; must contain placeholders only. |
| `alembic.ini`, `alembic/env.py`, `alembic/README`, `alembic/script.py.mako`, and every file in `alembic/versions/` | Keep in place; do not rename revisions. | Alembic convention and immutable migration history. Validate `python -m alembic heads` and use a disposable database for upgrades. Current head: `c80d3f91a642`. |
| `pytest.ini` | Keep in `backend/`; constrain discovery to `tests` and exclude integration tests by default. | Keeps pytest focused on categorized test modules and prevents accidental PostgreSQL integration runs. Explicit `-m integration` remains available. |
| `requirements.txt`, `requirements-dev.txt` | Keep | Deployment/environment tooling expects these paths. Pylance import analysis found no unresolved top-level imports. |
| `README.md` | Keep | Backend operational commands and protocol/evaluation documentation. Update when a documented path changes. |
| `audit_restaurant_coverage.py`, `evaluate_image_search.py`, `evaluate_rag.py`, `evaluate_restaurant_search.py` | Keep as root CLI/evaluation entry points. | Their direct commands are documented and convenient; keep executable entrypoints stable. Validate `--help` and fixture-only tests; do not run live benchmarks. |
| `evaluation/utils.py` | Move shared evaluation fingerprint/report helpers out of the backend root. | It is an importable library, not a CLI or tool-discovered configuration. Evaluator entrypoints import it as `evaluation.utils`; validate utility tests and each CLI `--help`. |
| `ingest_document.py`, `ingest_images.py`, `ingest_madrid_commons_images.py`, `ingest_osm_place_embeddings.py` | Keep as root CLI entry points. | Existing operations and README commands depend on these paths. Validate `--help` and mocked/unit coverage; do not run network, model, or database ingestion. |
| `scripts/diagnostics/check_database_connection.py` | Move the former `test_connection.py` into operational diagnostics and guard execution with `main()`. | It is a manual DB diagnostic, not a pytest test; run only via `python -m scripts.diagnostics.check_database_connection` against an explicitly configured database. |
| `data/samples/menu.txt` | Move the sample document into data. | A sample corpus file is data, not backend application configuration. Its contents are unchanged; no code or docs depended on the old root path. |
| `static/index.html` | Keep under `static/`. | Runtime asset served by FastAPI; validate the corresponding API/static behavior if changed. |
| `data/evaluation/`, `data/image_evaluation/`, `data/restaurant_evaluation/`, and `data/images/` | Keep as corpus/evaluation assets. | Evaluators and persisted image paths rely on these locations; no corpus or benchmark was changed or run. |
| `infrastructure/docker/docker-compose*.yml`, `init-db.sql`, and `README.md` | Keep in the repository infrastructure tree. | Docker Compose and documented PostgreSQL/Phoenix operations depend on these paths. No services were started for this audit. |
| `docs/ROADMAP.md` | Keep under project docs, unchanged. | It is ignored by the existing `docs/` Git rule; do not silently change ignore behavior as part of backend organization. |
| Local `.env` files, virtual environments, caches, bytecode, and downloaded images | Keep local/untracked; do not inventory as source or move. | May contain secrets or generated data. No secret values were read. |

## Test inventory and organization decisions

`tests/conftest.py` stays at the test root because it supplies shared fixtures.
Tests are grouped by their primary responsibility; existing test functions,
fixtures, and markers are preserved. README selectors are updated to the new
paths, while `testpaths = tests` continues to discover the complete tree.

| Group / files | Category | Decision / risk |
| --- | --- | --- |
| `tests/conftest.py` | Shared SQLite and FastAPI fixtures | Keep at root; changing the global dependency override risks every API test. |
| `tests/architecture/*.py` | Architecture | Keep in `architecture/`; enforce dependency direction, context application independence, domain purity, and HTTP composition. |
| `tests/integration/{conftest.py,test_postgres_queries.py}` | PostgreSQL/pgvector integration | Keep opt-in; requires guarded `POSTGRES_TEST_DATABASE_URL`, migrations, and pgvector. |
| `tests/unit/domain/{test_domain_policies.py,test_search_evidence.py,test_text_processing.py}` | Pure evidence, freshness, and text policies | Keep separate from infrastructure and delivery contracts. |
| `tests/unit/application/test_application_use_cases.py` | Application orchestration | Keeps cross-context use-case tests together. |
| `tests/unit/contexts/discovery/{test_image_pipeline.py,test_restaurant_discovery_postgres_adapter.py,test_restaurant_discovery_query_intent.py,test_restaurant_photo_workflow.py,test_restaurant_search_agent.py,test_restaurant_search_inclusion.py}` | Restaurant Discovery and image retrieval behavior | Protects ranking/filter regressions, policies, adapter telemetry, orchestration, and evidence behavior. |
| `tests/unit/contexts/ingestion/{test_document_ingestion.py,test_ingestion_compatibility.py,test_ingestion_postgres_adapters.py,test_ingestion_use_cases.py,test_open_data_sources.py,test_osm_ingestion.py}` | Ingestion behavior | Protects provider-independent flows, provenance, adapters, and legacy imports. |
| `tests/unit/contexts/knowledge/{test_document_search.py,test_rag.py}` | Knowledge/RAG behavior | Protects retrieval/answer formats, sources, and no-document behavior. |
| `tests/unit/contexts/planning/test_itinerary_planner.py` | Itinerary planning policy | Synthetic candidates only; no agent or provider calls. |
| `tests/unit/infrastructure/{test_coverage_audit.py,test_embeddings.py,test_model_routing.py,test_observability.py}` | Shared provider, observability, and report behavior | Provider calls remain mocked; no live traces or benchmarks. |
| `tests/contracts/http/test_api.py` | HTTP contract | Preserve route schemas and status/error behavior. |
| `tests/contracts/tools/{test_restaurant_search_tool.py,test_retrieval_tools.py}` | Tool contract | Preserve tool argument validation, visibility, and output behavior. |
| `tests/contracts/mcp/test_mcp_server.py` | MCP contract | Preserve stdio tool schemas, output, and sanitized errors. |
| `tests/contracts/a2a/{test_a2a_server.py,test_itinerary_planner_agent_client.py,test_itinerary_planner_client.py,test_itinerary_planner_server.py}` | A2A protocol contract | Preserve Agent Cards, artifacts, task lifecycle, and sanitized errors. |
| `tests/evaluation/{test_evaluation_data.py,test_evaluation_utils.py,test_image_evaluation.py,test_rag_evaluation.py,test_restaurant_evaluation.py}` | Evaluation datasets, helper, and metric tests | Do not invoke live benchmarks as routine validation. |

`pytest.ini` sets `testpaths = tests`; operational scripts live outside the
test tree and cannot be imported by ordinary discovery. The default marker excludes
integration tests; `pytest tests/integration -m integration` explicitly selects
them. The integration fixture must only be pointed at a disposable test
database.

No project CI workflow or coverage threshold configuration was present in the
audited tree, so there were no CI paths or coverage settings to migrate.

## Validation baseline

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests\integration -m integration --collect-only -q
.\.venv\Scripts\python.exe -m alembic heads
```

The first command uses mocks and the in-memory SQLite fixtures. The integration
collection command does not run SQL; executing the integration tests requires
an explicitly configured isolated PostgreSQL/pgvector service. Do not invoke
Gemini, download external data, or run evaluation benchmarks during an
architectural reorganization.
