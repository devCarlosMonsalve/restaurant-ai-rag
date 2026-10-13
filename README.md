# Restaurant AI RAG

A modular restaurant-discovery application and AI Engineering learning project. It combines a FastAPI backend, a Next.js chat interface, PostgreSQL with pgvector, model-backed retrieval and generation, and separate MCP and A2A integration points.

[Overview](#what-it-does) · [Architecture](#architecture-at-a-glance) · [Run locally](#run-locally) · [Services](#local-services) · [Checks](#checks)

## Project at a glance

| Area | Technology or responsibility |
| --- | --- |
| API and agent orchestration | Python, FastAPI, LangGraph, LiteLLM |
| Web interface | Next.js, React, TypeScript |
| Retrieval and persistence | PostgreSQL, pgvector, Gemini embeddings |
| Image search | OpenCLIP |
| Interoperability | MCP over `stdio`; A2A over local HTTP/JSON-RPC |

## What it does

- Searches restaurant records and OSM restaurant metadata.
- Retrieves indexed documents and can generate answers grounded in retrieved passages.
- Searches indexed restaurant photos with OpenCLIP; photo attribution and restaurant association are preserved.
- Exposes restaurant and document capabilities through backend endpoints, an Agent with Tools, and a local MCP server.
- Supports A2A delegation from the Itinerary Planner Agent to Restaurant Discovery and explicit restaurant comparisons from Restaurant Search to the Review & Recommendation Agent.
- Compares candidates against stated preferences using supplied OSM fields; indexed documents are not reliably associated with restaurants as reviews.
- Includes scripts for importing restaurant data, documents, and images.

> [!NOTE]
> Provider-backed requests depend on locally configured data and credentials. Similarity is a retrieval signal, not a confidence score; the Review Agent does not claim that the current document corpus contains restaurant-linked reviews.

Search results and indexed data depend on the configured database and the imports performed locally. Model-backed requests require provider configuration and may incur provider charges.

## Architecture at a glance

```mermaid
flowchart LR
    Browser["Browser"] --> Next["Next.js frontend"]
    Next --> Proxy["Next.js search route"]
    Proxy --> API["FastAPI backend"]
    API --> Agent["Restaurant Search Agent"]
    Agent --> Tools["Registered Tools"]
    Tools --> Discovery["Restaurant Discovery"]
    Agent -->|"A2A for explicit comparisons"| Review["Review & Recommendation A2A server"]
    Review -->|"search_documents operation"| Knowledge
    API --> Knowledge["Knowledge / RAG"]
    Discovery --> DB["PostgreSQL + pgvector"]
    Knowledge --> DB
    Agent --> Models["Configured model providers"]
    Knowledge --> Models
    Ingestion["Ingestion scripts"] --> DB
    Sources["OSM, Wikimedia Commons, local documents and images"] --> Ingestion
    Planner["Itinerary Planner Agent"] -->|"A2A"| DiscoveryServer["Restaurant Discovery A2A server"]
    DiscoveryServer --> Agent
    MCP["Local MCP server (stdio)"] --> Tools
```

The backend is a modular monolith: HTTP, Agents, Workflows, MCP, and A2A are distinct entry points that reuse application capabilities where appropriate. See the [backend architecture guide](backend/ARCHITECTURE.md) for the verified module boundaries and detailed process diagrams.

## Repository map

| Path | Purpose |
| --- | --- |
| [`backend/`](backend/) | FastAPI API, application modules, Agents, Tools, Workflows, MCP/A2A servers, ingestion and evaluation scripts. |
| [`frontend/`](frontend/) | Next.js and React interface for restaurant-search chat. |
| [`infrastructure/docker/`](infrastructure/docker/) | PostgreSQL with pgvector, optional Phoenix, and a disposable integration-test database. |

## Run locally

### 1. Start PostgreSQL

Follow the [infrastructure guide](infrastructure/docker/README.md) to configure the local Compose environment and start PostgreSQL with pgvector. The database container does not run application migrations.

### 2. Start the backend

From `backend`, create a virtual environment, install the development requirements, copy `.env.example` to `.env`, and set the database URL and model-provider credentials needed by the features you plan to use. Then apply migrations and start the API:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env before starting model-backed requests.
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The API listens on `http://127.0.0.1:8000`; interactive API documentation is at [`/docs`](http://127.0.0.1:8000/docs). See the [backend README](backend/README.md) for configuration, ingestion, tests, MCP, A2A, and feature-specific instructions.

For explicit comparison requests, start the Review & Recommendation A2A
service in another terminal from `backend`:

```powershell
.\.venv\Scripts\python.exe -m app.interfaces.a2a.review_recommendation_server
```

It listens at `http://127.0.0.1:8003`. The main API calls it through
`REVIEW_RECOMMENDATION_AGENT_URL`, which defaults to that address. Real
provider-backed requests also require the database and model configuration
described in the backend guide.

### 3. Start the frontend

In a separate terminal, from `frontend`:

```powershell
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

Open `http://localhost:3000`. The frontend defaults to the backend at `http://127.0.0.1:8000`; configure `BACKEND_API_URL` in `frontend/.env.local` if it runs elsewhere. See the [frontend README](frontend/README.md) for development and production commands.

## Local services

| Service | Default address | Notes |
| --- | --- | --- |
| Next.js frontend | `http://localhost:3000` | Restaurant-search chat UI. |
| FastAPI backend | `http://127.0.0.1:8000` | HTTP API and OpenAPI docs at `/docs`. |
| PostgreSQL | `localhost:5432` | PostgreSQL 16 with pgvector; started through Docker Compose. |
| Restaurant Discovery A2A | `http://127.0.0.1:8001` | Optional standalone local agent server. |
| Itinerary Planner A2A | `http://127.0.0.1:8002` | Optional; delegates to Restaurant Discovery. |
| Review & Recommendation A2A | `http://127.0.0.1:8003` | Optional; called by Restaurant Search for explicit comparisons. |
| Phoenix | `http://127.0.0.1:6006` | Optional local tracing collector; see the infrastructure and backend guides. |

The MCP server uses local `stdio` transport and does not listen on an HTTP port. The A2A servers are intended for local development and are not configured for public deployment.

## Checks

From `backend`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

From `frontend`:

```powershell
npm run lint
npx tsc --noEmit --incremental false
npm run build
```

The frontend package does not define a separate test script. Some backend evaluations call external model providers; consult the backend guide before running evaluation commands.

## Configuration and data

- Keep credentials in local `.env` files. Do not commit provider API keys or database passwords.
- `backend/.env.example` and `frontend/.env.example` are templates; copy them to local environment files and update values as needed.
- Restaurant, document, and image search depend on what has been imported into the local database.
- OpenCLIP model weights are downloaded when image ingestion or search first requires them.
- Model-backed search, embedding, and answer-generation calls may incur provider charges.

## Project guides

- [Backend setup and feature guide](backend/README.md)
- [Backend architecture and process diagrams](backend/ARCHITECTURE.md)
- [Backend integration test setup](backend/tests/integration/README.md)
- [Frontend development guide](frontend/README.md)
- [Docker infrastructure guide](infrastructure/docker/README.md)
