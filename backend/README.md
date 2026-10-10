# Backend

## Text-to-image search with OpenCLIP

Start the API from this directory, then open `http://127.0.0.1:8000` in a
browser to search imported Madrid restaurant photos and view their license and
attribution. The page uses the same `/images/search` endpoint as the API.

From the `backend` directory, apply database migrations and index either one
image or a directory of `.jpg`, `.jpeg`, `.png`, and `.webp` files:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m scripts.ingestion.ingest_images .\data\images
```

The first image ingestion downloads the OpenCLIP `ViT-B-32` pretrained weights
to the local Hugging Face cache. Inference uses the GPU when PyTorch detects
CUDA; otherwise it runs on CPU. Re-indexing a file at the same path updates its
existing database row.

Start the API and send a text query to `POST /images/search`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

```json
{
  "query": "pasta with tomato sauce",
  "top_k": 5
}
```

The response contains matching filenames, local file paths, and cosine
similarities. CLIP image and text vectors share a 512-dimensional space and are
stored in the PostgreSQL `image_embeddings` table. This endpoint returns paths,
not image file contents.
The optional `osm_places_only` request field filters results to real places
imported from OpenStreetMap; it defaults to `false` so the sample image
evaluation remains unchanged. The optional `city` field restricts results to
that exact imported city, and `cuisine` filters by an exact OSM cuisine tag.
The web page populates both selectors from the OSM places available in the
selected city; places without cuisine metadata remain available under
“Cualquier cocina”.

## Evaluate image search

The sample image set has 12 answerable queries and 12 out-of-corpus queries,
balanced between English and Spanish. The negative cases include visually
similar dishes and unrelated objects. After indexing the images, run the
evaluator from this directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_image_search --top-k 3
```

It reports Hit@1 and Hit@3 (or the selected `--top-k`) overall and per language.
It also prints positive and negative top-similarity ranges to help calibrate a
future rejection threshold; negative results are not filtered yet. The command
exits with an error if any expected image is missing from the database or fails
to appear within the selected top-k results. The sample evaluator excludes OSM
photos so its metrics remain specific to the illustration test corpus.

To evaluate retrieval against the imported Madrid photos instead, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_image_search --cases .\data\image_evaluation\madrid_cases.json --osm-places-only --top-k 3
```

This real-photo set contains 10 scored Spanish and English queries for
restaurant entrances, interiors, and dishes, 2 cases held out for manual label
review, and 6 out-of-corpus cases. Hit@1/Hit@K measures retrieval of the labeled
image or restaurant result; negative non-empty rate only describes how often
the current search returns something for an out-of-corpus query. It is **not**
an abstention metric because image search has no rejection threshold.
Search results use exact text matches against photo filenames and OSM
restaurant metadata alongside the visual CLIP ranking. The displayed CLIP
similarity is not a confidence score.
When none of the top results match indexed text metadata, the web page labels
them as visually close suggestions instead of implying an exact match.

The two legacy queries asking for Mestizo currently expect a result named
Xamach. They are printed for manual review and excluded from relevance scores
until the target OSM identity is verified. To audit candidate-specific photo
association for the three OSM places that exposed the cross-restaurant issue,
run:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_image_search --attribution --top-k 3 --report "$env:TEMP\restaurant-ai-photo-attribution.json"
```

This checks the exact OSM ID foreign key and `source_url` for every returned
photo, and that candidates with no photos in the index return no photos. It
does not establish that an associated image is visually relevant or that no
other images exist outside the index. The cases fail as stale if the indexed
place name or exact OSM URL no longer matches.

## Import Madrid places and Commons photos

The importer stores named Madrid restaurants from OpenStreetMap, including
places without photos. It only looks up a photo when OSM explicitly links to a
Wikimedia Commons `Category:` or `File:`. It downloads only JPEG, PNG, or WebP
images with CC0, public-domain, or CC BY 1.0–4.0 licenses; other licenses are
skipped. File-page URL, license, author attribution, and the OSM place
reference are stored alongside each embedding. Places without an indexed photo
appear as text-matched fichas without an image; no photo is guessed or
associated by name alone.

Apply the latest migration and run the limited import:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m scripts.ingestion.ingest_madrid_commons_images --photos-per-place 2
```

By default the import fetches all named restaurants returned by the Madrid
Overpass query. Use `--limit N` for a smaller test import.

The API exposes imported places at `GET /restaurants/osm`, including a
`has_photos` flag. Image search results
include Commons photo attribution and license links, plus OSM place attribution.
Display those attributions when showing the images. OSM data is available under
the ODbL; see [OpenStreetMap copyright](https://www.openstreetmap.org/copyright).

To search restaurant fichas without indexed photos semantically, apply the
latest migration and generate place embeddings with the configured Gemini
embedding model:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m scripts.ingestion.ingest_osm_place_embeddings
```

The embedding importer is resumable: it indexes places whose metadata changed
or whose vector is missing. `POST /restaurants/osm/search` uses those vectors
with optional city and cuisine filters. Its similarity score is independent of
the CLIP photo score. Explicitly tagged OSM attributes such as outdoor seating,
diet options, wheelchair access, air conditioning, and reservations are included
in the metadata text. Missing or negative tags are not treated as available
features. Free-text `wheelchair:description` notes are not used as positive
features. Per the
[OSM wheelchair tagging guide](https://wiki.openstreetmap.org/wiki/Key:wheelchair),
`wheelchair=yes` denotes step-free entry and rooms; the interface still advises
users to confirm that community-maintained data is current. Kosher certifier
and check-date tags are shown when OSM provides them, but are not independently
verified by the application. Kosher results are considered current only when
`check_date:diet:kosher` is no older than 365 days; missing, stale, and future
dates are not returned as verified results. The current Madrid OSM snapshot has no
`diet:kosher`-tagged restaurants, so those searches deliberately return no
verified matches instead of inferring certification.
For searches that explicitly request supported features, photo and no-photo
results are restricted to places with matching OSM tags. The semantic search
response includes `results`, `evidence_status`, and `evidence_message`; queries
for unsupported details such as a quiet atmosphere may still show suggestions,
but are marked as unverified.

## Tool layer

The application layer groups use cases by bounded context.
`app.restaurant_discovery` owns restaurant discovery domain policies,
application use cases and its PostgreSQL adapter. Its repository port separates
the use cases from pgvector/query persistence. The previous
`app.application.restaurant_discovery` and infrastructure import paths remain
compatibility facades.
Knowledge/RAG use cases live in `app.knowledge.application`: document retrieval
and answer generation are separate dependencies, and the RAG use case owns the
retrieval-to-answer coordination. Its ports are specific to Knowledge.
Document search and answer DTOs are owned by
`app.knowledge.application.contracts`; the legacy `app.schemas` module
re-exports the same classes for existing API, Tool, MCP, and client imports.
`app.knowledge.infrastructure.postgres` owns the SQLAlchemy/pgvector query,
while `app.knowledge.infrastructure.generation` owns the LangChain chain and
Gemini generation adapter.

The `app.tools` package exposes these use cases through thin Tool boundaries.
Their docstrings are the human- and LLM-facing descriptions. The explicit
registry in `app.tools.registry` publishes their input schemas and dispatches
validated calls to the existing functions.

| Tool | Use when | Inputs | Returns | Does not guarantee |
| --- | --- | --- | --- | --- |
| `search_restaurants` | The user wants restaurant candidates. | Natural-language `query`; optional `top_k`, `city`, and exact OSM `cuisine` tag. | Candidate fichas, OSM attributes and source, semantic similarity, `evidence_status`, and `evidence_message`. | A final personalized recommendation, complete/current OSM data, or confirmation of unsupported details. Similarity is not confidence. |
| `search_restaurant_photos` | The user asks for photos of restaurants or wants to inspect associated imagery. | Natural-language `query`; optional `top_k`, `city`, and exact OSM `cuisine` tag. | Indexed OSM-linked photos with available image source, license, attribution, restaurant metadata, and visual similarity. | That an image is current, depicts current conditions, or proves restaurant suitability. Similarity is not confidence. |
| `search_documents` | Raw excerpts are needed as evidence for inspection or further synthesis. | `query` and optional `top_k`. | Ranked indexed-document chunks with filename, chunk index, and similarity. | A generated answer, completeness, or freshness of the indexed corpus. |
| `answer_from_documents` | The user wants a direct answer grounded in indexed documents. | Natural-language `query` and optional `top_k`. | A generated RAG answer and the source chunks used. | Facts beyond the retrieved corpus or independent verification of source accuracy/currentness. |

`search_documents` returns raw retrieved excerpts; it does not generate an
answer. `answer_from_documents` performs retrieval plus grounded answer
generation. Restaurant Tools serve restaurant discovery and associated images,
not document Q&A. Photo search is fixed to the OSM-linked restaurant corpus;
the Tool does not expose `osm_places_only` as an Agent choice. The HTTP image
search endpoint retains its existing corpus-selection behavior.

The request schema validates query and `top_k` bounds. The host injects the
database session; it is not a semantic Tool input. `search_restaurants` also
supports optional city and cuisine filters; the other Tools have only the
filters shown above. Search similarity values are ranking signals, not
confidence scores. OSM feature evidence may be incomplete or stale; unsupported
details can remain unverified. Kosher evidence is subject to the documented
freshness policy but its certifier is not independently validated.

FastAPI routes, Tools, Agents, workflows, MCP, A2A, and CLI scripts are distinct
delivery/integration boundaries. The backend is a modular monolith organized
around functional contexts where the code supports a real boundary. Its
bounded-context map, current tree, and keep/move decisions for app modules,
tests, scripts, configuration, data, and migrations are documented in
[ARCHITECTURE.md](./ARCHITECTURE.md).

The basic restaurant catalog endpoints in `app.main` now delegate through the
`RestaurantCatalog` application port to a PostgreSQL adapter; the HTTP module
keeps route registration, schema declarations, and `get_db` dependency wiring.

The dependency-free policies cover restaurant-search evidence,
candidate-photo association, and document text preparation. Intent detection,
feature matching, and kosher freshness live in
`app.restaurant_discovery.domain.evidence`; query intent lives in
`app.restaurant_discovery.domain.query_intent`; exact-source photo association
lives in `app.restaurant_discovery.domain.photo_association`; and Unicode
normalization and chunking live in `app.knowledge.domain.text`.

PostgreSQL restaurant and image query implementations live under
`app.infrastructure.persistence.postgres`; the original `app.infrastructure.persistence.postgres.restaurant_queries`
and `app.infrastructure.persistence.postgres.image_queries` paths remain compatibility facades. The context-specific
PostgreSQL adapter is in `app.restaurant_discovery.infrastructure.postgres`.
Text and image ingestion use cases live in `app.ingestion.application`, with
legacy imports retained in the old paths; OSM ingestion delegates through
`app.ingestion.composition`. Knowledge retrieval and generation remain in
`app.knowledge`; filesystem, model, database, and external data clients remain
infrastructure. ORM records and API schemas are not
domain entities.

## Knowledge/RAG generation

The document-answer path composes a PostgreSQL document retriever and a
grounded-answer generator at the HTTP or Tool boundary. The application
retrieves chunks with the existing cosine-distance query, returns the existing
Spanish no-documents answer without invoking a model when retrieval is empty,
formats excerpts in retrieval order, invokes the generator, and returns all
retrieved chunks as sources. The generator receives only the prepared question
and excerpt context; it has no database dependency.

Generation uses `ChatPromptTemplate`, LCEL, `RunnableLambda`, and
`StrOutputParser`. The official
[`ChatGoogleGenerativeAI` integration](https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai)
uses the Google GenAI SDK's content-generation interface. The current RAG
provider call uses the Interactions API and explicitly sets `store=False`;
the LangChain chat adapter does not establish equivalent support for that
Interactions API parameter. The adapter is therefore not substituted and
`langchain-google-genai` is not added as a dependency. The existing Google
client remains inside `RunnableLambda` to preserve the model, request options,
and privacy behavior. See Google's
[Interactions API documentation](https://ai.google.dev/gemini-api/docs/interactions-overview).

Focused Knowledge/RAG tests, which use fake embeddings, sessions, and model
responses and do not call Gemini, can be run from `backend` with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\contexts\knowledge\test_document_search.py tests\unit\contexts\knowledge\test_rag.py tests\unit\application\test_application_use_cases.py tests\contracts\tools\test_retrieval_tools.py tests\evaluation\test_rag_evaluation.py tests\unit\infrastructure\test_observability.py tests\unit\domain\test_text_processing.py tests\contracts\http\test_api.py tests\contracts\mcp\test_mcp_server.py
```

## Restaurant search Agent

`POST /agents/restaurant-search` runs the first Agent using the existing
`gemini-3.8-flash` model through a local LiteLLM Router. A request-scoped
LangGraph `StateGraph` makes the orchestration explicit. Its `call_model` node
handles model turns, `execute_tools` validates and runs Tool calls,
`force_photo_search` enforces an explicit photo request if the model omitted
it, and `finalize` constructs the existing response. Conditional edges route
between these nodes; the graph state is discarded after each request and has
no checkpointer or persistent memory.

LiteLLM maps the `restaurant-search-agent` model alias to
`gemini/gemini-3.8-flash`. If `OPENAI_API_KEY` is configured, a failed Gemini
request falls back once to `openai/gpt-4.1-mini`; without that key, only Gemini
is configured. Retries and response caching are disabled, and each model
request has a 60-second timeout. When fallback is used, the same prompt,
conversation history, and tool results are sent to OpenAI. The existing Google
GenAI Tool schemas and conversation history are translated to LiteLLM's
chat-completion format; the Agent graph still owns tool execution and its call
limits.

When Phoenix tracing is enabled, model-call spans record the requested and
responding model, output-token limit, and provider-reported input, output, and
total token counts. Prompts, conversation messages, and model responses are not
added to these spans.

The routing and tool-cycle tests use synthetic LiteLLM responses and do not
contact Gemini or OpenAI:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\infrastructure\test_model_routing.py tests\unit\contexts\discovery\test_restaurant_search_agent.py
```

The database session is passed as graph context, never included in the state
sent to the model or in a Tool declaration. The existing four Tools remain
registered through `app.tools.registry`, and their arguments are validated
against the existing request schemas. Photo results omit local `image_path`
values before returning to the model. Embeddings and RAG answer generation
remain on the Google GenAI SDK; in particular, the RAG Interactions request
still uses `store=False`.

The Agent permits at most four Tool call attempts per request, including
invalid or failed calls, and at most six model turns. When photos are requested,
one Tool call is reserved for photo search. Tool errors are returned to the
model as explicit error results; model-route failures produce an HTTP 502
response instead of a fabricated answer. When photo search returns matches,
the API response includes a `photos` array alongside `answer`.
The response also includes a `restaurants` array containing the structured
candidates used by the Agent, with public source and attribution fields but no
database UUID. Its `similarity` values are retrieval ranking signals, not
confidence scores; the candidate array is empty when no restaurant Tool
candidate was retrieved.
Each photo has a safe API-relative `image_url` (for example,
`/images/files/commons-photo.jpg`) and available source, license, attribution,
and restaurant metadata. The local `image_path` is never sent to the model or
returned by the Agent endpoint. Clients can render each `image_url` using the
API base URL; Postman displays the URL as JSON rather than rendering the image.
When restaurant and photo search are both used, photos are searched by
candidate OSM ID and included only when the photo's `restaurant_source_url`
exactly matches the candidate's `source_url`; repeated Commons sources are
removed. Restaurant calls are executed before other Tools in a model turn, but
their results are returned to Gemini in the original call order. If no photos
remain associated with the candidates, the answer says so.

LangGraph's concrete benefit here is an explicit, testable state machine for
model/tool cycles, forced photo lookup, finalization, and turn-limit handling
instead of one opaque manual loop. It does not add a new retrieval or ranking
path, personalized preferences, or persistent conversation state. The separate
photo workflow and MCP server are not invoked or changed by this Agent graph.

There are no Tools for live reservation availability or current menu prices.
For requests that explicitly ask for those facts, the Agent response begins
with a limitation notice; it may still provide restaurant candidates, but does
not claim to have verified availability or current prices.

Evaluate the semantic restaurant ranking against known Madrid cuisines,
features, and one dish-name case. Environment queries are reported for manual
review because OSM does not provide reliable labels for atmosphere:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_restaurant_search --top-k 5
```

The evaluator reports Hit@1, Hit@3, MRR@K, Precision@3, explicit city/cuisine
filter compliance, unexpected empty queries, and the manually labeled
no-evidence case (`cocina kosher`). The 15 answerable ranking labels remain
separate from that empty-result case and from the manual-review queries. It does
not apply a similarity threshold. Each query requires a Gemini embedding API
request.

To save a machine-readable run record:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_restaurant_search --top-k 5 --report "$env:TEMP\restaurant-ai-restaurants.json"
```

## Evaluate document retrieval and RAG answers

The default command evaluates retrieval only. It checks labeled source files,
including one question requiring chunks from two documents, and reports
Hit@K, source Recall@K, chunk Precision@K, and MRR@K:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_rag --top-k 3 --report "$env:TEMP\restaurant-ai-rag.json"
```

The retrieval command makes one Gemini embedding request per query. Answer
fidelity, reference correctness, and abstention are not inferred from the
retrieval metrics. To generate answers for manual comparison against the
reference facts and returned excerpts, explicitly opt in:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_rag --top-k 3 --generate-answers --report "$env:TEMP\restaurant-ai-rag-answers.json"
```

`--generate-answers` can make up to one additional Gemini generation request
per case with retrieved chunks. Review the answer against the cited chunk and
reference fact; the report leaves those human judgments unset rather than
presenting an automatic faithfulness score. `--index-missing` is a separate
opt-in that embeds and commits the evaluation documents to the configured
database; use it only against a disposable evaluation database.

When `--report` is supplied, each evaluator fingerprints the case file and the
complete indexed rows it uses (including stored embeddings), and records the
Git commit/dirty state, UTC run time, model, dimensions, ranking configuration,
and result metrics. Do not publish reports if their case text or retrieved
document chunks are sensitive. A changed corpus fingerprint means the results
are not directly comparable; missing OSM IDs, changed exact URLs, or missing
labeled document sources fail validation and require the affected labels to be
reviewed against their source data.

Audit the current Madrid catalog's evidence coverage, embeddings, photo coverage,
kosher freshness, and saved manual holdout judgments with:

```powershell
.\.venv\Scripts\python.exe -m scripts.maintenance.audit_restaurant_coverage
.\.venv\Scripts\python.exe -m scripts.maintenance.audit_restaurant_coverage --format json
```

Use `--as-of YYYY-MM-DD` to reproduce freshness counts for a specific date.
Feature percentages use all named restaurants in the selected city as the
denominator. Kosher freshness follows the same 365-day rule as search. The
holdout summary reports stored human judgments only; it is not an automatic
ranking score.

To inspect a separate set of novel queries without using them as labeled
benchmark cases, run:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluation.evaluate_restaurant_search --cases .\data\restaurant_evaluation\madrid_holdout_queries.json --top-k 5
```

This prints the top results with their OSM cuisine and feature attributes plus
their evidence status for manual acceptance review; it does not calculate
ranking metrics for this set.

## LangChain in RAG answer generation

Document retrieval remains the existing SQLAlchemy/pgvector query. The
generation step composes a `ChatPromptTemplate`, a `RunnableLambda` adapter for
the existing Gemini SDK call, and `StrOutputParser` as an LCEL chain:

```text
question + retrieved excerpts
  -> ChatPromptTemplate
  -> RunnableLambda (_generate_with_gemini)
  -> google-genai interactions.create
  -> AIMessage
  -> StrOutputParser
  -> answer string
```

The adapter intentionally keeps `google-genai` and the current
`interactions.create(..., store=False)` request. This preserves the existing
provider request and its explicit storage setting instead of assuming that a
different chat-model adapter has identical privacy behavior. The prompt text,
Gemini model, temperature, token limit, no-document path, and public RAG
answer and source fields remain unchanged. The HTTP `/documents/ask` response
can optionally include a separate `photos` field by setting
`include_photos: true` in the request. Photo retrieval reuses the original
question and `top_k`, and is limited to OSM-linked photos served by the existing
image-file endpoint. Each photo has an API-relative `image_url`, source/license
and available attribution metadata; local image paths and database IDs are not
returned. Photos are never passed to the text generator and are not document
citations or evidence for the generated answer. The `answer_from_documents`
Tool retains its answer-and-sources contract. OpenCLIP loading and retrieval
costs are incurred only when `include_photos` is enabled. The chain currently
parses plain text; it
does not independently validate citations or change the retrieved sources.

`langchain-core` is a direct dependency because the application imports its
prompt, runnable, message, and parser APIs. The normal tests use a fake Gemini
client and make no provider calls:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\contexts\knowledge\test_rag.py
```

The existing `GEMINI_API_KEY` setting still configures the provider through
the backend environment or its local `.env`; no tracing service is required
unless optional Phoenix tracing is enabled. The API continues to expose this
generation through `POST /documents/ask`. The official
`ChatGoogleGenerativeAI` adapter was not
used because equivalence with the current explicit `store=False` request was
not established. This synchronous chain does not add streaming, retries,
structured output, or citation validation.

This integration adds LCEL composition to generation only. It does not add a
LangChain vector-store retriever, change retrieval behavior, or run the
real-provider RAG benchmark. A provider-adapter migration can be considered
later if its privacy and request semantics are verified.

## Optional local tracing with Phoenix

Phoenix tracing is opt-in and disabled by default. Start the separate local
Phoenix service from the repository root:

```powershell
docker compose -f infrastructure/docker/docker-compose.phoenix.yml up -d
```

Open `http://127.0.0.1:6006` for the Phoenix UI. The compose file binds the UI
and OTLP HTTP receiver only to loopback and stores Phoenix data in its own
named Docker volume; it does not change the PostgreSQL compose service.

Set these values in `backend/.env` to enable export from the API:

```dotenv
PHOENIX_TRACING_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=http://127.0.0.1:6006/v1/traces
PHOENIX_PROJECT_NAME=restaurant-ai-rag
```

Run the API as usual from `backend`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The application uses OpenTelemetry OTLP over HTTP. Spans cover document and
restaurant retrieval, RAG generation, photo retrieval, Agent model/tool calls,
and the photo workflow. Only operational metadata is recorded, such as model
and tool names, limits, result counts, and exception types. Queries, prompts,
retrieved content, tool arguments/results, secrets, database identifiers, and
local image paths are deliberately excluded. Automatic SDK instrumentation is
not enabled, and the RAG LCEL, Agent, and photo-workflow graph calls explicitly
disable automatic LangSmith tracing even if it is enabled elsewhere in the
process. The LangSmith context helper is used only to suppress that tracing;
it does not export runs. FastAPI and the standalone MCP process flush and shut
down the exporter when they stop. The MCP process uses the same `PHOENIX_*`
settings from its environment or the backend's local `.env`. To disable
tracing, set `PHOENIX_TRACING_ENABLED=false`; Phoenix can
be stopped independently with:

```powershell
docker compose -f infrastructure/docker/docker-compose.phoenix.yml down
```

## Retrieval baseline — FROZEN

As of 2026-10-08, the Madrid restaurant retrieval baseline contains 4,838 OSM
restaurants with 768-dimensional Gemini `gemini-embedding-2` embeddings.
Semantic restaurant search compares query and restaurant vectors by cosine
distance, applies optional exact city/cuisine filters, and returns only
restaurants without indexed photos. Requests for supported characteristics
add OSM evidence filters; unsupported atmosphere requests remain suggestions
and are marked unverified.

Image retrieval uses OpenCLIP `ViT-B-32` with pretrained weights
`laion2b_s34b_b79k`. Image and text vectors are normalized to 512 dimensions
and ranked in the shared CLIP space. For OSM photo searches, exact token matches
in place and image metadata are used as a ranking tie-break ahead of CLIP
cosine distance. City, cuisine, and supported feature evidence filters are
applied to OSM-linked photos. Image CLIP similarity and restaurant semantic
similarity are independent scores, not confidence values.

The frozen curated restaurant benchmark baseline is:

| Metric | Baseline |
| --- | ---: |
| Hit@1 | 15/15 |
| Hit@3 | 15/15 |
| MRR@5 | 1.000 |
| Precision@3 | 93% |

This is a small, curated benchmark and is not a general performance guarantee.
Its manual holdout judgments are separate and are not included as automatic
ranking metrics. The baseline test suite had 66 passing tests; this frozen
implementation also includes regression coverage for OSM-tagged live music.

This phase is **FROZEN**: do not change embedding models, CLIP models,
preprocessing, corpus, or ranking to pursue marginal benchmark gains. Only
verified correctness fixes that preserve the stated baseline are in scope.
Potential retrieval optimizations belong in a future phase and must be
evaluated separately; they are not part of this baseline.

## Restaurant photo workflow

`POST /workflows/restaurant-photo-search` coordinates the existing restaurant
and photo capabilities with LangGraph. It searches up to `candidate_limit`
restaurant candidates, including places with indexed photos, then performs a
targeted OSM photo search for each candidate. Photos are included only when
`restaurant_source_url` exactly matches the candidate's `source_url`;
unassociated results are not matched by name. The response separates
restaurants with returned photos from candidates without returned photos and
does not claim that an empty photo search proves no photos exist.

The request accepts `query`, optional `city` and exact OSM `cuisine`, and
`candidate_limit` (default 5, maximum 20). `photos_per_candidate` defaults to
1 and can be set from 1 to 5. The workflow injects the database session through
its runtime context rather than including it in workflow state or Tool inputs.
Tool failures return HTTP 502; they are not treated as empty search results.
The workflow-specific option changes only candidate eligibility. The general
restaurant-search Tool still excludes places with indexed photos by default;
only this Workflow opts into including them. Embedding generation, semantic
ranking, and the existing photo Tool contract are unchanged.

## MCP server

The standalone MCP server uses the official Python SDK v2 and the local
`stdio` transport. It reuses the existing four Tools and opens/closes a
SQLAlchemy session for each tool call. It does not add an HTTP listener or
change the FastAPI endpoints. `search_restaurants` and
`search_restaurant_photos` accept optional city and cuisine filters;
`search_restaurant_photos` keeps the existing OSM-only restriction internal.
Responses omit internal database IDs and local image paths while retaining
public source and attribution URLs.

From the `backend` directory, run it with:

```powershell
.\.venv\Scripts\python.exe -m app.interfaces.mcp.server
```

Configure an MCP-compatible local client to launch the same command using
`stdio`. For example, a VS Code MCP configuration can use:

```json
{
  "servers": {
    "restaurant-ai-rag": {
      "type": "stdio",
      "command": "C:\\path\\to\\restaurant-ai-rag\\backend\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.interfaces.mcp.server"],
      "cwd": "C:\\path\\to\\restaurant-ai-rag\\backend"
    }
  }
}
```

Replace the example paths with the local checkout path. Provide the same
database and provider environment variables required by the relevant Tools
through the client environment or a local, untracked environment file; do not
put credentials in the checked-in MCP configuration. Restaurant and photo
search need the configured database and embedding provider; document search
needs the database and document embeddings; `answer_from_documents` also needs
the configured generation provider. Local `stdio` is suitable for a desktop
client. A remotely deployed server should use Streamable HTTP with explicit
authentication and deployment-specific resource lifecycle management; that
transport is not enabled here.

The protocol and tool tests use the SDK's in-memory client and mocked Tool
handlers, so they do not require PostgreSQL or external providers:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\contracts\mcp\test_mcp_server.py
```

## A2A server

The standalone A2A server exposes the existing Restaurant Search Agent to
independent agents. A separate Itinerary Planner Agent can delegate restaurant
discovery and receive structured evidence, while retaining responsibility for
the dining draft. Both services run as independent processes; neither changes
the existing FastAPI or MCP contracts.

Start the server from `backend`:

```powershell
.\.venv\Scripts\python.exe -m app.interfaces.a2a.restaurant_discovery_server
```

It listens on loopback at `http://127.0.0.1:8001`. The Agent Card is available at
`http://127.0.0.1:8001/.well-known/agent-card.json`, and JSON-RPC requests use
the root path. A client can discover the card and send a text request with the
official SDK:

```python
import asyncio

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest
import httpx


async def main():
    async with httpx.AsyncClient() as http_client:
        card = await A2ACardResolver(
            http_client,
            "http://127.0.0.1:8001",
        ).get_agent_card()

    client = await create_client(
        agent=card,
        client_config=ClientConfig(streaming=False),
    )
    try:
        request = SendMessageRequest(
            message=new_text_message(
                "Find vegetarian restaurants in Madrid",
                role=Role.ROLE_USER,
            )
        )
        async for response in client.send_message(request):
            print(response)
    finally:
        await client.close()


asyncio.run(main())
```

The task artifact contains the existing Agent response as an
`application/json` data part. Any relative `image_url` values in that response
are resolved against the FastAPI service, not the A2A port. Requests must be
plain text between 1 and 2,000 characters. The task store is in memory, so
history is lost when the server restarts; cancellation is not supported.
Incoming message history is retained in that store for the task lifetime, but
Phoenix spans exclude the request text and result content. The server binds
only to loopback and has no authentication, so it is for local development,
not remote deployment. A valid search uses the same database and Gemini
configuration as the existing Agent.

The A2A protocol tests use an in-memory ASGI transport and a mocked Agent
result; they do not require PostgreSQL or call Gemini:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\contracts\a2a\test_a2a_server.py
```

### Itinerary planner A2A client

`app.itinerary_planning.infrastructure.a2a.restaurant_discovery_client` is a separate client-side delegation component:
it discovers the restaurant Agent Card, sends a text task, checks the task
state, and validates the `restaurant_discovery_result` JSON artifact against
the existing response schema. It returns restaurant evidence to its caller;
it does not invent or generate itinerary details.

With the A2A server running, a real request can be sent from a second terminal:

```powershell
.\.venv\Scripts\python.exe -m scripts.a2a.restaurant_discovery_client "Find vegetarian restaurants in Madrid"
```

This command invokes the existing restaurant Agent and therefore needs its
database and Gemini configuration; a successful live call may incur provider
costs. The client tests instead use the A2A server through an in-memory
transport and replace its Agent execution with a synthetic result:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\contracts\a2a\test_itinerary_planner_client.py
```

The separate `app.itinerary_planning.application.planner` module builds a deterministic dining draft
from those structured candidates. Give it the trip length explicitly:

```powershell
.\.venv\Scripts\python.exe -m scripts.a2a.plan_itinerary --days 3 "Find vegetarian restaurants in Madrid"
```

The MVP assigns at most one candidate per day in the Agent's retrieval order,
never repeats a restaurant, and returns extra candidates as alternatives.
Days without enough evidence remain unfilled. Day numbers are placeholders;
the planner does not infer travel routes, opening hours, availability, or
reservations. Its in-memory tests exercise the A2A exchange with synthetic
candidates and do not call Gemini or PostgreSQL:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\contexts\planning\test_itinerary_planner.py
```

### Itinerary Planner A2A service

To let other A2A clients delegate the full dining-draft task, start the
Itinerary Planner Agent in a second terminal while the Restaurant Discovery
Agent is running:

```powershell
.\.venv\Scripts\python.exe -m app.interfaces.a2a.itinerary_planner_server
```

It listens on `http://127.0.0.1:8002`; its Agent Card is at
`http://127.0.0.1:8002/.well-known/agent-card.json`. Send one
`application/json` data part containing exactly `query` and `day_count`, for
example `{"query":"vegetarian restaurants in Madrid","day_count":3}`. The
completed task returns an `itinerary_dining_draft` JSON artifact. The planner
delegates to the Restaurant Discovery Agent on port 8001, so a valid task uses
the same database and Gemini configuration and may incur provider costs. The
planner's task store is in memory, and cancellation is not supported. It binds
to loopback without authentication and is intended for local development.

An A2A client can discover the planner and send the structured request with the
official SDK:

```python
import asyncio

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import new_data_message
from a2a.types import Role, SendMessageRequest


async def main():
    async with httpx.AsyncClient() as http_client:
        card = await A2ACardResolver(
            http_client,
            "http://127.0.0.1:8002",
        ).get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=False,
                httpx_client=http_client,
                accepted_output_modes=["application/json"],
            ),
        )
        try:
            request = SendMessageRequest(
                message=new_data_message(
                    {
                        "query": "vegetarian restaurants in Madrid",
                        "day_count": 3,
                    },
                    media_type="application/json",
                    role=Role.ROLE_USER,
                )
            )
            async for response in client.send_message(request):
                print(response.task)
        finally:
            await client.close()


asyncio.run(main())
```

The end-to-end A2A test connects both services through in-memory transports and
uses synthetic restaurant candidates; it does not require PostgreSQL or call
Gemini:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\contracts\a2a\test_itinerary_planner_server.py
```

For a terminal client, with both agents running, use:

```powershell
.\.venv\Scripts\python.exe -m scripts.a2a.itinerary_planner_agent_client --days 3 "vegetarian restaurants in Madrid"
```

The client discovers the Itinerary Planner Agent, sends the JSON request,
checks the task state, and validates the returned draft artifact. This live
command delegates to restaurant discovery and therefore uses its database and
Gemini configuration. The client integration tests run the complete chain
with synthetic candidates:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\contracts\a2a\test_itinerary_planner_agent_client.py
```
