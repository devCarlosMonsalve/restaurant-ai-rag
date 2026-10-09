# Backend

## Text-to-image search with OpenCLIP

Start the API from this directory, then open `http://127.0.0.1:8000` in a
browser to search imported Madrid restaurant photos and view their license and
attribution. The page uses the same `/images/search` endpoint as the API.

From the `backend` directory, apply database migrations and index either one
image or a directory of `.jpg`, `.jpeg`, `.png`, and `.webp` files:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe ingest_images.py .\data\images
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
.\.venv\Scripts\python.exe evaluate_image_search.py --top-k 3
```

It reports Hit@1 and Hit@3 (or the selected `--top-k`) overall and per language.
It also prints positive and negative top-similarity ranges to help calibrate a
future rejection threshold; negative results are not filtered yet. The command
exits with an error if any expected image is missing from the database or fails
to appear within the selected top-k results. The sample evaluator excludes OSM
photos so its metrics remain specific to the illustration test corpus.

To evaluate retrieval against the imported Madrid photos instead, run:

```powershell
.\.venv\Scripts\python.exe evaluate_image_search.py --cases .\data\image_evaluation\madrid_cases.json --osm-places-only --top-k 3
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
.\.venv\Scripts\python.exe evaluate_image_search.py --attribution --top-k 3 --report "$env:TEMP\restaurant-ai-photo-attribution.json"
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
.\.venv\Scripts\python.exe ingest_madrid_commons_images.py --photos-per-place 2
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
.\.venv\Scripts\python.exe ingest_osm_place_embeddings.py
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

The application layer groups use cases by bounded context. The
`app.application.restaurant_discovery` module coordinates restaurant and photo
search; `app.application.knowledge` coordinates document search and grounded
answers. Each use case accepts an existing request schema and database session,
then delegates to the frozen service without changing its algorithm or result.

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

FastAPI routes in `app.main` and the Tool adapters are separate entry points
that call the same application use cases. The existing domain policies,
SQLAlchemy models, database access, embedding providers, and external-data
clients remain in their current modules; this incremental structure does not
add empty domain/infrastructure packages, repositories, or interfaces.

This is a modular-monolith boundary, not a full dependency-inverted DDD
reorganization. It leaves the current session-based frozen services in place
until there is a concrete need to separate their persistence dependencies.

## Restaurant search Agent

`POST /agents/restaurant-search` runs the first Agent using the existing
`gemini-3.8-flash` model and the `google-genai` SDK. It manually registers the
four Tools through `app.tools.registry`, validates every function-call
argument against the existing request schemas, and executes calls sequentially.
The database session is injected by the API host and is never included in a
function declaration or sent as a model argument. Photo results omit local
`image_path` values before returning to the model.

The Agent permits at most four Tool call attempts per request, including
invalid or failed calls. Automatic SDK function execution is disabled. Tool
errors are returned to the model as explicit error results; Gemini/API failures
produce an HTTP 502 response instead of a fabricated answer. When photo search
returns matches, the API response includes a `photos` array alongside `answer`.
Each photo has a safe API-relative `image_url` (for example,
`/images/files/commons-photo.jpg`) and available source, license, attribution,
and restaurant metadata. The local `image_path` is never sent to the model or
returned by the Agent endpoint. Clients can render each `image_url` using the
API base URL; Postman displays the URL as JSON rather than rendering the image.
When restaurant and photo search are both used, photos are restricted to names
present in the restaurant candidates and repeated Commons sources are removed.
For explicit photo requests, the host reserves a Tool call and performs the
photo search if the model omits it. If no photos remain after matching them to
the candidates, the answer says that no associated indexed photos were found.

The Agent does not add a new retrieval or ranking path, personalized
preferences, workflows, or persistent conversation state. It does not
introduce LangGraph or MCP.

There are no Tools for live reservation availability or current menu prices.
For requests that explicitly ask for those facts, the Agent response begins
with a limitation notice; it may still provide restaurant candidates, but does
not claim to have verified availability or current prices.

Evaluate the semantic restaurant ranking against known Madrid cuisines,
features, and one dish-name case. Environment queries are reported for manual
review because OSM does not provide reliable labels for atmosphere:

```powershell
.\.venv\Scripts\python.exe evaluate_restaurant_search.py --top-k 5
```

The evaluator reports Hit@1, Hit@3, MRR@K, Precision@3, explicit city/cuisine
filter compliance, unexpected empty queries, and the manually labeled
no-evidence case (`cocina kosher`). The 15 answerable ranking labels remain
separate from that empty-result case and from the manual-review queries. It does
not apply a similarity threshold. Each query requires a Gemini embedding API
request.

To save a machine-readable run record:

```powershell
.\.venv\Scripts\python.exe evaluate_restaurant_search.py --top-k 5 --report "$env:TEMP\restaurant-ai-restaurants.json"
```

## Evaluate document retrieval and RAG answers

The default command evaluates retrieval only. It checks labeled source files,
including one question requiring chunks from two documents, and reports
Hit@K, source Recall@K, chunk Precision@K, and MRR@K:

```powershell
.\.venv\Scripts\python.exe evaluate_rag.py --top-k 3 --report "$env:TEMP\restaurant-ai-rag.json"
```

The retrieval command makes one Gemini embedding request per query. Answer
fidelity, reference correctness, and abstention are not inferred from the
retrieval metrics. To generate answers for manual comparison against the
reference facts and returned excerpts, explicitly opt in:

```powershell
.\.venv\Scripts\python.exe evaluate_rag.py --top-k 3 --generate-answers --report "$env:TEMP\restaurant-ai-rag-answers.json"
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
.\.venv\Scripts\python.exe audit_restaurant_coverage.py
.\.venv\Scripts\python.exe audit_restaurant_coverage.py --format json
```

Use `--as-of YYYY-MM-DD` to reproduce freshness counts for a specific date.
Feature percentages use all named restaurants in the selected city as the
denominator. Kosher freshness follows the same 365-day rule as search. The
holdout summary reports stored human judgments only; it is not an automatic
ranking score.

To inspect a separate set of novel queries without using them as labeled
benchmark cases, run:

```powershell
.\.venv\Scripts\python.exe evaluate_restaurant_search.py --cases .\data\restaurant_evaluation\madrid_holdout_queries.json --top-k 5
```

This prints the top results with their OSM cuisine and feature attributes plus
their evidence status for manual acceptance review; it does not calculate
ranking metrics for this set.

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
.\.venv\Scripts\python.exe -m app.mcp_server
```

Configure an MCP-compatible local client to launch the same command using
`stdio`. For example, a VS Code MCP configuration can use:

```json
{
  "servers": {
    "restaurant-ai-rag": {
      "type": "stdio",
      "command": "C:\\path\\to\\restaurant-ai-rag\\backend\\.venv\\Scripts\\python.exe",
      "args": ["-m", "app.mcp_server"],
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
.\.venv\Scripts\python.exe -m pytest tests\test_mcp_server.py
```
