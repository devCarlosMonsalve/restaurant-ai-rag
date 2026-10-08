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

This real-photo set contains 12 answerable Spanish and English queries for
restaurant entrances, interiors, and dishes, plus 6 negative cases. Its
Hit@1/Hit@3 metrics provide a retrieval baseline, while the negative similarity
range supports threshold calibration. Search results use exact text matches
against photo filenames and OSM restaurant metadata alongside the visual CLIP
ranking. The displayed CLIP similarity is not a confidence score.
When none of the top results match indexed text metadata, the web page labels
them as visually close suggestions instead of implying an exact match.

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

Evaluate the semantic restaurant ranking against known Madrid cuisines,
features, and one dish-name case. Environment queries are reported for manual
review because OSM does not provide reliable labels for atmosphere:

```powershell
.\.venv\Scripts\python.exe evaluate_restaurant_search.py --top-k 5
```

The evaluator reports Hit@1, Hit@3, MRR@K, and Precision@3 for feature cases.
It does not apply a similarity threshold.

To inspect a separate set of novel queries without using them as labeled
benchmark cases, run:

```powershell
.\.venv\Scripts\python.exe evaluate_restaurant_search.py --cases .\data\restaurant_evaluation\madrid_holdout_queries.json --top-k 5
```

This prints the top results with their OSM cuisine and feature attributes plus
their evidence status for manual acceptance review; it does not calculate
ranking metrics for this set.
