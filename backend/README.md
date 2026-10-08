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
evaluation remains unchanged.

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

This real-photo set contains known Spanish and English queries for restaurant
entrances, interiors, and dishes; its Hit@1/Hit@3 metrics provide a baseline
for retrieval. Search results use exact text matches against photo filenames
and OSM restaurant metadata alongside the visual CLIP ranking. The displayed
CLIP similarity is not a confidence score.

## Import Madrid places and Commons photos

The pilot uses Madrid restaurant entries that explicitly link to a Wikimedia
Commons `Category:` or `File:` in OpenStreetMap. It downloads only JPEG, PNG, or
WebP images with CC0, public-domain, or CC BY 1.0–4.0 licenses; other licenses
are skipped. File-page URL, license, author attribution, and the OSM place
reference are stored alongside each embedding.

Apply the latest migration and run the limited import:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe ingest_madrid_commons_images.py --limit 5 --photos-per-place 2
```

The API exposes imported places at `GET /restaurants/osm`. Image search results
include Commons photo attribution and license links, plus OSM place attribution.
Display those attributions when showing the images. OSM data is available under
the ODbL; see [OpenStreetMap copyright](https://www.openstreetmap.org/copyright).
