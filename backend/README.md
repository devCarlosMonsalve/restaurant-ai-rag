# Backend

## Text-to-image search with OpenCLIP

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

## Evaluate image search

The sample image set has 12 labeled queries in English and Spanish. After
indexing the images, run the evaluator from this directory:

```powershell
.\.venv\Scripts\python.exe evaluate_image_search.py --top-k 3
```

It reports Hit@1 and Hit@3 (or the selected `--top-k`) overall and per language.
The command exits with an error if any expected image is missing from the
database or fails to appear within the selected top-k results.
