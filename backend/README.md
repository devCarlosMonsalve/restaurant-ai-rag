# Backend

## Text-to-image search with OpenCLIP

From the `backend` directory, apply database migrations and index either one
image or a directory of `.jpg`, `.jpeg`, `.png`, and `.webp` files:

```powershell
python -m alembic upgrade head
python ingest_images.py .\data\images
```

The first image ingestion downloads the OpenCLIP `ViT-B-32` pretrained weights
to the local Hugging Face cache. Inference uses the GPU when PyTorch detects
CUDA; otherwise it runs on CPU. Re-indexing a file at the same path updates its
existing database row.

Start the API and send a text query to `POST /images/search`:

```powershell
python -m uvicorn app.main:app --reload
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
