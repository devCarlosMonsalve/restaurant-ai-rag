from functools import lru_cache
from pathlib import Path
from typing import Callable

import open_clip
import torch
from PIL import Image


CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
IMAGE_EMBEDDING_DIMENSIONS = 512
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@lru_cache(maxsize=1)
def _load_clip() -> tuple[
    torch.nn.Module,
    Callable[[Image.Image], torch.Tensor],
    Callable[[list[str]], torch.Tensor],
    torch.device,
]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return model, preprocess, tokenizer, device


def _normalized_vector(embedding: torch.Tensor) -> list[float]:
    vector = torch.nn.functional.normalize(embedding.float(), p=2, dim=-1)
    values = vector.squeeze(0).cpu().tolist()
    if len(values) != IMAGE_EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"Expected a {IMAGE_EMBEDDING_DIMENSIONS}-dimensional CLIP vector, "
            f"got {len(values)} dimensions"
        )
    return [float(value) for value in values]


def embed_image(path: str | Path) -> list[float]:
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported image type '{image_path.suffix}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))}"
        )

    model, preprocess, _, device = _load_clip()
    with Image.open(image_path) as image:
        image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model.encode_image(image_input)

    return _normalized_vector(embedding)


def embed_text_for_image_search(text: str) -> list[float]:
    query = text.strip()
    if not query:
        raise ValueError("The image search query cannot be empty")

    model, _, tokenizer, device = _load_clip()
    tokens = tokenizer([query]).to(device)
    with torch.inference_mode():
        embedding = model.encode_text(tokens)

    return _normalized_vector(embedding)
