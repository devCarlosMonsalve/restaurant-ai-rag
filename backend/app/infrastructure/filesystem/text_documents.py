from pathlib import Path

from app.domain.knowledge.text import (
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_OVERLAP_WORDS,
    chunk_text,
    clean_text,
)


def ingest_txt(
    path: str | Path,
    *,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    text_file = Path(path)
    if text_file.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported")

    text = clean_text(text_file.read_text(encoding="utf-8-sig"))
    return chunk_text(
        text,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )
