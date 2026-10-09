import re
import unicodedata

DEFAULT_CHUNK_SIZE_WORDS = 500
DEFAULT_OVERLAP_WORDS = 75


def clean_text(text: str) -> str:
    normalized_text = unicodedata.normalize("NFC", text)
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [
        re.sub(r"[ \t\f\v]+", " ", line).strip()
        for line in normalized_text.split("\n")
    ]
    cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned_text.strip()


def chunk_text(
    text: str,
    *,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words

    return chunks
