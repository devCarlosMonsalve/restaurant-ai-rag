import pytest

from app.domain.knowledge.text import chunk_text, clean_text
from app.infrastructure.filesystem.text_documents import ingest_txt


def test_clean_text_normalizes_unicode_and_whitespace() -> None:
    text = " Cafe\u0301 \t con leche\r\n\r\n\r\nPan  "

    assert clean_text(text) == "Café con leche\n\nPan"


def test_ingest_txt_chunks_with_configured_overlap(tmp_path) -> None:
    text_file = tmp_path / "menu.txt"
    text_file.write_text(
        "uno dos tres cuatro cinco seis siete ocho nueve diez",
        encoding="utf-8",
    )

    chunks = ingest_txt(text_file, chunk_size_words=4, overlap_words=1)

    assert chunks == [
        "uno dos tres cuatro",
        "cuatro cinco seis siete",
        "siete ocho nueve diez",
    ]


def test_ingest_txt_rejects_other_file_types(tmp_path) -> None:
    text_file = tmp_path / "menu.md"
    text_file.write_text("contenido", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Only \.txt files are supported"):
        ingest_txt(text_file)


@pytest.mark.parametrize(
    ("chunk_size_words", "overlap_words"),
    [(0, 0), (4, -1), (4, 4)],
)
def test_chunk_text_rejects_invalid_window(
    chunk_size_words: int,
    overlap_words: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_text(
            "uno dos tres",
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
