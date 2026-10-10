import argparse
from pathlib import Path

from app.database import SessionLocal
from app.ingestion import ingest_txt_to_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk a UTF-8 .txt document, embed its chunks with Gemini, and store them in pgvector."
    )
    parser.add_argument("path", type=Path, help="Path to the .txt document")
    args = parser.parse_args()

    with SessionLocal() as session:
        document_id, chunk_count = ingest_txt_to_database(args.path, session)

    print(f"Stored {chunk_count} chunks for document {document_id}")


if __name__ == "__main__":
    main()
