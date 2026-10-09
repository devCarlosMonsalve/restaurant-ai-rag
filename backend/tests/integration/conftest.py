import os
from collections.abc import Generator
from pathlib import Path
import subprocess
import sys

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parents[2]
TEST_DATABASE_NAME = "restaurant_ai_integration_test"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


@pytest.fixture(scope="session")
def postgres_engine() -> Generator[Engine, None, None]:
    database_url = os.environ.get("POSTGRES_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip(
            "Set POSTGRES_TEST_DATABASE_URL to run PostgreSQL integration tests"
        )

    parsed_url = make_url(database_url)
    if (
        parsed_url.drivername != "postgresql+psycopg"
        or parsed_url.database != TEST_DATABASE_NAME
        or parsed_url.host not in LOOPBACK_HOSTS
    ):
        pytest.fail(
            "Integration tests require a loopback psycopg URL targeting "
            f"the dedicated {TEST_DATABASE_NAME!r} database"
        )

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            extension_version = connection.execute(
                text(
                    "SELECT extversion FROM pg_extension "
                    "WHERE extname = 'vector'"
                )
            ).scalar_one_or_none()
        if extension_version is None:
            pytest.fail(
                "The integration database is missing the pgvector extension"
            )

        migration_environment = os.environ.copy()
        migration_environment["DATABASE_URL"] = database_url
        subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-c",
                str(BACKEND_ROOT / "alembic.ini"),
                "upgrade",
                "head",
            ],
            cwd=BACKEND_ROOT,
            env=migration_environment,
            check=True,
            capture_output=True,
            text=True,
        )

        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def postgres_session(postgres_engine: Engine) -> Generator[Session, None, None]:
    connection = postgres_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, autoflush=False, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
