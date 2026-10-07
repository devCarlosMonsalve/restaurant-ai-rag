import os
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://test:test@localhost/test",
)

from app.database import get_db
from app.main import app
from app.models.base import Base


@pytest.fixture
def sqlite_engine() -> Generator[Engine, None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def db_session(sqlite_engine: Engine) -> Generator[Session, None, None]:
    test_session = sessionmaker(bind=sqlite_engine, autoflush=False)
    with test_session() as session:
        yield session


@pytest.fixture
def client(sqlite_engine: Engine) -> Generator[TestClient, None, None]:
    test_session = sessionmaker(bind=sqlite_engine, autoflush=False)

    def override_get_db() -> Generator[Session, None, None]:
        with test_session() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
