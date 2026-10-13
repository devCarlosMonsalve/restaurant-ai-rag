# Backend integration tests

Integration tests are opt-in and split into database-backed PostgreSQL tests
and a local A2A process test. The default backend test command excludes both.

| Test group | External requirement | Data behavior |
| --- | --- | --- |
| PostgreSQL/pgvector | Dedicated loopback test database | Applies migrations and rolls back inserted rows |
| A2A process | Local port 8003 available | Synthetic inputs; no database connection or provider call |

## PostgreSQL and pgvector

These tests use a dedicated local PostgreSQL database with the real pgvector
extension. They run Alembic migrations against that database and roll back
each test's inserted rows. They do not use the development Compose service or
its persistent volume.

From the repository root, start the isolated, non-persistent test service:

    docker compose -p restaurant-ai-integration -f infrastructure/docker/docker-compose.integration.yml up --wait -d

In PowerShell, from `backend`, point the integration suite only at its guarded
test database and run it:

    $env:POSTGRES_TEST_DATABASE_URL = "postgresql+psycopg://restaurant_ai_test:restaurant_ai_test@127.0.0.1:55432/restaurant_ai_integration_test"
    .\.venv\Scripts\python.exe -m pytest tests\integration\test_postgres_queries.py -m integration -q

The fixture rejects URLs that are not loopback `postgresql+psycopg` connections
to `restaurant_ai_integration_test`. It also requires the `vector` extension
before applying the Alembic migrations.

## Review Agent A2A process test

The local process test starts the Review & Recommendation server on loopback
port 8003 and drives the Restaurant Search Agent with synthetic model and
search results. It uses a deterministic server executor and an in-memory
SQLite URL to initialize the database adapter without opening a connection.
It does not connect to PostgreSQL, call a model provider, or modify imported
data. Port 8003 must be free.

From `backend`, run it explicitly:

    .\.venv\Scripts\python.exe -m pytest tests\integration\a2a\test_review_recommendation_process.py -m integration -q

## Run all opt-in integration tests

To run both groups, start the isolated PostgreSQL service, set
`POSTGRES_TEST_DATABASE_URL` as above, ensure port 8003 is free, and run:

    .\.venv\Scripts\python.exe -m pytest tests\integration -m integration -q

Run unit tests independently with:

    .\.venv\Scripts\python.exe -m pytest tests -m "not integration" -q

When finished, remove only the isolated integration service:

    docker compose -p restaurant-ai-integration -f infrastructure/docker/docker-compose.integration.yml down
