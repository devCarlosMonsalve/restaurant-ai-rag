# PostgreSQL integration tests

These tests are opt-in and use a dedicated local PostgreSQL database with the
real pgvector extension. They run Alembic migrations against that database and
roll back each test's inserted rows. They do not use the development Compose
service or its persistent volume.

From the repository root, start the isolated, non-persistent test service:

    docker compose -p restaurant-ai-integration -f infrastructure/docker/docker-compose.integration.yml up --wait -d

In PowerShell, from `backend`, point the integration suite only at its guarded
test database and run it:

    $env:POSTGRES_TEST_DATABASE_URL = "postgresql+psycopg://restaurant_ai_test:restaurant_ai_test@127.0.0.1:55432/restaurant_ai_integration_test"
    .\.venv\Scripts\python.exe -m pytest tests\integration -m integration -q

The fixture rejects URLs that are not loopback `postgresql+psycopg` connections
to `restaurant_ai_integration_test`. It also requires the `vector` extension
before applying the Alembic migrations.

Run unit tests independently with:

    .\.venv\Scripts\python.exe -m pytest tests -m "not integration" -q

When finished, remove only the isolated integration service:

    docker compose -p restaurant-ai-integration -f infrastructure/docker/docker-compose.integration.yml down
