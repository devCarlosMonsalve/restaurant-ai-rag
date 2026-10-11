# Local infrastructure

This directory contains Docker Compose configurations for local PostgreSQL
with pgvector, optional Phoenix tracing, and a separate disposable database for
integration tests.

## PostgreSQL with pgvector

The main Compose service uses PostgreSQL 16 with the `pgvector` image. Create
the local `.env` file from the checked-in template if you do not already have
one:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Review `.env` and replace any password placeholder with a strong local
password. The template defines the Compose variables `POSTGRES_DB`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_PORT`. Keep `.env` local
and do not commit real credentials. The application connection string must
use the same database, user, password, and port.

From this directory, start the database and check its status:

```powershell
docker compose up -d
docker compose ps
```

The `init-db.sql` script enables the `vector` extension when Docker creates a
new database volume. Verify it with:

```powershell
docker compose exec postgres psql -U restaurant_ai -d restaurant_ai -c "\dx"
```

The database container does not apply application migrations. From `backend`,
run `.\.venv\Scripts\python.exe -m alembic upgrade head` after the database is
available.

## Optional services

Start Phoenix alongside PostgreSQL when local tracing is needed:

```powershell
docker compose -f docker-compose.yml -f docker-compose.phoenix.yml up -d
```

Phoenix is exposed on `127.0.0.1:6006`. Enable tracing in the backend
environment to export spans. Starting this container alone does not verify
that the backend successfully exported traces.

The integration-test database uses PostgreSQL with pgvector on host port
`55432` and a temporary filesystem-backed data directory:

```powershell
docker compose -f docker-compose.integration.yml up -d
docker compose -f docker-compose.integration.yml ps
```

Its data is disposable and is lost when the container is removed.

## Persistence and cleanup

The main database stores data in the named Docker volume
`restaurant-ai-postgres-data`. Stopping and recreating the container preserves
that volume:

```powershell
docker compose down
docker compose up -d
```

Do not run `docker compose down --volumes` unless you intend to permanently
delete the local database. Inspect Docker volumes with:

```powershell
docker volume ls
```
