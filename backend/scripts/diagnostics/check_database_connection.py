from sqlalchemy import text

from app.infrastructure.persistence.postgres.database import engine


def main() -> None:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print(result.scalar())


if __name__ == "__main__":
    main()