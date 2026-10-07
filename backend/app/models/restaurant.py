from uuid import UUID, uuid4

from sqlalchemy import String, Text
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Restaurant(Base):
    __tablename__ = "restaurants"

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    cuisine: Mapped[str] = mapped_column(String(100))
    price_range: Mapped[str] = mapped_column(String(10))
    location: Mapped[str] = mapped_column(String(255))
