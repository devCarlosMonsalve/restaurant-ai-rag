from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Float, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class OsmPlace(Base):
    __tablename__ = "osm_places"
    __table_args__ = (
        UniqueConstraint("osm_type", "osm_id", name="uq_osm_places_osm_type_osm_id"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    osm_type: Mapped[str] = mapped_column(String(10), nullable=False)
    osm_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    cuisine: Mapped[str | None] = mapped_column(String(255), nullable=True)
    location: Mapped[str | None] = mapped_column(String(512), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    features: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    wikimedia_commons: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
