from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base
from app.models.osm_place import OsmPlace


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    __table_args__ = (
        UniqueConstraint("image_path", name="uq_image_embeddings_image_path"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    source_name: Mapped[str] = mapped_column(String(255), nullable=False)
    image_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    license_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    license_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    attribution: Mapped[str | None] = mapped_column(Text, nullable=True)
    osm_place_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("osm_places.id", ondelete="SET NULL"),
        nullable=True,
    )
    osm_place: Mapped[OsmPlace | None] = relationship()
