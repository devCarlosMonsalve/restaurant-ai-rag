from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


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
