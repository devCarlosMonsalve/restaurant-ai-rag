"""create image embeddings table

Revision ID: c1d9e5a47f62
Revises: 8f4b2c7a1d93
Create Date: 2026-10-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "c1d9e5a47f62"
down_revision: Union[str, Sequence[str], None] = "8f4b2c7a1d93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "image_embeddings",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("source_name", sa.String(length=255), nullable=False),
        sa.Column("image_path", sa.String(length=2048), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_path", name="uq_image_embeddings_image_path"),
    )


def downgrade() -> None:
    op.drop_table("image_embeddings")
