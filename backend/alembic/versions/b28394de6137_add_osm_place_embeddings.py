"""Add semantic embeddings to OSM restaurants.

Revision ID: b28394de6137
Revises: a47c812ee920
Create Date: 2026-10-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "b28394de6137"
down_revision: Union[str, Sequence[str], None] = "a47c812ee920"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "osm_places",
        sa.Column("embedding", Vector(768), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("osm_places", "embedding")
