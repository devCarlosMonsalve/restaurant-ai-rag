"""Allow OSM places without Commons references.

Revision ID: a47c812ee920
Revises: d2a7c49e5810
Create Date: 2026-10-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a47c812ee920"
down_revision: Union[str, Sequence[str], None] = "d2a7c49e5810"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("osm_places", "wikimedia_commons", nullable=True)


def downgrade() -> None:
    op.execute(
        sa.text(
            "UPDATE osm_places SET wikimedia_commons = '' "
            "WHERE wikimedia_commons IS NULL"
        )
    )
    op.alter_column(
        "osm_places",
        "wikimedia_commons",
        nullable=False,
    )
