"""Store searchable OSM restaurant features.

Revision ID: c80d3f91a642
Revises: b28394de6137
Create Date: 2026-10-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c80d3f91a642"
down_revision: Union[str, Sequence[str], None] = "b28394de6137"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "osm_places",
        sa.Column("features", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("osm_places", "features")
