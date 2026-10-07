"""add open data image provenance

Revision ID: d2a7c49e5810
Revises: c1d9e5a47f62
Create Date: 2026-10-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d2a7c49e5810"
down_revision: Union[str, Sequence[str], None] = "c1d9e5a47f62"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "osm_places",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("osm_type", sa.String(length=10), nullable=False),
        sa.Column("osm_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("city", sa.String(length=100), nullable=False),
        sa.Column("cuisine", sa.String(length=255), nullable=True),
        sa.Column("location", sa.String(length=512), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("wikimedia_commons", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.String(length=2048), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "osm_type",
            "osm_id",
            name="uq_osm_places_osm_type_osm_id",
        ),
    )
    op.add_column(
        "image_embeddings",
        sa.Column("source_url", sa.String(length=2048), nullable=True),
    )
    op.add_column(
        "image_embeddings",
        sa.Column("license_name", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "image_embeddings",
        sa.Column("license_url", sa.String(length=2048), nullable=True),
    )
    op.add_column(
        "image_embeddings",
        sa.Column("attribution", sa.Text(), nullable=True),
    )
    op.add_column(
        "image_embeddings",
        sa.Column("osm_place_id", sa.UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_image_embeddings_osm_place_id_osm_places",
        "image_embeddings",
        "osm_places",
        ["osm_place_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_image_embeddings_osm_place_id_osm_places",
        "image_embeddings",
        type_="foreignkey",
    )
    op.drop_column("image_embeddings", "osm_place_id")
    op.drop_column("image_embeddings", "attribution")
    op.drop_column("image_embeddings", "license_url")
    op.drop_column("image_embeddings", "license_name")
    op.drop_column("image_embeddings", "source_url")
    op.drop_table("osm_places")
