"""Compatibility export for the context-owned PostgreSQL adapter."""

from app.restaurant_discovery.infrastructure.postgres import (
    PostgresRestaurantDiscoveryAdapter,
)

__all__ = ["PostgresRestaurantDiscoveryAdapter"]
