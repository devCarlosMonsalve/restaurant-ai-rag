from typing import Protocol

from app.itinerary_planning.application.contracts import RestaurantSearchEvidence


class RestaurantEvidenceProvider(Protocol):
    async def search_restaurants(self, query: str) -> RestaurantSearchEvidence:
        """Retrieve restaurant evidence for an itinerary-planning request."""
