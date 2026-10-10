import argparse
import asyncio

from app.itinerary_planning.application.planner import plan_itinerary_dining
from app.itinerary_planning.infrastructure.a2a.restaurant_discovery_client import (
    DEFAULT_RESTAURANT_AGENT_URL,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft restaurant suggestions for an itinerary using A2A evidence."
    )
    parser.add_argument("query", help="Restaurant request to delegate.")
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Number of itinerary days to fill with restaurant suggestions.",
    )
    parser.add_argument(
        "--agent-url",
        default=DEFAULT_RESTAURANT_AGENT_URL,
        help="Base URL of the Restaurant Discovery Agent.",
    )
    args = parser.parse_args()
    draft = asyncio.run(
        plan_itinerary_dining(
            args.query,
            args.days,
            agent_url=args.agent_url,
        )
    )
    print(draft.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
