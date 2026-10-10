import argparse
import asyncio

from app.itinerary_planning.infrastructure.a2a.restaurant_discovery_client import (
    DEFAULT_RESTAURANT_AGENT_URL,
    delegate_restaurant_discovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delegate restaurant discovery to the local A2A Agent."
    )
    parser.add_argument(
        "query",
        help="Restaurant request to delegate to the Restaurant Discovery Agent.",
    )
    parser.add_argument(
        "--agent-url",
        default=DEFAULT_RESTAURANT_AGENT_URL,
        help="Base URL of the Restaurant Discovery Agent.",
    )
    args = parser.parse_args()
    response = asyncio.run(
        delegate_restaurant_discovery(
            args.query,
            agent_url=args.agent_url,
        )
    )
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
