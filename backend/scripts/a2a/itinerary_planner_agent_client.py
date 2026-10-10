import argparse
import asyncio

from app.interfaces.a2a.itinerary_planner_agent_client import (
    DEFAULT_ITINERARY_PLANNER_AGENT_URL,
    delegate_itinerary_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Request a dining draft from the Itinerary Planner Agent."
    )
    parser.add_argument("query", help="Restaurant request for the itinerary.")
    parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Number of itinerary days to fill with restaurant suggestions.",
    )
    parser.add_argument(
        "--agent-url",
        default=DEFAULT_ITINERARY_PLANNER_AGENT_URL,
        help="Base URL of the Itinerary Planner Agent.",
    )
    args = parser.parse_args()
    draft = asyncio.run(
        delegate_itinerary_plan(
            args.query,
            args.days,
            agent_url=args.agent_url,
        )
    )
    print(draft.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
