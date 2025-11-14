import argparse
import asyncio
import logging
import logfire

from hostile_copilot.config import load_config, OmegaConfig
from hostile_copilot.client.app import HostileCoPilotApp
from hostile_copilot.client.tasks.extract_locations import ExtractLocationsTask


async def run_search_gw(query: str, app: HostileCoPilotApp) -> int:
    # Call the tool and print results
    results = await app._tool_search_gravity_well_locations(query)
    for item in results:
        print(item)

    return 0


async def run_loc_extract(filename: str, app: HostileCoPilotApp) -> int:
    task = ExtractLocationsTask(app._config, app._app_config, app._location_provider, filename)
    await task.run()

    for loc in (task.locations or []):
        print(loc)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HostileCoPilot tools tester")
    parser.add_argument("--config", "-c", default=None, help="Path to configuration file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search_gw subcommand
    p_search = subparsers.add_parser("search_gw", help="Search gravity well locations")
    p_search.add_argument("query", help="Search query for gravity well locations")

    # loc_extract subcommand
    p_extract = subparsers.add_parser("loc_extract", help="Extract locations from a file")
    p_extract.add_argument("filename", help="Path to image/text file containing locations")

    return parser

async def load_app(config: OmegaConfig) -> HostileCoPilotApp:
    app = HostileCoPilotApp(config)
    await app.initialize_agents()
    return app

async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    logfire_instance = logfire.configure(
        service_name="hostile_copilot",
        environment="development",
        console=False
    )

    logfire_instance.instrument_pydantic_ai()
    logfire_instance.instrument_requests()

    # Load config
    config: OmegaConfig = load_config(args.config)
    app = await load_app(config)


    if args.command == "search_gw":
        return await run_search_gw(args.query, app)
    elif args.command == "loc_extract":
        return await run_loc_extract(args.filename, app)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
