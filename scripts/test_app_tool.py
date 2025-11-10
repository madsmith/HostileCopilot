import argparse
import asyncio
from typing import Optional

from hostile_copilot.config import load_config, OmegaConfig
from hostile_copilot.client.app import HostileCoPilotApp


async def main(query: str, config_path: Optional[str] = None) -> int:
    # Load configuration
    config: OmegaConfig = load_config(config_path)

    # Initialize app (agents only; skip audio/voice)
    app = HostileCoPilotApp(config)
    await app.initialize_agents()

    # Call the tool and print results
    results = await app._tool_search_gravity_well_locations(query)
    for item in results:
        print(item)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test calling a specific HostileCoPilotApp tool")
    parser.add_argument("query", nargs="?", default="7", help="Search query for gravity well locations")
    parser.add_argument("--config", "-c", default=None, help="Path to configuration file")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(main(args.query, args.config)))
