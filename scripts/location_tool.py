import argparse
import asyncio
import json
from typing import Any, Iterable

from hostile_copilot.client.uexcorp import UEXCorpClient
from hostile_copilot.client.components.locations import LocationProvider
from hostile_copilot.config import OmegaConfig, load_config
from hostile_copilot.client.uexcorp import BaseLocationID
from hostile_copilot.client.components import (
    LocationType,
    Planet,
    Moon,
    SpaceStation,
    City,
    Orbits,
    Outpost,
    PointOfInterest,
)


async def run_script(args: argparse.Namespace):
    config: OmegaConfig = load_config(args.config)

    client = UEXCorpClient(config)
    provider = LocationProvider(config, client)

    locations: list[LocationType] = await provider.get_locations()

    if args.debug:
        with open("locations.json", "w", encoding="utf-8") as f:
            payload = {str(k): v.model_dump(mode="json") for k, v in locations.items()}
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    if args.mode == "print":
        await print_locations(locations)
    elif args.mode == "search":
        await search_locations(provider)
    elif args.mode == "stem":
        await stem_locations(provider)


async def print_locations(locations: list[LocationType]):
    for location in locations:
        print(location.name)

async def search_locations(provider: LocationProvider):
    try:
        while True:
            search = input("Search (or 'exit' to quit): ")
            if search.lower() == "exit" or search.lower() == "quit" or search.lower() == "q":
                break
            locations = await provider.search(search)
            for location in locations:
                print(location)
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass

async def stem_locations(provider: LocationProvider):
    try:
        while True:
            search = input("Search (or 'exit' to quit): ")
            if search.lower() == "exit" or search.lower() == "quit" or search.lower() == "q":
                break

            locations = await provider.search(search)
            if len(locations) == 1:
                selected_location = locations[0]
            elif len(locations) > 1:
                selected_location = None
                while selected_location is None:
                    print("Multiple matching locations found:")
                    index = 1
                    for location in locations:
                        print(f"{index}: {location}")
                        index += 1
                
                    selection = input("Select one (or 'exit' to quit): ")
                    try:
                        if selection.lower() == "exit" or selection.lower() == "quit" or selection.lower() == "q":
                            break
                        selected_location = locations[int(selection) - 1]
                    except (ValueError, IndexError):
                        print("Invalid selection.")
                        continue
            else:
                print("No matching locations found.")
                continue

            print(f"Selected: {selected_location.name}")
            stem = await provider.get_location_stem(selected_location)
            print(f"Stem: {stem}")
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", "-c", help="Path to configuration file", default=None)
    argparser.add_argument(
        "mode", 
        nargs="?", 
        default="print", 
        choices=["print", "search", "stem"], 
        help="Mode to run: 'print' (default)."
    )
    argparser.add_argument("--debug", "-d", help="Enable debug mode", action="store_true")
    
    args = argparser.parse_args()

    try:
        asyncio.run(run_script(args))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()