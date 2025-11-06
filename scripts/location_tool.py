import argparse
import asyncio
import json
from typing import Any, Iterable

from hostile_copilot.client.uexcorp import UEXCorpClient
from hostile_copilot.client.regolith import RegolithClient
from hostile_copilot.client.components.locations import LocationProvider
from hostile_copilot.config import OmegaConfig, load_config
from hostile_copilot.client.components import (
    LocationType,
    Planet,
    Moon,
    SpaceStation,
    City,
    Orbits,
    Outpost,
    PointOfInterest,
    GravityWell,
)

async def run_script(args: argparse.Namespace):
    config: OmegaConfig = load_config(args.config)

    uex_client = UEXCorpClient(config)
    regolith_client = RegolithClient(config)
    provider = LocationProvider(config, uex_client, regolith_client)

    locations: list[LocationType] = await provider.get_locations()

    if args.debug:
        with open("locations.json", "w", encoding="utf-8") as f:
            payload = {str(location.id): location.model_dump(mode="json") for location in locations}
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    if args.mode == "print":
        await print_locations(locations)
    elif args.mode == "search":
        await search_locations(provider)
    elif args.mode == "stem":
        await stem_locations(provider)
    elif args.mode == "regolith":
        await regolith_locations(regolith_client)
    elif args.mode == "dup_check":
        dup_check(locations)


async def print_locations(locations: list[LocationType]):
    for location in locations:
        print(location.name)

async def search_locations(provider: LocationProvider):
    try:
        nav_search = False
        verbose = False
        while True:
            search = input("Search (or 'exit' to quit): ")
            if search.lower() == "exit" or search.lower() == "quit" or search.lower() == "q":
                break
            if search.lower() == "nav":
                nav_search = not nav_search
                print(f"Navigation search: {nav_search}")
                continue
            if search.lower() == "verbose":
                verbose = not verbose
                print(f"Verbose results: {verbose}")
                continue

            if nav_search:
                locations = await provider.get_nav_locations()
                locations = [loc for loc in locations if search.lower() in loc.name.lower()]
            else:
                locations = await provider.search(search)
            for location in locations:
                print(location if not verbose else location.__repr__())
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

async def regolith_locations(regolith_client: RegolithClient):
    try:
        while True:
            search = input("Search (or 'exit' to quit): ")
            if search.lower() == "exit" or search.lower() == "quit" or search.lower() == "q":
                break

            locations = await regolith_client.fetch_gravity_wells()
                

            for location in locations:
                if search.lower() in location['label'].lower():
                    print(location)
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass

def dup_check(locations: list[LocationType]):
    code_lookup = {}
    for location in locations:
        if hasattr(location, "code"):
            code = location.code
            if code and (code in code_lookup):
                print(f"Duplicate code: {code}")
                print(f"    Location: {location}")
                print(f"    Other: {code_lookup[code]}")
            else:
                code_lookup[code] = location

    for location in locations:
        if hasattr(location, "label"):
            gw_code = location.id
            if gw_code in code_lookup:
                print(f"Duplicate GW code: {gw_code}")
                print(f"    Location: {location}")
                print(f"    Other: {code_lookup[gw_code]}")
            else:
                code_lookup[gw_code] = location
def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", "-c", help="Path to configuration file", default=None)
    argparser.add_argument(
        "mode", 
        nargs="?", 
        default="print", 
        choices=["print", "search", "stem", "regolith", "dup_check"], 
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