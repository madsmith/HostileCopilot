import argparse
import asyncio
import json
from typing import Any, Iterable

from hostile_copilot.client.uexcorp import UEXCorpClient
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


def filter_valid_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in data if item["is_available_live"] and item["is_available"] and item['is_visible']]

def shortest_unique_stem(target: str, all_names: Iterable[str]) -> str | None:
    """Return the shortest prefix of `target` that uniquely identifies it among `all_names`.

    Comparison is case-insensitive. If no unique prefix exists (e.g., duplicates of the
    exact same name are present), returns None.

    Args:
        target: The name to identify.
        all_names: Iterable of candidate names to compare against.

    Returns:
        The shortest unique prefix (stem) of `target`, or None if uniqueness is impossible.
    """
    target_norm = target.strip()
    target_lower = target_norm.lower()

    # Build the comparison list excluding the exact same string (case-insensitive) instances
    others = [n for n in all_names if n is not None and n.strip().lower() != target_lower]

    # If any other entry starts with the full target, there is no unique prefix
    # (e.g., target = "Arc", other = "Arc") would have been filtered above; but
    # target = "ArcCorp" and other = "ArcCorp Mining" means no unique prefix shorter than full target.
    # We'll still try; if even the full target isn't unique as a prefix, return None.

    for i in range(1, len(target_norm) + 1):
        prefix = target_norm[:i]
        prefix_lower = prefix.lower()
        if not any(o.strip().lower().startswith(prefix_lower) for o in others):
            return prefix

    # Check if the full target itself is unique as a prefix
    if not any(o.strip().lower().startswith(target_lower) for o in others):
        return target_norm

    return None

async def run_script(args: argparse.Namespace):
    config: OmegaConfig = load_config(args.config)

    client = UEXCorpClient(config)

    locations: dict[BaseLocationID, LocationType] = await retrieve_locations(client)

    if args.debug:
        with open("locations.json", "w", encoding="utf-8") as f:
            payload = {str(k): v.model_dump(mode="json") for k, v in locations.items()}
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    if args.mode == "print":
        await print_locations(locations)
    elif args.mode == "search":
        await search_locations(locations)
    elif args.mode == "stem":
        await stem_locations(locations)

async def retrieve_locations(client: UEXCorpClient) -> dict[BaseLocationID, LocationType]:
    star_systems = await client.fetch_star_systems()
    valid_star_systems = [star_system for star_system in star_systems if star_system["is_available_live"]]

    locations: dict[BaseLocationID, LocationType] = {}

    for star_system in valid_star_systems:
        # Fetch planets
        planets: list[dict[str, Any]] = await client.fetch_planets(star_system["id"])
        planets = filter_valid_data(planets)

        for planet in planets:
            obj = Planet(**planet)
            if obj.id not in locations:
                locations[obj.id] = obj

        # Fetch moons
        moons: list[dict[str, Any]] = await client.fetch_moons(star_system["id"])
        moons = filter_valid_data(moons)

        for moon in moons:
            obj = Moon(**moon)
            if obj.id not in locations:
                locations[obj.id] = obj

        # Fetch space stations
        space_stations: list[dict[str, Any]] = await client.fetch_stations(star_system["id"])
        space_stations = filter_valid_data(space_stations)

        for space_station in space_stations:
            obj = SpaceStation(**space_station)
            if obj.id not in locations:
                locations[obj.id] = obj

        # Fetch cities
        cities: list[dict[str, Any]] = await client.fetch_cities(star_system["id"])
        cities = filter_valid_data(cities)

        for city in cities:
            obj = City(**city)
            if obj.id not in locations:
                locations[obj.id] = obj

        # Fetch outposts
        outposts: list[dict[str, Any]] = await client.fetch_outposts(star_system["id"])
        outposts = filter_valid_data(outposts)

        for outpost in outposts:
            obj = Outpost(**outpost)
            if obj.id not in locations:
                locations[obj.id] = obj

        # Fetch points of interest
        points_of_interest: list[dict[str, Any]] = await client.fetch_points_of_interest(star_system["id"])
        points_of_interest = filter_valid_data(points_of_interest)

        for point_of_interest in points_of_interest:
            obj = PointOfInterest(**point_of_interest)
            if obj.id not in locations:
                locations[obj.id] = obj

        # Fetch orbits
        orbits: list[dict[str, Any]] = await client.fetch_orbits(star_system["id"])
        orbits = filter_valid_data(orbits)

        for orbit in orbits:
            if not orbit["is_planet"]:
                obj = Orbits(**orbit)
                if obj.id not in locations:
                    locations[obj.id] = obj

    return locations

async def print_locations(locations: dict[BaseLocationID, LocationType]):
    for location in locations.values():
        print(location)

async def search_locations(locations: dict[BaseLocationID, LocationType]):
    try:
        while True:
            search = input("Search (or 'exit' to quit): ")
            if search.lower() == "exit" or search.lower() == "quit" or search.lower() == "q":
                break
            for location in locations.values():
                if search.lower() in location.name.lower():
                    print(location)
    except EOFError:
        pass
    except KeyboardInterrupt:
        pass

async def stem_locations(locations: dict[BaseLocationID, LocationType]):
    try:
        while True:
            search = input("Search (or 'exit' to quit): ")
            if search.lower() == "exit" or search.lower() == "quit" or search.lower() == "q":
                break

            matching_locations = []
            for location in locations.values():
                if search.lower() in location.name.lower():
                    matching_locations.append(location)

            if len(matching_locations) > 1:
                selected_location = None
                while selected_location is None:
                    print("Multiple matching locations found:")
                    index = 1
                    for location in matching_locations:
                        print(f"{index}: {location}")
                        index += 1
                
                selection = input("Select one: ")
                try:
                    selected_location = matching_locations[int(selection) - 1]
                except (ValueError, IndexError):
                    print("Invalid selection.")
                    continue
            elif len(matching_locations) == 1:
                selected_location = matching_locations[0]
            else:
                print("No matching locations found.")
                continue

            print(f"Selected: {selected_location.name}")
            location_names = [location.name for location in locations.values()]
            print(f"Stem: {shortest_unique_stem(selected_location.name, location_names)}")
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