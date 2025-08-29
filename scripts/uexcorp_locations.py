import asyncio
import json
from pydantic import BaseModel, Field, model_validator
from typing import Any, Annotated, Union, Literal


from hostile_copilot.client.uexcorp import UEXCorpClient
from hostile_copilot.config import load_config

class Location(BaseModel):
    id: int
    name: str
    code: str | None = None

class Planet(Location):
    kind: Literal["planet"] = "planet"

class Moon(Location):
    kind: Literal["moon"] = "moon"

class SpaceStation(Location):
    kind: Literal["space_station"] = "space_station"

    @model_validator(mode="before")
    @classmethod
    def derive_code(cls, values: dict):
        if "code" not in values and "nickname" in values:
            values["code"] = values["nickname"]
        return values

class City(Location):
    kind: Literal["city"] = "city"

class Orbits(Location):
    kind: Literal["orbits"] = "orbits"

LocationType = Annotated[Union[Planet, Moon, SpaceStation, City, Orbits], Field(discriminator="kind")]

def filter_valid_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in data if item["is_available_live"] and item["is_available"] and item['is_visible']]

async def run_script():
    config = load_config()
    client = UEXCorpClient(config)

    star_systems = await client.fetch_star_systems()
    valid_star_systems = [star_system for star_system in star_systems if star_system["is_available_live"]]

    locations = []

    for star_system in valid_star_systems:
        # Fetch planets
        planets: list[dict[str, Any]] = await client.fetch_planets(star_system["id"])
        planets = filter_valid_data(planets)

        for planet in planets:
            locations.append(Planet(**planet))

        moons: list[dict[str, Any]] = await client.fetch_moons(star_system["id"])
        moons = filter_valid_data(moons)

        for moon in moons:
            locations.append(Moon(**moon))

        space_stations: list[dict[str, Any]] = await client.fetch_stations(star_system["id"])
        space_stations = filter_valid_data(space_stations)

        for space_station in space_stations:
            locations.append(SpaceStation(**space_station))

        cities: list[dict[str, Any]] = await client.fetch_cities(star_system["id"])
        cities = filter_valid_data(cities)

        for city in cities:
            locations.append(City(**city))

        orbits: list[dict[str, Any]] = await client.fetch_orbits(star_system["id"])
        orbits = filter_valid_data(orbits)

        for orbit in orbits:
            if not orbit["is_planet"]:
                locations.append(Orbits(**orbit))

    for location in locations:
        print(location)
def main():
    asyncio.run(run_script())

if __name__ == "__main__":
    main()