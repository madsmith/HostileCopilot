import asyncio
import time
from typing import Any, Iterable
from enum import Enum

from hostile_copilot.client.uexcorp import UEXCorpClient, BaseLocationID
from hostile_copilot.config import OmegaConfig

from .types import (
    LocationType,
    Planet,
    Moon,
    SpaceStation,
    City,
    Outpost,
    PointOfInterest,
    Orbits,
)

class LocationValidationResponse(Enum):
    VALID = 0
    AMBIGUOUS = 1
    INVALID = 2

class LocationProvider:
    def __init__(self, config: OmegaConfig, client: UEXCorpClient):
        self._config = config
        self._client = client
        self._locations: dict[BaseLocationID, LocationType] = {}
        self._location_update_time: int | None = None
        self._lock = asyncio.Lock()

    async def get_locations(self) -> list[LocationType]:
        expiry_time = self._config.get("app.locations.cache_expiry", 24 * 3600)

        if self._location_update_time is not None and (time.time() - self._location_update_time) < expiry_time:
            return self._locations.values()

        async with self._lock:
            locations = await self._build_locations()
            self._locations = locations
            self._location_update_time = time.time()

        return self._locations.values()

    async def is_valid_location(self, location: str) -> LocationValidationResponse:
        locations = await self.get_locations()
        search_key = location.lower()
        
        candidates = [loc for loc in locations if search_key in loc.name.lower()]

        if len(candidates) == 1:
            return LocationValidationResponse.VALID
        elif len(candidates) > 1:
            return LocationValidationResponse.AMBIGUOUS
        else:
            return LocationValidationResponse.INVALID

    async def get_location_stem(self, location: str) -> str | None:
        locations = await self.get_locations()
        stem = self._shortest_unique_stem(location, locations)
        return stem

    async def _build_locations(self) -> dict[BaseLocationID, LocationType]:
        locations: dict[BaseLocationID, LocationType] = {}
        
        star_systems = await self._client.fetch_star_systems()
        valid_star_systems = [star_system for star_system in star_systems if star_system["is_available_live"]]

        locations: dict[BaseLocationID, LocationType] = {}

        for star_system in valid_star_systems:
            # Fetch planets
            planets: list[dict[str, Any]] = await self._client.fetch_planets(star_system["id"])
            planets = self._filter_valid(planets)

            for planet in planets:
                obj = Planet(**planet)
                if obj.id not in locations:
                    locations[obj.id] = obj

            # Fetch moons
            moons: list[dict[str, Any]] = await self._client.fetch_moons(star_system["id"])
            moons = self._filter_valid(moons)

            for moon in moons:
                obj = Moon(**moon)
                if obj.id not in locations:
                    locations[obj.id] = obj

            # Fetch space stations
            space_stations: list[dict[str, Any]] = await self._client.fetch_stations(star_system["id"])
            space_stations = self._filter_valid(space_stations)

            for space_station in space_stations:
                obj = SpaceStation(**space_station)
                if obj.id not in locations:
                    locations[obj.id] = obj

            # Fetch cities
            cities: list[dict[str, Any]] = await self._client.fetch_cities(star_system["id"])
            cities = self._filter_valid(cities)

            for city in cities:
                obj = City(**city)
                if obj.id not in locations:
                    locations[obj.id] = obj

            # Fetch outposts
            outposts: list[dict[str, Any]] = await self._client.fetch_outposts(star_system["id"])
            outposts = self._filter_valid(outposts)

            for outpost in outposts:
                obj = Outpost(**outpost)
                if obj.id not in locations:
                    locations[obj.id] = obj

            # Fetch points of interest
            points_of_interest: list[dict[str, Any]] = await self._client.fetch_points_of_interest(star_system["id"])
            points_of_interest = self._filter_valid(points_of_interest)

            for point_of_interest in points_of_interest:
                obj = PointOfInterest(**point_of_interest)
                if obj.id not in locations:
                    locations[obj.id] = obj

            # Fetch orbits
            orbits: list[dict[str, Any]] = await self._client.fetch_orbits(star_system["id"])
            orbits = self._filter_valid(orbits)

            for orbit in orbits:
                # Planets have already been added
                if not orbit["is_planet"]:
                    obj = Orbits(**orbit)
                    if obj.id not in locations:
                        locations[obj.id] = obj

        return locations
    
    def _filter_valid(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            item for item in data
            if item["is_available_live"] and item["is_available"] and item['is_visible']
        ]

    def _shortest_unique_stem(self, target: str, locations: Iterable[LocationType]) -> str | None:
        """Return the shortest prefix of `target` that uniquely identifies it among `all_names`.

        Comparison is case-insensitive. If no unique prefix exists (e.g., duplicates of the
        exact same name are present), returns None.

        Args:
            target: The name to identify.
            all_names: Iterable of candidate names to compare against.

        Returns:
            The shortest unique prefix (stem) of `target`, or None if uniqueness is impossible.
        """
        target_normalized = self._normalize_name(target)

        # Build the comparison list excluding the exact same string (case-insensitive) instances
        names = [
            normalized_name
            for loc in locations
            if loc.name is not None and (normalized_name := self._normalize_name(loc.name)) != target_normalized
        ]

        # If any other entry starts with the full target, there is no unique prefix
        # (e.g., target = "Arc", other = "Arc") would have been filtered above; but
        # target = "ArcCorp" and other = "ArcCorp Mining" means no unique prefix shorter than full target.
        # We'll still try; if even the full target isn't unique as a prefix, return None.

        for i in range(1, len(target_normalized) + 1):
            prefix = target_normalized[:i]

            # Check if any other entry starts with prefix
            if not any(name.startswith(prefix) for name in names):
                return prefix

        # Check if the full target itself is unique as a prefix
        if not any(name.startswith(target_normalized) for name in names):
            return target_normalized

        return None

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower()