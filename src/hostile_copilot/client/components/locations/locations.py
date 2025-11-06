import asyncio
from enum import Enum
import logging
import re
import time
from typing import Any, Iterable

from hostile_copilot.client.uexcorp import UEXCorpClient, BaseLocationID
from hostile_copilot.client.regolith import RegolithClient
from hostile_copilot.config import OmegaConfig

from .types import (
    LocationType,
    LocationInfo,
    Planet,
    Moon,
    SpaceStation,
    City,
    Outpost,
    PointOfInterest,
    Orbits,
    GravityWell,
)

logger = logging.getLogger(__name__)

class LocationValidationResponse(Enum):
    VALID = 0
    AMBIGUOUS = 1
    INVALID = 2
    
    def __bool__(self) -> bool:
        # Only VALID should evaluate to True in boolean context
        return self is LocationValidationResponse.VALID

class LocationProvider:
    def __init__(self, config: OmegaConfig, client: UEXCorpClient, regolith_client: RegolithClient):
        self._config = config
        self._uex_client = client
        self._regolith_client = regolith_client
        self._locations: dict[BaseLocationID, LocationInfo] = {}
        self._location_update_time: int | None = None
        self._lock = asyncio.Lock()

    async def get_nav_locations(self) -> list[LocationInfo]:

        locations = await self.get_locations()
        return [loc for loc in locations if loc.is_navigable]
        
    async def get_locations(self) -> list[LocationInfo]:
        expiry_time = self._config.get("app.locations.cache_expiry", 24 * 3600)

        if self._location_update_time is not None and (time.time() - self._location_update_time) < expiry_time:
            return self._locations.values()

        async with self._lock:
            locations = await self._build_locations()
            self._locations = locations
            self._location_update_time = time.time()

        return self._locations.values()

    async def search(self, search_str: str) -> list[LocationInfo]:
        locations = await self.get_locations()
        search_key = self._normalize_name(search_str)
        
        candidates = [loc for loc in locations if search_key in self._normalize_name(loc.name)]
        return candidates
    
    async def is_valid_location(self, location: str) -> LocationValidationResponse:
        locations = await self.search(location)

        if len(locations) == 1:
            return LocationValidationResponse.VALID
        elif len(locations) > 1:
            return LocationValidationResponse.AMBIGUOUS
        else:
            return LocationValidationResponse.INVALID

    async def get_location_stem(self, location: str | LocationInfo) -> str | None:
        locations = await self.get_locations()
        if not isinstance(location, str):
            location = location.name
        stem = self._shortest_unique_stem(location, locations)
        return stem

    async def _build_locations(self) -> dict[BaseLocationID, LocationInfo]:
        locations: dict[BaseLocationID, LocationInfo] = {}
        
        star_systems = await self._uex_client.fetch_star_systems()
        valid_star_systems = [star_system for star_system in star_systems if star_system["is_available_live"]]

        non_navigable_locations = self._config.get("location_provider.uexcorp_non_navigable_locations", [])
        non_navigable_location_patterns = [re.compile(loc) for loc in non_navigable_locations]

        navigable_locations = self._config.get("location_provider.regolith_navigable_locations", [])
        navigable_location_patterns = [re.compile(loc) for loc in navigable_locations]

        def is_navigable(location: LocationType, patterns: list[re.Pattern[str]], default: bool = True) -> bool:
            """
            Helper function to determine if a location is navigable based on a list of patterns.
            """
            if default:
                return not any(loc.match(location.name) for loc in patterns)
            else:
                return any(loc.match(location.name) for loc in patterns)
            
        location_infos: dict[BaseLocationID, LocationInfo] = {}

        # First load gravity wells from Regolith
        gravity_wells: list[dict[str, Any]] = await self._regolith_client.fetch_gravity_wells()

        for gravity_well in gravity_wells:
            loc = LocationInfo.from_gravity_well(gravity_well)
            loc.is_navigable = is_navigable(loc, navigable_location_patterns, False)
            location_infos[loc.id] = loc

            obj = GravityWell(**gravity_well)
            if obj.id not in locations:
                # Don't add most gravity wells to the list of nav locations
                obj.is_navigable = is_navigable(obj, navigable_location_patterns, False)
                locations[obj.id] = obj

        # Load locations from UEXCorp
        for star_system in valid_star_systems:
            # == Fetch planets ==
            planets: list[dict[str, Any]] = await self._uex_client.fetch_planets(star_system["id"])
            planets = self._filter_valid(planets)

            for planet in planets:
                loc = LocationInfo.from_planet(planet)
                location_infos[loc.id] = loc

                obj = Planet(**planet)
                if obj.id not in locations:
                    obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                    locations[obj.id] = obj

            # == Fetch moons ==
            moons: list[dict[str, Any]] = await self._uex_client.fetch_moons(star_system["id"])
            moons = self._filter_valid(moons)

            for moon in moons:
                loc = LocationInfo.from_moon(moon)
                location_infos[loc.id] = loc

                obj = Moon(**moon)
                if obj.id not in locations:
                    obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                    locations[obj.id] = obj

            # == Fetch space stations ==
            space_stations: list[dict[str, Any]] = await self._uex_client.fetch_stations(star_system["id"])
            space_stations = self._filter_valid(space_stations)

            for space_station in space_stations:
                loc = LocationInfo.from_space_station(space_station)
                location_infos[loc.id] = loc

                obj = SpaceStation(**space_station)
                if obj.id not in locations:
                    obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                    locations[obj.id] = obj

            # == Fetch cities ==
            cities: list[dict[str, Any]] = await self._uex_client.fetch_cities(star_system["id"])
            cities = self._filter_valid(cities)

            for city in cities:
                loc = LocationInfo.from_city(city)
                location_infos[loc.id] = loc

                obj = City(**city)
                if obj.id not in locations:
                    obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                    locations[obj.id] = obj

            # == Fetch outposts ==
            outposts: list[dict[str, Any]] = await self._uex_client.fetch_outposts(star_system["id"])
            outposts = self._filter_valid(outposts)

            for outpost in outposts:
                loc = LocationInfo.from_outpost(outpost)
                location_infos[loc.id] = loc

                obj = Outpost(**outpost)
                if obj.id not in locations:
                    obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                    locations[obj.id] = obj

            # == Fetch points of interest ==
            if self._config.get("location_provider.include_points_of_interest", False):
                points_of_interest: list[dict[str, Any]] = await self._uex_client.fetch_points_of_interest(star_system["id"])
                points_of_interest = self._filter_valid(points_of_interest)

                for point_of_interest in points_of_interest:
                    loc = LocationInfo.from_point_of_interest(point_of_interest)
                    location_infos[loc.id] = loc

                    obj = PointOfInterest(**point_of_interest)
                    if obj.id not in locations:
                        obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                        locations[obj.id] = obj

            # Fetch orbits
            orbits: list[dict[str, Any]] = await self._uex_client.fetch_orbits(star_system["id"])
            orbits = self._filter_valid(orbits)

            for orbit in orbits:
                # Planets have already been added
                if not orbit["is_planet"]:
                    obj = Orbits(**orbit)
                    if obj.id not in locations:
                        obj.is_navigable = is_navigable(obj, non_navigable_location_patterns, True)
                        locations[obj.id] = obj
        
        # Fetch Gravity Wells from Regolith
        gravity_wells: list[dict[str, Any]] = await self._regolith_client.fetch_gravity_wells()
        
        return location_infos
    
    def _filter_valid(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            item for item in data
            if item["is_available_live"] and item["is_available"] and item['is_visible']
        ]

    def _shortest_unique_stem(self, target: str, locations: Iterable[LocationInfo]) -> str | None:
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

        # Consider collisions at ANY word boundary in compared names, not just the very start.
        # Word boundary is defined as index 0 or any position preceded by a non-alphanumeric.
        def word_boundaries(s: str) -> list[int]:
            return [i for i in range(len(s)) if i == 0 or not s[i - 1].isalnum()]

        def any_word_prefix(s: str, prefix: str) -> bool:
            for i in word_boundaries(s):
                if s.startswith(prefix, i):
                    return True
            return False

        # If any other entry starts with the prefix at ANY word boundary, it's not unique yet.
        for i in range(1, len(target_normalized) + 1):
            prefix = target_normalized[:i]
            if not any(any_word_prefix(name, prefix) for name in names):
                return prefix

        # Check if the full target itself is unique as a prefix at any word boundary
        if not any(any_word_prefix(name, target_normalized) for name in names):
            return target_normalized

        return None

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower()