import asyncio
from difflib import SequenceMatcher
from enum import Enum
import logging
import re
import time
from typing import Any, get_args, Iterable

from hostile_copilot.client.uexcorp import (
    UEXCorpClient,
    BaseLocationID,
    SpaceStationID,
    GravityWellID,
    OutpostID,
)
from hostile_copilot.client.regolith import RegolithClient
from hostile_copilot.config import OmegaConfig

from .text import CanonicalNameProcessor, NormalizedName
from .custom_types import (
    LocationType,
    LocationInfo,
    UEXType,
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

type LocationsDict = dict[BaseLocationID, LocationInfo]

class LocationProvider:
    def __init__(self, config: OmegaConfig, client: UEXCorpClient, regolith_client: RegolithClient):
        self._config = config
        self._uex_client = client
        self._regolith_client = regolith_client
        self._locations: dict[BaseLocationID, LocationInfo] = {}
        self._location_update_time: int | None = None
        self._lock = asyncio.Lock()

        self._canonical_name_processor = CanonicalNameProcessor(config)

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

    async def search(self, search_str: str, gravity_well: bool = False, navigable: bool = False) -> list[LocationInfo]:
        locations = await self.get_locations()

        search_key = NormalizedName(search_str)

        if gravity_well:
            locations = [loc for loc in locations if loc.is_gravity_well]

        if navigable:
            locations = [loc for loc in locations if loc.is_navigable]
        
        candidates = [
            loc
            for loc in locations
            if (
                search_key.matches(loc.name)
                or (search_key.matches(loc.code) if loc.code is not None else False)
                or any(search_key.matches(alias) for alias in loc.aliases if loc.aliases is not None)
            )
        ]

        # Rank candidates by canonical name similarity
        target_name = self._canonical_name_processor.process(search_str)
        def rank_candidate(candidate: LocationInfo) -> float:
            canonical_name = self._canonical_name_processor.process(candidate.name)
            return SequenceMatcher(None, target_name, canonical_name).ratio()
        
        candidates.sort(key=rank_candidate, reverse=True)
        
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

    async def _build_locations(self) -> LocationsDict:
        location_infos: LocationsDict = {}
        
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
            

        # First load gravity wells from Regolith
        gravity_wells: list[dict[str, Any]] = await self._regolith_client.fetch_gravity_wells()

        for gravity_well in gravity_wells:
            loc = LocationInfo.from_gravity_well(gravity_well)
            loc.is_navigable = is_navigable(loc, navigable_location_patterns, False)
            location_infos[loc.id] = loc

        # Load locations from UEXCorp
        for star_system in valid_star_systems:
            # == Fetch planets ==
            planets: list[dict[str, Any]] = await self._uex_client.fetch_planets(star_system["id"])
            planets = self._filter_valid(planets)

            for planet in planets:
                loc = LocationInfo.from_planet(planet)
                loc.is_navigable = is_navigable(loc, non_navigable_location_patterns, True)

                if loc.id not in location_infos:
                    location_infos[loc.id] = loc

            # == Fetch moons ==
            moons: list[dict[str, Any]] = await self._uex_client.fetch_moons(star_system["id"])
            moons = self._filter_valid(moons)

            for moon in moons:
                loc = LocationInfo.from_moon(moon)
                loc.is_navigable = is_navigable(loc, non_navigable_location_patterns, True)

                if loc.id not in location_infos:
                    location_infos[loc.id] = loc

            # == Fetch space stations ==
            space_stations: list[dict[str, Any]] = await self._uex_client.fetch_stations(star_system["id"])
            space_stations = self._filter_valid(space_stations)

            for space_station in space_stations:
                loc = LocationInfo.from_space_station(space_station)
                loc.is_navigable = is_navigable(loc, non_navigable_location_patterns, True)

                if loc.id not in location_infos:
                    location_infos[loc.id] = loc

            # == Fetch cities ==
            cities: list[dict[str, Any]] = await self._uex_client.fetch_cities(star_system["id"])
            cities = self._filter_valid(cities)

            for city in cities:
                loc = LocationInfo.from_city(city)
                loc.is_navigable = is_navigable(loc, non_navigable_location_patterns, True)

                if loc.id not in location_infos:
                    location_infos[loc.id] = loc

            # == Fetch outposts ==
            outposts: list[dict[str, Any]] = await self._uex_client.fetch_outposts(star_system["id"])
            outposts = self._filter_valid(outposts)

            for outpost in outposts:
                loc = LocationInfo.from_outpost(outpost)
                loc.is_navigable = is_navigable(loc, non_navigable_location_patterns, True)

                if loc.id not in location_infos:
                    location_infos[loc.id] = loc

            def has_matching_outpost_by_name(name: str) -> bool:
                search_key = NormalizedName(name)
                return any(search_key.matches(loc.name) for loc in location_infos.values() if isinstance(loc.id, OutpostID))

            # == Fetch points of interest ==
            if self._config.get("location_provider.include_points_of_interest", True):
                points_of_interest: list[dict[str, Any]] = await self._uex_client.fetch_points_of_interest(star_system["id"])
                points_of_interest = self._filter_valid(points_of_interest)

                for point_of_interest in points_of_interest:
                    loc = LocationInfo.from_point_of_interest(point_of_interest)
                    loc.is_navigable = is_navigable(loc, non_navigable_location_patterns, True)

                    if loc.id not in location_infos and not has_matching_outpost_by_name(loc.name):
                        location_infos[loc.id] = loc

            # == Fetch orbits ==
            if self._config.get("location_provider.include_orbits", False):
                # At the moment the only marker we really get from this that's shown on the map is
                # the stars and CRU-L2, both of which are not navigable, so there's not really a 
                # reason to add this data to the location provider with the current feature set.
                orbits: list[dict[str, Any]] = await self._uex_client.fetch_orbits(star_system["id"])
                orbits = self._filter_valid(orbits)

                for orbit in orbits:
                    # Planets have already been added
                    if not orbit["is_planet"]:
                        loc = LocationInfo.from_orbit(orbit)
                        loc.is_navigable = is_navigable(loc, navigable_locations, False)

                        if loc.id not in location_infos:
                            location_infos[loc.id] = loc
        
        cleaned_locations = self._post_process_locations(location_infos)
        
        return cleaned_locations

    def _post_process_locations(self, location_infos: LocationsDict) -> LocationsDict:
        cleaned_locations: LocationsDict = {}

        processed_codes = set()
        processed_ids = set()

        for location in location_infos.values():
            # Already processed
            if location.code:
                if location.code not in processed_codes and location.id not in processed_ids:
                    merged_location, processed_locations = self._merge_locations(location_infos, location)
                    cleaned_locations[merged_location.id] = merged_location
                    processed_codes.add(location.code)
                    processed_ids.update(loc.id for loc in processed_locations)
            else:
                if location.id not in processed_ids:
                    cleaned_locations[location.id] = location
                    processed_ids.add(location.id)

        return cleaned_locations

    def _merge_locations(self, location_infos: LocationsDict, location: LocationInfo) -> tuple[LocationInfo, list[LocationInfo]]:
        matching_locations = [
            loc 
            for loc in location_infos.values()
            if (location.code == loc.code) or
               (loc.aliases is not None and location.code in loc.aliases)
        ]

        # Force merge UEX Space stations against Regolith Gravity Wells
        # We're processing Regolith first, so only do this if the current location is
        # a gravity well and see if there's a space station matching it
        if isinstance(location.id, GravityWellID):
            space_stations = [loc for loc in location_infos.values() if isinstance(loc.id, SpaceStationID)]
            for station in space_stations:
                search_key = NormalizedName(station.name)
                if search_key.matches(location.name):
                    matching_locations.append(station)

        # Remove POI
        # Check merge list
        merge_list = self._config.get("location_provider.merge_list", [])
        location_id_str = str(location.id)
        for merge_group in merge_list:
            if location_id_str in merge_group:
                print(f"Found merge group: {merge_group}")
                other_ids = [entry for entry in merge_group if entry != location_id_str]
                additional_locations = [loc for loc in location_infos.values() if str(loc.id) in other_ids]
                matching_locations.extend(additional_locations)

        preferred_name = next(
            (loc.name for loc in matching_locations if not isinstance(loc.id, GravityWellID)),
            None,
        )

        name = preferred_name or location.name

        aliases = list({
            alias
            for loc in matching_locations 
            for alias in (loc.aliases or []) + [loc.name] if alias != name
        })

        UEXTYPE_SET = set(get_args(UEXType))
        uex_id = next((loc.id for loc in matching_locations if loc.id in UEXTYPE_SET), None)
        uex_type = next((loc.kind for loc in matching_locations if loc.kind in UEXTYPE_SET), None)

        gravity_well_type = next((loc.gravity_well_type for loc in matching_locations if loc.gravity_well_type is not None), None)
        is_gravity_well = any(loc.is_gravity_well for loc in matching_locations)

        is_navigable = any(loc.is_navigable for loc in matching_locations)
        is_space = any(loc.is_space for loc in matching_locations)
        is_surface = any(loc.is_surface for loc in matching_locations)
        has_gems = any(loc.has_gems for loc in matching_locations)
        has_rocks = any(loc.has_rocks for loc in matching_locations)

        raw_data = [loc.raw_data for loc in matching_locations]

        return LocationInfo(
            kind=location.kind,
            name=name,
            aliases=aliases,
            id=location.id,
            uex_id=uex_id,
            uex_type=uex_type,
            code=location.code,
            gravity_well_type=gravity_well_type,
            is_gravity_well=is_gravity_well,
            is_navigable=is_navigable,
            is_space=is_space,
            is_surface=is_surface,
            has_gems=has_gems,
            has_rocks=has_rocks,
            raw_data=raw_data
        ), matching_locations
    
    def _filter_valid(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        show_quantum_missing = self._config.get("location_provider.show_quantum_missing", False)

        def qt_filter(item: dict[str, Any]) -> bool:
            return show_quantum_missing or item.get("has_quantum_marker", True)
            
        return [
            item for item in data
            if item["is_available_live"] and item["is_available"] and item['is_visible'] and qt_filter(item)
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
        search_key = NormalizedName(target)
        target_canonical = self._canonical_name_processor.process(target)

        # Build the comparison list excluding the exact same string (case-insensitive) instances
        names = [
            self._canonical_name_processor.process(loc.name)
            for loc in locations
            if loc.name is not None and not search_key.matches(loc.name)
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
        for i in range(1, len(target_canonical) + 1):
            prefix = target_canonical[:i]
            if not any(any_word_prefix(name, prefix) for name in names):
                return prefix

        # Check if the full target itself is unique as a prefix at any word boundary
        if not any(any_word_prefix(name, target_canonical) for name in names):
            return target_canonical

        return None